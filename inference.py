# inference.py
import asyncio
import json
import os
from typing import List

from openai import OpenAI
import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
TASKS = os.environ.get("TASKS", "easy,medium,hard")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.7"))
MAX_TOTAL_REWARD = float(os.environ.get("MAX_TOTAL_REWARD", "1.0"))
SEED = os.environ.get("SEED")


def _parse_seed(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _sanitize_field(value: object) -> str:
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(text.split())


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    safe_action = _sanitize_field(action)
    err = "null" if error is None else _sanitize_field(error)
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_str} error={err}",
        flush=True,
    )


def log_end(success, steps, rewards):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def get_model_message(client: OpenAI, observation: dict, history: List[str]) -> str:
    prompt = f"""
You are debugging a PyTorch training job. Respond ONLY with valid JSON matching this exact schema:
{{
  "current_hypothesis": {{"bug_type": "<string>", "affected_file": "<string>", "confidence": <0.0-1.0>}},
  "investigation_action": {{"action": "reveal_file", "target": "<filename>"}},
  "commit_diagnosis": false,
  "final_diagnosis": null
}}

Valid action types: reveal_file, extend_loss_curve, extend_gpu_profile, reveal_log_chunk, run_diagnostic
Valid bug types: missing_zero_grad, data_leakage, memory_leak, learning_rate_too_high, gradient_explosion, wrong_loss_function, amp_overflow

Observation:
{json.dumps(observation)[:8000]}
History: {history}
"""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    return (completion.choices[0].message.content or "").strip()


async def _run_task(task: str, client: OpenAI) -> None:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    seed_value = _parse_seed(SEED)

    log_start(task=task, env="pytorch-debug-env", model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(timeout=60.0) as session:
            reset_params = {"task_id": task}
            if seed_value is not None:
                reset_params["seed"] = seed_value
            reset_resp = await session.post(f"{ENV_URL}/reset", params=reset_params)
            reset_resp.raise_for_status()
            result = reset_resp.json()

            session_id = result.get("session_id")
            observation = result.get("observation")
            if not session_id:
                raise RuntimeError("Missing session_id in reset response")
            if observation is None:
                raise RuntimeError("Missing observation in reset response")

            for step in range(1, MAX_STEPS + 1):
                if result.get("done"):
                    break

                action_text = "null"
                try:
                    action_text = get_model_message(client, observation, history)
                except Exception as exc:
                    reward = 0.0
                    done = True
                    error = f"model_error: {exc}"
                    rewards.append(reward)
                    steps_taken = step
                    log_step(step=step, action=action_text, reward=reward, done=done, error=error)
                    break

                try:
                    action_json = json.loads(action_text)
                    step_resp = await session.post(
                        f"{ENV_URL}/step",
                        params={"session_id": session_id},
                        json=action_json,
                    )
                    step_resp.raise_for_status()
                    result = step_resp.json()
                    reward = result.get("reward", 0.0)
                    done = result.get("done", False)
                    error = result.get("error")
                    observation = result.get("observation", observation)
                except Exception as exc:
                    reward = 0.0
                    done = True
                    error = f"step_error: {exc}"

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_text, reward=reward, done=done, error=error)
                history.append(f"step={step} reward={reward:.3f}")

                if done:
                    break
    except Exception:
        pass

    score = min(max(rewards[-1] if rewards else 0.0, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, rewards=rewards)


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = [task.strip() for task in TASKS.split(",") if task.strip()]

    for task in tasks:
        await _run_task(task, client)


if __name__ == "__main__":
    asyncio.run(main())
