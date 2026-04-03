# inference.py
import asyncio
import json
import os
from typing import List

from openai import OpenAI
import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
TASK_NAME = os.environ.get("TASK_NAME", "easy")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.7"))
MAX_TOTAL_REWARD = float(os.environ.get("MAX_TOTAL_REWARD", "1.0"))


def log_start(task, env, model):
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
    }), flush=True)


def log_step(step, action, reward, done, error):
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": float(reward),
        "done": bool(done),
        "error": error,
    }), flush=True)


def log_end(success, steps, score, rewards):
    print(json.dumps({
        "type": "END",
        "success": bool(success),
        "steps": steps,
        "score": float(score),
        "rewards": [float(r) for r in rewards],
    }), flush=True)


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
Valid bug types: missing_zero_grad, data_leakage, memory_leak, learning_rate_too_high, gradient_explosion

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


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards = []
    history = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env="pytorch-debug-env", model=MODEL_NAME)

    async with httpx.AsyncClient(timeout=60.0) as session:
        reset_resp = await session.post(f"{ENV_URL}/reset", params={"task_id": TASK_NAME})
        reset_resp.raise_for_status()
        result = reset_resp.json()
        session_id = result.get("session_id")
        observation = result["observation"]

        for step in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            action_text = get_model_message(client, observation, history)
            try:
                action_json = json.loads(action_text)
                step_resp = await session.post(f"{ENV_URL}/step", params={"session_id": session_id}, json=action_json)
                step_resp.raise_for_status()
                result = step_resp.json()
                reward = result.get("reward", 0.0)
                done = result.get("done", False)
                error = None
                observation = result["observation"]
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_text, reward=reward, done=done, error=error)
            history.append(f"step={step} reward={reward:.3f}")

            if done:
                break

    score = min(max(rewards[-1] if rewards else 0.0, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
