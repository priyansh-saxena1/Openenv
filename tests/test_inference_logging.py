from inference import log_end, log_start, log_step


def test_log_start_format(capsys):
    log_start(task="easy", env="pytorch-debug-env", model="test-model")
    out = capsys.readouterr().out.strip()
    assert out == "[START] task=easy env=pytorch-debug-env model=test-model"


def test_log_step_sanitizes_fields(capsys):
    log_step(
        step=1,
        action="line1\nline2",
        reward=0.0,
        done=False,
        error="bad\nerr",
    )
    out = capsys.readouterr().out.strip()
    assert "\n" not in out
    assert "action=line1 line2" in out
    assert "error=bad err" in out
    assert "done=false" in out


def test_log_end_format(capsys):
    log_end(success=True, steps=3, rewards=[0.0, 0.1, 1.0])
    out = capsys.readouterr().out.strip()
    assert out == "[END] success=true steps=3 rewards=0.00,0.10,1.00"
