from contextlib import redirect_stdout
from io import StringIO

from calender_en import inference


def test_inference_runs_end_to_end_without_crashing() -> None:
    output = StringIO()

    with redirect_stdout(output):
        inference.main()

    rendered = output.getvalue()
    lines = rendered.strip().splitlines()
    assert len(lines) == 6
    assert lines[0] == "[START] task=smart_calendar_resolution env=calender_en model=deterministic-baseline"
    assert lines[-1] == "[END] success=true steps=4 rewards=1.00,1.50,4.00,3.00"
    assert all(line.startswith("[STEP]") for line in lines[1:5])


def test_inference_output_is_deterministic() -> None:
    first = StringIO()
    second = StringIO()

    with redirect_stdout(first):
        inference.main()
    with redirect_stdout(second):
        inference.main()

    assert first.getvalue() == second.getvalue()
