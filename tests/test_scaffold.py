import os
from pathlib import Path
import subprocess
import sys


def test_package_import_is_inert():
    result = subprocess.run(
        [sys.executable, "-c", "import head_sharing; assert head_sharing.__all__ == []"],
        env={**os.environ, "PYTHONPATH": "src"}, check=True, capture_output=True, text=True,
        timeout=5,
    )
    assert not result.stdout


def test_native_config_is_confined_and_cpu_only():
    import torch
    from ultralytics.utils import USER_CONFIG_DIR

    expected = (Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").resolve(strict=True)
    assert USER_CONFIG_DIR.resolve(strict=True) == expected
    assert expected.is_relative_to(Path(os.environ["HEAD_SHARING_TEST_ROOT"]).resolve())
    assert torch.get_num_threads() == 1
    assert not torch.cuda.is_available()


def test_wrapper_rejects_unowned_root():
    result = subprocess.run([sys.executable, "scripts/check_cpu.py", "--seconds", "1", "--"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "pytest path" in result.stderr


def test_wrapper_reaps_lingering_child_after_success(tmp_path):
    script = Path("scripts/check_cpu.py").resolve()
    marker = tmp_path / "child-survived"
    test_file = tmp_path / "test_child.py"
    child_code = f"import time; time.sleep(1); open({str(marker)!r}, 'w').close()"
    test_file.write_text(
        "import subprocess, sys\ndef test_spawn():\n"
        f"    subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--seconds", "5", "--", str(test_file)],
        capture_output=True, text=True, timeout=8,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert not marker.exists()


def test_wrapper_bounds_child_and_reaps_timeout(tmp_path):
    script = Path("scripts/check_cpu.py").resolve()
    slow_test = tmp_path / "test_slow.py"
    slow_test.write_text("import time\ndef test_slow(): time.sleep(10)\n")
    result = subprocess.run(
        [sys.executable, str(script), "--seconds", "0.2", "--", str(slow_test)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 124
    assert "owned process group reaped" in result.stderr
