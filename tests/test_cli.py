import subprocess
import sys

import pytest

from fpextract.cli import build_parser

SUBCOMMANDS = ["extract", "aggregate", "classify", "try-one", "try-crop", "try-norm",
               "dq-profile", "dq-clean", "sync-pg"]


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_subcommand_help(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args([cmd, "--help"])
    assert e.value.code == 0
    assert cmd in capsys.readouterr().out


def test_module_entrypoint_version():
    out = subprocess.run([sys.executable, "-m", "fpextract", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and "fpextract" in out.stdout


def test_try_norm_runs_without_network(samples_dir, tmp_path):
    pytest.importorskip("cv2")
    img = samples_dir / "harbour-view" / "harbour-view-type-a1.png"
    out = subprocess.run([sys.executable, "-m", "fpextract", "try-norm", str(img), "--no-browser",
                          "--out", str(tmp_path / "n.html"), "--out-image", str(tmp_path / "n.png")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "n.html").stat().st_size > 1000 and (tmp_path / "n.png").exists()


def test_extract_refuses_batch_with_json_mode(samples_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    out = subprocess.run([sys.executable, "-m", "fpextract", "extract", "--input", str(samples_dir),
                          "--output", str(tmp_path / "e.jsonl"), "--batch", "--mode", "json"],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "--batch requires --mode tool" in out.stderr
