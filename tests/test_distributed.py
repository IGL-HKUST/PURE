import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.skipif(
    os.environ.get("PURE_TEST_DDP") != "1",
    reason="Enable PURE_TEST_DDP=1 for the two-rank test",
)
def test_two_process_accumulation_and_resume(tmp_path, monkeypatch):
    # A file rendezvous avoids the PyTorch 2.2 / Python 3.12 macOS TCPStore
    # crash. The actual DDP gradients still communicate via Gloo sockets.
    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    worker = Path(__file__).with_name("ddp_worker.py")
    processes = []
    handles = []
    try:
        for rank in range(2):
            handle = (tmp_path / f"rank{rank}.log").open("w")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(worker), str(tmp_path), str(rank)],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        for process in processes:
            code = process.wait(timeout=120)
            assert code == 0, "\n".join(
                p.read_text() for p in tmp_path.glob("rank*.log")
            )
        assert (tmp_path / "success").exists()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
