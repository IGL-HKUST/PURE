"""Exercise the batch entry point without downloading weights or requiring CUDA."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.demo import run_examples


def test_batch_dry_run_is_side_effect_free(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / 'scripts/demo/run_examples.py'), '--sample', 'all',
         '--model', 'all', '--dry-run'], cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    records = [json.loads(line) for line in result.stdout.splitlines()]
    commands = [x['command'] for x in records if 'command' in x]
    assert len(commands) == 12
    assert len([x for x in records if 'download' in x]) == 13
    assert list(tmp_path.iterdir()) == []
    for command in commands:
        if '--images' in command:
            assert command[command.index('--frames') + 1] == '10'
            images = command[command.index('--images') + 1:]
            assert len(images) == 10
            assert [Path(x).stem for x in images] == [f'{i:05d}' for i in range(0, 50, 5)]
        else:
            assert command[command.index('--frames') + 1] == '48'
        assert '--keep-sky' not in command


def test_download_verifies_before_publishing_and_reuses_cache(tmp_path, monkeypatch):
    content = b'example video bytes'
    info = {'url': 'https://example.invalid/sample.mp4', 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}
    target = tmp_path / 'inputs/sample.mp4'
    monkeypatch.setattr(run_examples, 'urlopen', lambda *a, **k: io.BytesIO(content))
    run_examples.download(info, target)
    assert target.read_bytes() == content
    def no_network(*args, **kwargs):
        raise AssertionError('Cached sample must not be downloaded again')
    monkeypatch.setattr(run_examples, 'urlopen', no_network)
    run_examples.download(info, target)
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Existing sample differs'):
        run_examples.download(info, target)
    assert target.read_bytes() == b'corrupt'


def test_bad_download_leaves_no_sample_or_partial(tmp_path, monkeypatch):
    target = tmp_path / 'sample.mp4'
    monkeypatch.setattr(run_examples, 'urlopen', lambda *a, **k: io.BytesIO(b'bad'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        run_examples.download({'url': 'https://example.invalid/x', 'bytes': 3,
                               'sha256': hashlib.sha256(b'good').hexdigest()}, target)
    assert list(tmp_path.iterdir()) == []
