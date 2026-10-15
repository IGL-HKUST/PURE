"""Download pinned examples and run the shared reconstruction CLI sequentially."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'examples/reconstruction/sources.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(info, destination):
    """Verify cached files; replace only a fully verified temporary download."""
    if destination.exists():
        if destination.stat().st_size != info['bytes'] or sha256(destination) != info['sha256']:
            raise ValueError(f'Existing sample differs from the manifest: {destination}')
        print(f'Cached and verified: {destination}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
            with urlopen(info['url'], timeout=60) as response:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        if temporary.stat().st_size != info['bytes'] or sha256(temporary) != info['sha256']:
            raise ValueError(f'Sample size or SHA256 mismatch: {destination.name}')
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Downloaded and verified: {destination}', flush=True)


def build_plan(args, manifest):
    cases = manifest['cases'] if args.sample == 'all' else [x for x in manifest['cases'] if x['name'] == args.sample]
    models = ['pi3', 'da3', 'vggt'] if args.model == 'all' else [args.model]
    downloads, commands = [], []
    for case in cases:
        name = case['name']
        selected = [x for x in manifest['files'] if x['path'] == f'examples/{name}.mp4' or x['path'].startswith(f'examples/{name}/')]
        paths = []
        for info in selected:
            relative = Path(info['path']).relative_to('examples')
            if '..' in relative.parts or relative.is_absolute():
                raise ValueError('Invalid sample path')
            target = args.inputs / relative
            paths.append(str(target))
            downloads.append((info, target))
        for model in models:
            output = args.output / f'{name}-{model}'
            command = [sys.executable, str(ROOT / 'scripts/demo/reconstruct.py'), '--pretrained', args.pretrained,
                       '--subfolder', model, '--frames', str(min(args.frames, case['frame_count'])),
                       '--output', str(output), '--device', args.device, '--precision', args.precision]
            command += ['--video', paths[0]] if case['kind'] == 'video' else ['--images', *paths]
            if args.keep_sky:
                command.append('--keep-sky')
            commands.append((command, output))
    return downloads, commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', choices=['motorcycle', 'penguin', 'sport', 'drift-straight', 'all'], default='motorcycle')
    parser.add_argument('--model', choices=['pi3', 'da3', 'vggt', 'all'], default='pi3')
    parser.add_argument('--pretrained', default=os.environ.get('PURE_MODEL_REPO', 'cyun9286/PURE'))
    parser.add_argument('--inputs', type=Path, default=ROOT / 'examples')
    parser.add_argument('--output', type=Path, default=Path('outputs/demo'))
    parser.add_argument('--frames', type=int, default=48)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    parser.add_argument('--keep-sky', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print downloads and commands without network, file writes or GPU work')
    mode.add_argument('--download-only', action='store_true', help='Download and verify sample inputs without model loading')
    args = parser.parse_args()
    if not 2 <= args.frames <= 48:
        parser.error('--frames must be between 2 and 48')
    downloads, commands = build_plan(args, json.loads(MANIFEST.read_text()))
    if args.dry_run:
        for info, target in downloads:
            print(json.dumps({'download': info['url'], 'destination': str(target), 'sha256': info['sha256']}))
        for command, _ in commands:
            print(json.dumps({'command': command}))
        return
    if not args.download_only:
        for _, output in commands:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                parser.error(f'Choose an empty output directory: {output}')
    for info, target in downloads:
        download(info, target)
    if not args.download_only:
        for command, _ in commands:
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
