# Demo

Three videos and ten drift-straight images from [SelfEvo](https://github.com/Self-Evo/SelfEvo)
are bundled here. [sources.json](reconstruction/sources.json) pins the upstream
revision and records every original URL, size and SHA256.

Run from the repository root:

```bash
pip install -e '.[inference]'
bash scripts/demo/run_demo.sh --sample motorcycle --model pi3
bash scripts/demo/run_demo.sh --sample all --model all
```

Select `motorcycle`, `penguin`, `sport` or `drift-straight` with `--sample`,
and `pi3`, `da3` or `vggt` with `--model`. Video runs use the first 48 consecutive
frames; drift-straight uses all ten supplied images.

Results go to `outputs/demo/<sample>-<model>/`. Use a new `--output` directory
for another run, or `--dry-run` to inspect commands. [Runner options](../scripts/README.md).
