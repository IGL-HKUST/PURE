"""Two-process integration worker; launched by test_distributed.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pure.training.cache import ClipDataset, write_cache
from pure.training.checkpoint import read_checkpoint
import pure.training.engine as engine
from examples.reference_training_smoke import TinyGeometry, make_synthetic_clip


def main():
    torch.set_num_threads(1)
    from datetime import timedelta

    root = Path(sys.argv[1])
    rank = int(sys.argv[2])
    root.mkdir(parents=True, exist_ok=True)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        if rank == 0:
            write_cache(root / "data/clip.npz", make_synthetic_clip(), synthetic=True)
        dist.barrier()
        dataset = ClipDataset(root / "data", allow_synthetic=True)
        config = {"steps": 3, "accumulation": 2, "lr": 1e-4, "seed": 7, "save_every": 1}
        complete = DistributedDataParallel(TinyGeometry())
        final = engine.fit(complete, dataset, config, root / "full")
        original = engine.optimizer_step
        calls = 0

        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InterruptedError(
                    "Test interruption at a synchronized step boundary"
                )
            return original(*args, **kwargs)

        engine.optimizer_step = interrupt
        try:
            engine.fit(
                DistributedDataParallel(TinyGeometry()),
                dataset,
                config,
                root / "partial",
            )
        except InterruptedError:
            pass
        finally:
            engine.optimizer_step = original
        dist.barrier()
        restored = DistributedDataParallel(TinyGeometry())
        engine.fit(
            restored,
            dataset,
            config,
            root / "partial",
            resume=root / "partial/latest.pt",
        )
        dist.barrier()
        for key, value in complete.module.state_dict().items():
            torch.testing.assert_close(
                restored.module.state_dict()[key], value, rtol=2e-6, atol=2e-8
            )
        try:
            engine.fit(restored, dataset, config, root / "full")
        except ValueError as error:
            assert "not empty" in str(error)
        else:
            raise AssertionError("Existing run was not protected")
        completed = engine.fit(
            restored, dataset, config, root / "already-complete", resume=final
        )
        assert completed == final and completed.is_file()
        for key, value in complete.module.state_dict().items():
            torch.testing.assert_close(restored.module.state_dict()[key], value)
        if rank == 0:
            state = read_checkpoint(final)
            assert state["world_size"] == 2 and len(state["rank_states"]) == 2
            assert not torch.equal(
                state["rank_states"][0]["rng"], state["rank_states"][1]["rng"]
            )
            (root / "success").write_text(
                "DDP accumulation, per-rank resume, output guard and completed resume passed\n"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
