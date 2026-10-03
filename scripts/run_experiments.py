from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


STAGES = (
    "audit", "split", "smoke", "dino-zs", "effovpr-zs", "train", "index",
    "effovpr-g", "effovpr-r", "ablations", "benchmark", "visualization", "report",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured EffoVPR experiment pipeline.")
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--stop-on-error", action="store_true", default=True)
    args = parser.parse_args()

    commands = {
        "audit": [sys.executable, "-m", "scripts.audit_dataset", "--config", args.config],
        "split": [sys.executable, "-m", "scripts.prepare_data", "--config", args.config, "--seed", str(args.seed)],
        "smoke": [sys.executable, "-m", "scripts.smoke_test", "--config", args.config],
        "dino-zs": [sys.executable, "-m", "scripts.evaluate", "--config", args.config, "--mode", "dino-zs"],
        "effovpr-zs": [sys.executable, "-m", "scripts.evaluate", "--config", args.config, "--mode", "effovpr-zs"],
        "train": [sys.executable, "-m", "scripts.train", "--config", args.config, "--seed", str(args.seed)],
        "index": [sys.executable, "-m", "scripts.build_gallery_index", "--config", args.config, "--checkpoint", args.checkpoint],
        "effovpr-g": [sys.executable, "-m", "scripts.evaluate", "--config", args.config, "--checkpoint", args.checkpoint, "--mode", "effovpr-g"],
        "effovpr-r": [sys.executable, "-m", "scripts.evaluate", "--config", args.config, "--checkpoint", args.checkpoint, "--mode", "effovpr-r"],
        "ablations": [sys.executable, "-m", "scripts.run_ablations", "--config", args.config, "--checkpoint", args.checkpoint],
        "benchmark": [sys.executable, "-m", "scripts.benchmark", "--config", args.config, "--checkpoint", args.checkpoint, "--mode", "effovpr-r"],
        "visualization": [sys.executable, "-m", "scripts.visualize", "--config", args.config, "--checkpoint", args.checkpoint],
        "report": None,
    }
    if args.epochs is not None:
        commands["train"].extend(("--epochs", str(args.epochs)))

    output_path = Path("artifacts/orchestration")
    output_path.mkdir(parents=True, exist_ok=True)
    run_state = {
        "status": "IN PROGRESS",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_stages": args.stages,
        "stages": [],
    }
    try:
        for stage in args.stages:
            if stage == "report":
                report = {
                    "status": "COMPLETED",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "evaluation_results": {},
                }
                for mode in ("dino-zs", "effovpr-zs", "effovpr-g", "effovpr-r"):
                    path = Path("artifacts/evaluation") / mode / "results.json"
                    if path.is_file():
                        report["evaluation_results"][mode] = json.loads(path.read_text(encoding="utf-8"))
                (Path("artifacts") / "experiment_report.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                run_state["stages"].append({"name": stage, "status": "COMPLETED"})
                continue
            command = commands[stage]
            if stage in {"index", "effovpr-g", "effovpr-r", "ablations", "benchmark", "visualization"} and not Path(args.checkpoint).is_file():
                raise FileNotFoundError(f"Stage {stage} requires checkpoint {args.checkpoint}")
            started = time.perf_counter()
            completed = subprocess.run(command, check=False)
            entry = {
                "name": stage,
                "command": command,
                "status": "COMPLETED" if completed.returncode == 0 else "FAILED",
                "return_code": completed.returncode,
                "elapsed_seconds": time.perf_counter() - started,
            }
            run_state["stages"].append(entry)
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, command)
        run_state["status"] = "COMPLETED"
    except Exception as error:
        run_state["status"] = "FAILED"
        run_state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run_state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (output_path / "run.json").write_text(
            json.dumps(run_state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(run_state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
