from __future__ import annotations

import argparse
import json

from src.effovpr.data.audit import compute_dataset_audit, save_dataset_audit
from scripts.common import get_dataset, load_config, write_run_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = get_dataset(config)
    stats = compute_dataset_audit(dataset)
    save_dataset_audit(stats, args.output_dir)
    write_run_metadata(f"{args.output_dir}/run_metadata.json", config, None, 42)
    print(json.dumps(stats, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
