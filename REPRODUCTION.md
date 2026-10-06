# Reproduction Guide

## Pipeline mapping

| Paper/pipeline step | Implementation |
| --- | --- |
| DINOv2 with four register tokens | `src/effovpr/models/dino_backbone.py`, `src/effovpr/models/dino_attention.py` |
| Normalized CLS global descriptor and optional projection | `src/effovpr/models/effovpr.py`, `src/effovpr/models/projection.py` |
| CosFace and final-five-block fine-tuning | `src/effovpr/models/cosface.py`, `src/effovpr/models/dino_backbone.py` |
| Dataset audit and duplicate-safe deterministic split | `src/effovpr/data/` |
| FAISS global retrieval | `src/effovpr/retrieval/` |
| Intermediate Q/K/V and CLS attention selection | `src/effovpr/models/effovpr.py` |
| T1 filtering, V local descriptors, MNN/T2, stable reranking | `src/effovpr/reranking/` |
| Label-based metrics and evaluation | `src/effovpr/evaluation/`, `scripts/evaluate.py` |

The implementation concatenates all attention-head Q/K/V dimensions back to the backbone's 1024-dimensional feature space. CLS and four register tokens are excluded from image-patch descriptors. Candidate reranking sorts by MNN count and uses the original global rank as a deterministic tie-breaker.

## Dataset interpretation

The source is `PhamPhien/VN_Attractions`, loaded from its `train` split. The experiment treats each `label` as a place/class identity. The data does not provide GPS coordinates, so evaluation is same-label retrieval, **not** 25-meter localization and not geographic Recall@K.

The split proportions are 70% train, 10% gallery, 10% validation queries, and 10% test queries. Exact decoded-image duplicates are grouped into a single split. A manifest records the seed, dataset ID digest, ratios, and CSV digests; changed inputs regenerate split files.

## Training and checkpoints

Training uses AdamW, CosFace, the last five transformer blocks, and configurable 1024/256/128-dimensional projection. Epochs do not run validation inference, gallery feature extraction, FAISS retrieval, or local reranking. The full `artifacts/checkpoints/best.pth` is selected by minimum mean training loss, while `last.pth` remains resumable; training history, class mapping, dataset audit, and run metadata are recorded in `artifacts/`. After training, the full best state is exported and verified under `artifacts/model/` as separate `config.json`, `model.safetensors`, and `README.md` files. Final evaluation remains an explicit post-training step.

No checkpoint or training result is included unless it was produced by an actual run. CPU training of the full ViT-L model is expected to be resource-intensive; use a smaller `--epochs` run only as a limited experiment, not a complete result.

## Evaluation

Run each mode with:

```powershell
python -m scripts.evaluate --mode dino-zs
python -m scripts.evaluate --mode effovpr-zs
python -m scripts.evaluate --mode effovpr-g --checkpoint artifacts/checkpoints/best.pt
python -m scripts.evaluate --mode effovpr-r --checkpoint artifacts/checkpoints/best.pt
```

Output includes micro and macro Recall@1/5/10, per-class Recall@K, MRR, per-query global and reranked rankings, feature dimension/storage, and measured extraction/search/reranking time. `--split val_queries` selects validation queries; the default is the held-out test queries. Optional CLI controls include `--top-k`, `--layer`, `--facet`, `--t1`, `--t2`, `--resolution`, and `--limit`. A limited run truncates gallery and query inputs and is explicitly marked `LIMITED EXPERIMENT`.

Compatible normalized global features are cached below `artifacts/features/global/` with metadata for checkpoint identity, model revision, resolution, split IDs, feature dimension, local layer, facet, T1, and T2. Local features remain on-demand and are not precomputed for the full gallery.

## Ablations, benchmarks, and visuals

`scripts/run_ablations.py` runs K, layer, facet, T1, T2, and resolution sweeps. Compact dimensions require a separately trained matching checkpoint; provide it as `--compact-checkpoint 256=PATH` and/or `--compact-checkpoint 128=PATH`. Missing dimensions are explicitly marked `NOT EXECUTED`.

`scripts/benchmark.py` reports warmup/repeated mean, median, and p95 stage timings plus model, feature/index, local-descriptor, and GPU memory metrics when CUDA is available. CPU runs accurately report GPU memory as unavailable. `scripts/visualize.py` saves global/reranked top-10 panels, selected patches, top-1 MNN matches, and attention heatmaps for evaluated queries.

## Commands

```powershell
python -m scripts.audit_dataset --config configs/effovpr_vn_attractions.yaml
python -m scripts.prepare_data --seed 42
python -m scripts.train --config configs/effovpr_vn_attractions.yaml
python -m scripts.build_gallery_index --config configs/effovpr_vn_attractions.yaml --checkpoint artifacts/checkpoints/best.pt
python -m scripts.evaluate --config configs/effovpr_vn_attractions.yaml --checkpoint artifacts/checkpoints/best.pt --mode effovpr-r
python -m scripts.run_ablations --config configs/effovpr_vn_attractions.yaml --checkpoint artifacts/checkpoints/best.pt
python -m scripts.benchmark --config configs/effovpr_vn_attractions.yaml --mode effovpr-r --checkpoint artifacts/checkpoints/best.pt
python -m scripts.visualize --config configs/effovpr_vn_attractions.yaml --mode effovpr-r --checkpoint artifacts/checkpoints/best.pt
python -m scripts.smoke_test --config configs/effovpr_vn_attractions.yaml
python -m scripts.run_experiments --config configs/effovpr_vn_attractions.yaml
python -m pytest -q tests
```

Run an orchestration subset with, for example:

```powershell
python -m scripts.run_experiments --stages audit split smoke
```

## Limitations and result integrity

- No GPS metadata is available; the original geographic cell labels and 25-meter localization metric cannot be reproduced.
- This project reports label-based retrieval for VN_Attractions.
- Training/evaluation at the specified scale is compute-intensive; this environment currently has CPU-only PyTorch, so full training and high-resolution reranking may be impractical.
- Hugging Face dataset/model loading requires access to Hugging Face and enough local disk. Resource files are cached under `artifacts/cache/huggingface/`.
- All reported metrics, timings, and dataset statistics must come from successful runs. A smoke test, limited experiment, unexecuted stage, or failed run is not a full experiment.
