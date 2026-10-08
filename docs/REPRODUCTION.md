# Kaggle workflow guide

## Where the executable workflows live

Use the root-level Jupyter notebooks for Kaggle training and evaluation. The main Kaggle notebooks include `%%writefile` cells that recreate the source package and command modules inside the Kaggle runtime; they do not rely on this checkout's removed `scripts/` directory.

- `kaggle_run_all.ipynb`: main end-to-end run.
- `EffoVPR_G.ipynb` and `EffoVPR_R.ipynb`: fine-tuned global and local-reranking experiments.
- `EffoVPR_ZS.ipynb` and `DINOv2_Global_ZS.ipynb`: zero-shot experiments.
- `EffoVPR_R_Evaluation_Kaggle.ipynb`: standalone evaluation of EffoVPR-R.
- `Kaggle_FineTune_DINOv2.ipynb`: an earlier standalone fine-tuning workflow.

Keep Kaggle input datasets and outputs attached to the corresponding notebook. Local output directories were removed from this checkout; `evaluation/` retains the compact result JSON files.

## Local inference

For the Streamlit comparison app, see [the Streamlit guide](STREAMLIT.md). `Use_Model.ipynb` is a separate notebook that predicts labels for images under `Images/` using the local EffoVPR_G bundle.

## Research reference

The [implementation specification](EFFOVPR_IMPLEMENTATION_SPEC.md) records the detailed model and evaluation design. It includes examples from the earlier local command-line pipeline; those `scripts.*` commands are not available in this cleaned checkout. The [EffoVPR paper](references/EffoVPR.pdf) is retained alongside it.
