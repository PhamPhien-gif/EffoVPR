# EffoVPR on VN_Attractions

A visual retrieval project built around DINOv2 ViT-L/14 with registers. The repository is centered on Kaggle notebooks for training/evaluation and a local Streamlit app for comparing the exported EffoVPR_G and EffoVPR_R models.

VN_Attractions has place labels but no GPS coordinates. The task is same-label image retrieval; it does not reproduce the paper's 25-meter geolocation protocol.

## Local Streamlit app

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app loads model and gallery bundles from `Models/EffoVPR_G/` and `Models/EffoVPR_R/`, and example queries from `data/examples/`. It imports inference and reranking code from the root `src/` package. See [the Streamlit guide](docs/STREAMLIT.md).

`notebooks/inference/Use_Model.ipynb` is a separate local notebook for predicting labels for images in `data/examples/`; it does not power the Streamlit app.

## Kaggle notebooks

The root-level notebooks are the training and evaluation workflows. The main notebooks embed their required `src/` and `scripts/` modules in `%%writefile` cells, so Kaggle can recreate those modules in its runtime. See [the Kaggle workflow guide](docs/REPRODUCTION.md).

## Project map

- `app.py`, `Models/`, `data/examples/`: local Streamlit app, model/gallery bundles, and demo query images.
- `notebooks/training/`: Kaggle training and evaluation workflows.
- `notebooks/inference/Use_Model.ipynb`: local label-prediction notebook.
- Root `src/`: shared model, artifact-loading, retrieval, and reranking code used by the app and local notebook.
- `evaluation/results/`: saved summary JSON results, intentionally retained.
- `docs/`: usage guides, implementation specification, and paper.
- `scripts/`, `tests/`, `artifacts/`, and old `*_outputs/` folders have been removed from this working copy. Kaggle notebooks embed pipeline code; a local pipeline run can recreate `artifacts/` only if its corresponding source is restored.
