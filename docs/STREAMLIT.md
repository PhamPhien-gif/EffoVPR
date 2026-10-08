# EffoVPR Streamlit comparison app

## Start locally

Run from the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## What it does

- Loads both `Models/EffoVPR_G/` and `Models/EffoVPR_R/` and runs the same query image through each.
- Shows each model's CosFace place prediction, the most similar individual gallery images, and the most likely locations grouped by label.
- Provides separate Top 1, 5, or 10 controls for the similar images and likely locations. Location scores are the mean cosine similarity of that label's best three gallery images; Softmax percentages are relative across all gallery labels.
- Optionally reranks each model's Top 10 with its own local MNN descriptors.
- Accepts images from `Images/` or uploaded files.

Each model uses its own `index/gallery_features.npy`, `gallery_metadata.parquet`, and `index_config.json`. `gallery.index` is not needed by this app. Keep each gallery bundle paired with the exact checkpoint used to create it. The CosFace score is a ranking score, not a calibrated probability. Location Softmax percentages use temperature 0.05 and are relative scores, not calibrated probabilities. The location percentages are computed across every distinct label in that model's gallery; changing the display dropdown only changes how many rows are shown. When MNN is enabled, it reranks only the similar-image list's Top 10 candidates; the location ranking remains based on cosine similarity.

The app can show IDs, labels, and scores from the local indexes without Hugging Face. Loading gallery thumbnails or running MNN requires first-time access to `PhamPhien/VN_Attractions`; disable those options in the sidebar to avoid loading gallery images.
