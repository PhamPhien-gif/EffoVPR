# EffoVPR Streamlit comparison app

## Start locally

Run from the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

Install the dependencies inside the same virtual environment used to launch Streamlit. If logs report `No module named 'torchvision'`, run `python -m pip install -r requirements.txt` from that environment and restart Streamlit; PyTorch and Torchvision need compatible builds.

## What it does

- Loads both `Models/EffoVPR_G/` and `Models/EffoVPR_R/` and runs the same query image through each.
- Shows each model's CosFace place prediction, the most similar individual gallery images, and the most likely locations grouped by label.
- Provides separate Top 1, 5, or 10 controls for the similar images and likely locations. Location scores are the mean cosine similarity of that label's best three gallery images; Softmax percentages are relative across all gallery labels.
- Optionally reranks EffoVPR_R's Top 10 with local MNN descriptors and visualizes the query-to-Top-1 MNN correspondences. EffoVPR_G remains ranked by cosine similarity.
- Accepts images from `Images/` or uploaded files.
- In `Upload images`, accepts pasted clipboard images with Ctrl+V as well as file selection.

To paste a query image, focus the paste field in the sidebar, press Ctrl+V, then press Enter. The image is queued until you clear it; click **Compare EffoVPR_G and EffoVPR_R** to run the demo. Clipboard paste requires a supported browser and a secure context; `localhost` works for local use.

Each model uses its own `index/gallery_features.npy`, `gallery_metadata.parquet`, and `index_config.json`. `gallery.index` is not needed by this app. Keep each gallery bundle paired with the exact checkpoint used to create it. The CosFace score is a ranking score, not a calibrated probability. Location Softmax percentages use temperature 0.05 and are relative scores, not calibrated probabilities. The location percentages are computed across every distinct label in that model's gallery; changing the display dropdown only changes how many rows are shown. When MNN is enabled, only EffoVPR_R's similar-image Top 10 is reranked; EffoVPR_G and both models' location rankings remain based on cosine similarity.

The MNN visualization appears below EffoVPR_R's similar-image table. It shows the query on the left and the reranked Top-1 gallery image on the right; colored lines connect mutual nearest patch descriptors that pass `T2`. The caption reports the number of selected patches and accepted MNN pairs. If no pair passes `T2`, the app says so and draws no lines.

The app can show IDs, labels, and scores from the local indexes without Hugging Face. Loading gallery thumbnails or running MNN requires first-time access to `PhamPhien/VN_Attractions`; disable those options in the sidebar to avoid loading gallery images.
