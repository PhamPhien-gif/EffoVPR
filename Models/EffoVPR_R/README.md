# EffoVPR

Full best-checkpoint weights, including frozen backbone, projection and classifier.
Architecture lives in the project Python source. No pretrained download is needed.
Use transformers 5.18.0, PyTorch and safetensors.

```python
from src.effovpr.utils.model_artifact import load_model_artifact, prepare_image
import torch
model, config = load_model_artifact('artifacts/model')
pixels = prepare_image(model, '/path/query.jpg', config['inference']['resolution'])
with torch.inference_mode():
    embedding = model.extract_global(pixels)
    qkv = model.extract_qkv(pixels, config['reranking']['layer'])
```

`notebooks/inference/Use_Model.ipynb` predicts labels for images in `data/examples/` using the EffoVPR_G model bundle; use this model bundle in the Streamlit comparison app.
Download config.json and model.safetensors as separate files; do not zip the model.

Verification against best checkpoint:
```json
{
  "global_shape": [
    1,
    1024
  ],
  "max_absolute_difference": 0.0,
  "cosine_similarity": 1.0,
  "local_patch_shape": [
    1,
    1296,
    1024
  ],
  "selected_local_count": 2,
  "local_max_absolute_difference": 0.0,
  "strict_load": true,
  "mnn_matches_equal": true
}
```