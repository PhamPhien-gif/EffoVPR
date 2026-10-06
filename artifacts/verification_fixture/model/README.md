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

Use_Model.ipynb accepts query/gallery file paths, retrieves Top 10 with global cosine similarity, then reranks those candidates by local MNN count (ties retain global rank).
Download config.json and model.safetensors as separate files; do not zip the model.

Verification against best checkpoint:
```json
{
  "global_shape": [
    1,
    128
  ],
  "max_absolute_difference": 0.0,
  "cosine_similarity": 1.0,
  "local_patch_shape": [
    1,
    4,
    32
  ],
  "selected_local_count": 4,
  "local_max_absolute_difference": 0.0,
  "strict_load": true,
  "mnn_matches_equal": true
}
```
