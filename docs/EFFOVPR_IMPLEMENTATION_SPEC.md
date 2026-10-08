# EffoVPR End-to-End Implementation Specification

## 0. IMPORTANT OPERATING MODE

You are a senior Computer Vision / Deep Learning / Visual Place Recognition engineer.

This repository is being implemented in a constrained coding-agent environment.

### Critical rule: DO NOT perform web research

You MUST NOT browse the web, search Google, search Hugging Face pages, search GitHub repositories, or perform external research to discover how EffoVPR or the dataset works.

All required research information is already provided in this specification.

Treat this document as the primary implementation specification.

You may still download/install dependencies and load the specified Hugging Face dataset programmatically when necessary for execution.

Do NOT pause implementation because you cannot browse the internet.

Do NOT ask the user to provide information that is already specified here.

Do NOT replace specified implementation details with another architecture merely because another implementation is easier.

When this document explicitly defines a value or behavior, use that value or behavior unless the existing repository makes it technically impossible.

When the document does NOT specify an implementation detail:

* choose a reasonable implementation
* make it configurable if practical
* document the decision
* do not silently change core EffoVPR mathematics

---

# 1. PROJECT OBJECTIVE

Rebuild a complete, clean, reproducible implementation of:

**EffoVPR: Effective Foundation Model Utilization for Visual Place Recognition**

based on the supplied EffoVPR paper PDF.

The final system must support:

1. DINOv2 zero-shot global retrieval
2. EffoVPR zero-shot retrieval + local re-ranking
3. Fine-tuned EffoVPR global retrieval
4. Fine-tuned EffoVPR two-stage retrieval + local re-ranking
5. VN_Attractions dataset loading
6. Deterministic train/gallery/query splitting
7. Gallery feature extraction
8. FAISS retrieval index construction
9. Global retrieval
10. EffoVPR local feature extraction
11. Attention-based keypoint selection
12. Value-feature local descriptors
13. Mutual Nearest Neighbor matching
14. Top-K re-ranking
15. Recall@1 / Recall@5 / Recall@10
16. MRR
17. Ablation experiments
18. Runtime and memory benchmarking
19. Failure analysis
20. Visualization
21. Unit tests
22. Smoke tests
23. Reproducible documentation

The deliverable is NOT merely a neural network.

The deliverable is the entire research pipeline:

```text
Dataset
   ↓
Dataset Audit
   ↓
Deterministic Split
   ↓
DINOv2 / EffoVPR Model
   ↓
Training
   ↓
Checkpoint
   ↓
Gallery Feature Extraction
   ↓
FAISS Index
   ↓
Global Retrieval
   ↓
Top-K Candidates
   ↓
Intermediate Q/K/V
   ↓
Attention-Based Local Selection
   ↓
V Local Descriptors
   ↓
Mutual Nearest Neighbor Matching
   ↓
T1/T2 Filtering
   ↓
Re-ranking
   ↓
Evaluation
   ↓
Ablation
   ↓
Visualization
   ↓
Report
```

---

# 2. SOURCE-OF-TRUTH HIERARCHY

Use this hierarchy:

### Source 1 — This specification

This document is authoritative for:

* dataset information
* implementation requirements
* default configuration
* experiment structure
* engineering constraints
* VN_Attractions evaluation protocol

### Source 2 — Supplied `EffoVPR.pdf`

Use the supplied paper for:

* architecture
* mathematical formulation
* Q/K/V methodology
* training strategy
* re-ranking strategy
* paper-specific hyperparameters
* paper ablations
* original evaluation methodology

### Source 3 — Existing repository

Use the current repository to determine:

* existing code
* existing interfaces
* existing dependencies
* what can be reused
* what must be refactored

Do NOT use external web research as an additional source.

---

# 3. PAPER-FAITHFUL EFFOVPR CONCEPT

EffoVPR consists of two retrieval stages.

## Stage 1 — Global Retrieval

A ViT backbone produces a global representation using the `[CLS]` token.

The global descriptor is optionally projected to a compact dimension.

The descriptor is L2-normalized.

Nearest-neighbor retrieval is performed over the gallery.

## Stage 2 — Local Re-ranking

The top-K candidates from Stage 1 are re-ranked using intermediate ViT features.

The method:

1. extracts Q/K/V from an intermediate transformer layer
2. computes an attention-based importance score for image patches
3. selects strong local patches using threshold `T1`
4. uses the Value facet `V` as local descriptors
5. computes cosine similarities between query/candidate local features
6. identifies Mutual Nearest Neighbors
7. keeps only matches above threshold `T2`
8. counts valid mutual matches
9. sorts candidates by MNN count

No additional learned re-ranking network is required.

No external VLAD/NetVLAD/GeM pooling is required.

No RANSAC/spatial verification is part of the canonical EffoVPR re-ranker.

---

# 4. BACKBONE

Use:

```text
DINOv2 ViT-L/14 with registers
```

Preferred identifier:

```text
facebook/dinov2-with-registers-large
```

Equivalent official DINOv2 implementation may be used if already available in the repository.

Expected backbone characteristics:

```text
Architecture: ViT-L/14
Transformer blocks: 24
Hidden dimension: 1024
Patch size: 14
Register tokens: 4
```

The input token structure is:

```text
[CLS]
[REGISTER 1]
[REGISTER 2]
[REGISTER 3]
[REGISTER 4]
[PATCH TOKENS...]
```

This token distinction is mandatory.

### Global feature

Use:

```text
CLS token
```

### Local features

Use:

```text
PATCH TOKENS ONLY
```

Never treat:

* CLS
* register tokens

as image patches.

---

# 5. REGISTER TOKEN HANDLING

This is a mandatory correctness requirement.

Implement explicit token parsing.

Conceptually:

```python
cls_token
register_tokens
patch_tokens
```

The implementation must know exactly which tokens correspond to image patches.

For image resolution `224`:

```text
224 / 14 = 16
16 × 16 = 256 patches
```

For image resolution `504`:

```text
504 / 14 = 36
36 × 36 = 1296 patches
```

Therefore:

```text
224 → 256 patch tokens
504 → 1296 patch tokens
```

Add runtime assertions confirming:

```text
actual_patch_count == grid_height * grid_width
```

Do not silently accept incorrect token counts.

---

# 6. GLOBAL FEATURE

The canonical EffoVPR global descriptor is based on the CLS representation.

Default:

```text
backbone dimension = 1024
global dimension = 1024
```

Support:

```text
1024
256
128
```

through one configurable projection mechanism.

Conceptually:

```text
DINOv2
   ↓
CLS
   ↓
Linear Projection
   ↓
L2 Normalize
   ↓
Global Descriptor
```

Do NOT replace the canonical global feature with:

* GeM
* NetVLAD
* VLAD
* average pooling
* max pooling
* external learned aggregation

The purpose is to reproduce EffoVPR's compact CLS-based global representation.

---

# 7. MODEL STRUCTURE

Create one canonical EffoVPR model.

Conceptually:

```text
EffoVPR
│
├── DINOv2 ViT-L/14 with registers
│
├── Global projection
│
├── CosFace classification head
│
└── Q/K/V extraction interface
```

The API should expose functionality equivalent to:

```python
extract_global(image)

extract_global_and_local(image, layer=...)

extract_qkv(image, layer=...)
```

Avoid duplicating the DINO feature extraction implementation across modules.

---

# 8. FINE-TUNING STRATEGY

Paper-faithful default:

```text
Initialize from pretrained DINOv2 with registers
Freeze most of the backbone
Fine-tune only the final five transformer blocks
Train global projection
Train CosFace classifier
```

ViT-L/14 contains 24 transformer blocks.

Only the final five blocks are trainable by default.

Implement:

```python
configure_trainable_layers(last_n_layers=5)
```

This function must:

1. freeze the backbone
2. unfreeze the final N blocks
3. unfreeze the projection head
4. unfreeze the classification head

Add tests verifying this behavior.

---

# 9. COSFACE

Use classification supervision.

For VN_Attractions, each `label` is treated as a place/class identity.

Implement CosFace explicitly.

Conceptually:

```text
Normalize embedding
Normalize classifier weights
Compute cosine logits
Subtract margin from target class
Multiply by scale
Cross entropy
```

Make the following configurable:

```text
cosface_scale
cosface_margin
```

Do not hardcode them deep inside the model.

The classifier must support arbitrary:

```text
num_classes
global_dim
```

---

# 10. OPTIMIZATION

Paper defaults:

```text
Backbone optimizer: AdamW
Classification heads optimizer: Adam
Learning rate: 1e-5
```

Use:

```text
backbone_lr = 1e-5
classifier_lr = 1e-5
```

Expose:

```text
backbone_lr
classifier_lr
weight_decay
```

in configuration.

---

# 11. TRAINING DEFAULTS

Paper-faithful defaults:

```text
Batch size = 16
Epochs = 25
Training resolution = 224 × 224
```

Support:

* CUDA
* CPU fallback
* mixed precision
* checkpoint resume
* deterministic seed
* configurable workers
* optional gradient accumulation

Use validation Recall@1 for checkpoint selection.

Do not select the best model only using training loss.

---

# 12. IMAGE PREPROCESSING

Training:

```text
224 × 224
```

Evaluation:

```text
224 × 224
322 × 322
504 × 504
```

Use DINOv2-compatible image normalization.

Query and gallery preprocessing MUST be identical.

Do not introduce hidden differences between:

* indexing
* query extraction
* validation
* testing

The canonical high-resolution evaluation is:

```text
504 × 504
```

---

# 13. DATASET — VN_ATTRACTIONS

## Exact dataset identifier

Use:

```text
PhamPhien/VN_Attractions
```

The dataset must be loaded programmatically using the Hugging Face `datasets` library.

Canonical loading:

```python
from datasets import load_dataset

dataset = load_dataset(
    "PhamPhien/VN_Attractions",
    split="train"
)
```

Do NOT search for another Vietnamese attractions dataset.

Do NOT substitute this dataset with another dataset.

---

# 14. VN_ATTRACTIONS DATASET DESCRIPTION

The current dataset specification is:

```text
Dataset name:
PhamPhien/VN_Attractions

Modality:
Image + label metadata

Available split:
train

Current dataset size:
5,902 samples

Current number of label values:
19

Approximate total dataset size:
810 MB
```

These values describe the dataset version used as the specification baseline.

The implementation should still inspect the loaded dataset at runtime to verify the actual values.

Do NOT browse the web to obtain additional information.

---

# 15. VN_ATTRACTIONS COLUMNS

The dataset has exactly these three primary columns:

## `id`

Type:

```text
string
```

Description:

```text
Unique identifier of an image sample.
```

Use it to:

* identify samples
* preserve sample identity
* perform duplicate checks
* save split assignments
* map retrieval results back to the original dataset

Do not discard this field.

---

## `image`

Type:

```text
Image
```

Description:

```text
The visual image of a Vietnamese attraction/place.
```

Use it as the model input.

Convert it safely to RGB before preprocessing.

Handle:

* decoding failures
* unusual image modes
* corrupted examples

Do not convert the image into a text representation.

---

## `label`

Type:

```text
string
```

Description:

```text
Attraction/place identity used as the retrieval class.
```

For the current dataset version there are:

```text
19 label values
```

Do NOT hardcode the 19 label names unless they are actually read from the dataset.

Build the class mapping dynamically from the loaded dataset.

Example conceptual mapping:

```python
label_to_class_id = {
    label_name: integer_class_id
}
```

Save this mapping with the checkpoint.

---

# 16. IMPORTANT DATASET INTERPRETATION

VN_Attractions is NOT the same type of dataset as the original geo-tagged VPR datasets used in the EffoVPR paper.

The original EffoVPR training strategy creates geographic cells such as:

```text
15m × 15m geographic cells
```

and uses those cells as classes.

VN_Attractions instead provides:

```text
attraction labels
```

Therefore the current adaptation is:

```text
Original EffoVPR:
geographic cell → class

VN_Attractions:
attraction label → class
```

This distinction MUST be explicitly documented.

---

# 17. DO NOT FABRICATE GEOLOCATION

VN_Attractions does not provide the geographic information required to reproduce the original:

```text
25-meter geolocation criterion
```

Therefore:

DO NOT:

* infer GPS from attraction names
* create fake latitude/longitude
* estimate coordinates from external sources
* convert labels into fake 15m cells
* report 25m geolocation accuracy

The current evaluation criterion is:

```text
same-attraction label match
```

---

# 18. DATASET AUDIT

Before training, create a dataset audit.

Measure:

```text
sample count
unique IDs
unique labels
samples per label
minimum samples per label
maximum samples per label
mean samples per label
median samples per label
invalid image count
duplicate ID count
duplicate image count
image mode statistics
image resolution statistics
```

Save:

```text
artifacts/dataset_audit.json
artifacts/dataset_audit.csv
```

The audit must run automatically.

Do not rely on the currently documented `5,902` count without checking the actual loaded dataset.

The documented number is the expected baseline, not a reason to ignore runtime verification.

---

# 19. DATA SPLIT

Because VN_Attractions only supplies a single `train` split, construct the experiment splits deterministically.

Default:

```text
70% training
10% gallery
10% validation query
10% test query
```

Perform the split at the attraction-label level.

Every label should appear in all required subsets whenever mathematically possible.

The split must be:

* deterministic
* reproducible
* leakage-safe

Save:

```text
artifacts/splits/train.csv
artifacts/splits/gallery.csv
artifacts/splits/val_queries.csv
artifacts/splits/test_queries.csv
```

Each record must contain at minimum:

```text
id
label
dataset_index
split
```

---

# 20. DUPLICATE / LEAKAGE DETECTION

Before finalizing splits:

Check:

```text
duplicate IDs
exact duplicate images
near-duplicate images where practical
query/gallery overlap
cross-split duplicate images
```

If duplicates exist:

* group them
* keep duplicate groups in one split

Do not allow an image to simultaneously serve as:

* gallery
* query
* training image

if it is effectively the same visual sample.

If severe leakage is detected:

* stop evaluation
* report the issue
* regenerate the split

Do not hide the leakage.

---

# 21. CURRENT EVALUATION PROTOCOL

The current evaluation is label-based retrieval.

For a query `q` with label `Lq`:

A retrieval is correct when at least one retrieved gallery image has:

```text
gallery_label == query_label
```

For Recall@K:

```text
correct(q,K) = 1
```

when one or more images with the same label occur in the top K.

Otherwise:

```text
correct(q,K) = 0
```

Primary metrics:

```text
Recall@1
Recall@5
Recall@10
```

Auxiliary:

```text
MRR
```

Also report:

```text
macro Recall@1
macro Recall@5
macro Recall@10

per-class Recall@1
per-class Recall@5
per-class Recall@10
```

---

# 22. EVALUATION NAMING

Use this terminology:

```text
VN_Attractions Label-Based Retrieval
```

Do NOT call this:

```text
25m localization
GPS localization accuracy
geographic Recall@1
```

unless actual geographic coordinates are introduced later.

---

# 23. RETRIEVAL DATABASE

The retrieval database means the vector retrieval index over gallery images.

Use FAISS.

Default:

```text
IndexFlatIP
```

because the initial VN_Attractions gallery is relatively small.

All global features must be L2-normalized before insertion.

Therefore inner product corresponds to cosine similarity.

Design the retrieval layer so another FAISS backend can be added later without changing model code.

---

# 24. GALLERY INDEX ARTIFACTS

Save:

```text
artifacts/index/
    gallery.index
    gallery_features.npy
    gallery_metadata.parquet
    index_config.json
```

Metadata must contain at least:

```text
image_id
label
dataset_index
global_feature_dimension
checkpoint identifier
input resolution
model identifier
```

Validate:

```text
number_of_vectors == number_of_metadata_rows
```

before allowing evaluation.

---

# 25. FEATURE CACHING

Global feature extraction is expensive.

Implement caching.

Recommended:

```text
artifacts/features/global/
artifacts/features/local/
```

Cache metadata must record:

```text
checkpoint
model identity
input resolution
feature dimension
local feature layer
facet
T1
T2
split
```

Never silently reuse incompatible caches.

---

# 26. LOCAL RE-RANKING

Default fine-tuned EffoVPR-R:

```text
local layer = n - 1
facet = V
T1 = 0.05
T2 = 0.65
top-K = 100
```

Procedure:

```text
Query
  ↓
Global Retrieval
  ↓
Top-100
  ↓
Intermediate Q/K/V
  ↓
Attention Score
  ↓
T1 patch filtering
  ↓
V descriptors
  ↓
MNN
  ↓
T2 similarity threshold
  ↓
MNN count
  ↓
Re-ranking
```

---

# 27. ZERO-SHOT LOCAL RE-RANKING

For EffoVPR-ZS:

```text
backbone = vanilla pretrained DINOv2
local layer = n - 2
facet = V
top-K = 100
T1 = 0.05
T2 = 0.65
```

Zero-shot mode MUST NOT update DINOv2 weights.

---

# 28. Q/K/V EXTRACTION

Extract actual internal Q/K/V features.

Do NOT replace them with:

* attention probabilities
* output tokens
* projected CLS only
* external CNN features

For ViT-L/14:

```text
hidden dimension = 1024
```

Reconstruct multi-head features into the original embedding dimension.

Expected conceptual output:

```text
Q: [num_tokens, 1024]
K: [num_tokens, 1024]
V: [num_tokens, 1024]
```

for a single image after head concatenation.

Actual batch tensor layout may differ internally, but the logical embedding dimension must remain 1024.

---

# 29. LOCAL TOKEN FILTERING

After Q/K/V extraction:

```text
exclude CLS
exclude registers
retain patch tokens
```

This produces:

```text
Q_patch
K_patch
V_patch
```

The CLS key representation is retained separately for attention-based patch selection.

---

# 30. ATTENTION-BASED KEYPOINT SELECTION

Implement the paper's local attention score conceptually as:

```text
S = Softmax(Q_l · k_cls)
```

where:

```text
Q_l   = query facet at selected intermediate layer
k_cls = key representation for CLS
```

The result is a score associated with each image patch.

Then:

```text
V_selected = {v_i | S_i > T1}
```

The number of selected features must be variable.

Do not force a fixed number of keypoints in the canonical implementation.

---

# 31. MULTI-HEAD HANDLING

Because the paper writes the equations in simplified tensor notation, use the following deterministic implementation policy:

1. obtain Q/K/V from all attention heads
2. concatenate head dimensions
3. reconstruct the full 1024-dimensional representation
4. use the CLS K vector for patch attention scoring
5. use patch V vectors as local descriptors

This implementation decision must be documented.

Do not add another learned projection to the local descriptors unless explicitly configured as an experimental variant.

---

# 32. T1

Default:

```text
T1 = 0.05
```

Support:

```text
0.00
0.05
0.10
```

Record:

```text
average selected patches
minimum selected patches
maximum selected patches
standard deviation
```

---

# 33. MUTUAL NEAREST NEIGHBORS

Given query descriptors:

```text
Vq = [v1, v2, ... vm]
```

and candidate descriptors:

```text
Vc = [u1, u2, ... un]
```

normalize descriptors.

Compute:

```text
similarity = Vq @ Vc.T
```

after normalization.

For every query descriptor:

* find nearest candidate descriptor

For every candidate descriptor:

* find nearest query descriptor

A pair is mutual only when:

```text
query → candidate
candidate → query
```

both agree.

Then require:

```text
cosine_similarity > T2
```

Default:

```text
T2 = 0.65
```

The re-ranking score is:

```text
MNN count
```

---

# 34. MNN IMPLEMENTATION

Use vectorized tensor operations.

Do NOT implement the main MNN computation as nested Python loops.

Preferred:

```python
similarity = query_local @ candidate_local.T
query_to_candidate = similarity.argmax(dim=1)
candidate_to_query = similarity.argmax(dim=0)
```

Then identify mutual pairs using vectorized indexing.

Threshold the corresponding similarities with `T2`.

---

# 35. TOP-K RE-RANKING

Default:

```text
K = 100
```

Support:

```text
K = 5
K = 10
K = 25
K = 50
K = 100
```

Tie-breaking:

```text
preserve original global rank
```

This ensures deterministic output.

---

# 36. EXPERIMENT MODES

Implement four explicit modes.

## Mode A — DINOv2-ZS

```text
pretrained DINOv2
↓
CLS
↓
normalize
↓
FAISS
↓
top-K
```

No fine-tuning.

No re-ranking.

---

## Mode B — EffoVPR-ZS

```text
pretrained DINOv2
↓
CLS global retrieval
↓
top-100
↓
n-2 Q/K/V
↓
attention selection
↓
V descriptors
↓
MNN
↓
T1/T2
↓
re-ranking
```

---

## Mode C — EffoVPR-G

```text
fine-tuned DINOv2
↓
CLS
↓
projection
↓
normalize
↓
FAISS
```

No re-ranking.

---

## Mode D — EffoVPR-R

```text
fine-tuned DINOv2
↓
global CLS retrieval
↓
top-100
↓
n-1 Q/K/V
↓
attention selection
↓
V descriptors
↓
MNN
↓
T1/T2
↓
re-ranking
```

---

# 37. MAIN DEFAULT CONFIGURATION

Use:

```yaml
model:
  backbone: "facebook/dinov2-with-registers-large"
  global_dim: 1024
  fine_tune_last_n_layers: 5
  with_registers: true

training:
  batch_size: 16
  epochs: 25
  training_resolution: 224
  backbone_lr: 1e-5
  classifier_lr: 1e-5

inference:
  resolution: 504

reranking:
  enabled: true
  top_k: 100
  layer: "n-1"
  zero_shot_layer: "n-2"
  facet: "V"
  T1: 0.05
  T2: 0.65

retrieval:
  backend: "faiss"
  index_type: "IndexFlatIP"
  normalize: true

evaluation:
  k_values: [1, 5, 10]
```

Create:

```text
configs/effovpr_vn_attractions.yaml
```

---

# 38. TRAINING SCRIPT

Create:

```text
scripts/train.py
```

Support:

```bash
python -m scripts.train \
    --config configs/effovpr_vn_attractions.yaml
```

The command must:

1. load dataset
2. validate dataset
3. load deterministic split
4. initialize DINOv2
5. configure trainable layers
6. train
7. validate after each epoch
8. save `last.pt`
9. save `best.pt`
10. save training history
11. save run metadata

---

# 39. CHECKPOINT CONTENT

Save:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict if applicable
epoch
best_metric
configuration
class mapping
random seed
dataset information
git commit if available
```

Save:

```text
artifacts/checkpoints/last.pt
artifacts/checkpoints/best.pt
```

---

# 40. VALIDATION

Use:

```text
validation Recall@1
```

to determine the best checkpoint.

Validation should perform:

```text
validation queries
↓
gallery
↓
global retrieval
↓
Recall@1
```

Do not select the best model using only training loss.

---

# 41. GALLERY INDEX SCRIPT

Create:

```text
scripts/build_gallery_index.py
```

Example:

```bash
python -m scripts.build_gallery_index \
    --config configs/effovpr_vn_attractions.yaml \
    --checkpoint artifacts/checkpoints/best.pt
```

The command must:

1. load gallery split
2. load checkpoint
3. extract global descriptors
4. normalize descriptors
5. build FAISS index
6. save index
7. save metadata
8. validate vector/metadata alignment

---

# 42. EVALUATION SCRIPT

Create:

```text
scripts/evaluate.py
```

Example:

```bash
python -m scripts.evaluate \
    --config configs/effovpr_vn_attractions.yaml \
    --checkpoint artifacts/checkpoints/best.pt \
    --mode effovpr-r
```

Supported values:

```text
dino-zs
effovpr-zs
effovpr-g
effovpr-r
```

---

# 43. EVALUATION OUTPUT

Save:

```text
artifacts/evaluation/<mode>/results.json
artifacts/evaluation/<mode>/query_predictions.csv
```

Each query prediction should record:

```text
query_id
query_label

global_rank_1_id
global_rank_1_label
global_rank_1_score

reranked_rank_1_id
reranked_rank_1_label
reranked_rank_1_score

top_5_ids
top_5_labels

top_10_ids
top_10_labels

is_correct_at_1
is_correct_at_5
is_correct_at_10
```

---

# 44. REQUIRED MODEL COMPARISON

The final results must compare:

| Method     | Global Retrieval | Re-ranking | Fine-tuned |
| ---------- | ---------------- | ---------- | ---------- |
| DINOv2-ZS  | Yes              | No         | No         |
| EffoVPR-ZS | Yes              | Yes        | No         |
| EffoVPR-G  | Yes              | No         | Yes        |
| EffoVPR-R  | Yes              | Yes        | Yes        |

For each:

```text
Recall@1
Recall@5
Recall@10
MRR
latency
feature dimension
gallery feature storage
```

---

# 45. COMPACT FEATURE EXPERIMENT

Evaluate:

```text
1024D
256D
128D
```

for EffoVPR-G.

Report:

```text
dimension
R@1
R@5
R@10
MRR
feature storage
index size
retrieval latency
```

Do not fabricate any values.

---

# 46. RE-RANK K ABLATION

Run:

```text
K = 5
K = 10
K = 25
K = 50
K = 100
```

Compare:

```text
global R@1
reranked R@1
reranked R@5
reranked R@10
latency
```

Save:

```text
artifacts/evaluation/ablations/rerank_k.csv
```

---

# 47. LAYER ABLATION

Support:

```text
n-5
n-4
n-3
n-2
n-1
n
```

Record:

```text
layer
actual transformer block index
R@1
R@5
R@10
average local features
latency
```

Default fine-tuned:

```text
n-1
```

Default zero-shot:

```text
n-2
```

---

# 48. FACET ABLATION

Evaluate:

```text
Q
K
V
```

as local descriptors.

Expected canonical EffoVPR facet:

```text
V
```

Save:

```text
artifacts/evaluation/ablations/facet.csv
```

---

# 49. T1 ABLATION

Run:

```text
T1 = 0.00
T1 = 0.05
T1 = 0.10
```

Keep T2 fixed.

Save:

```text
artifacts/evaluation/ablations/t1.csv
```

---

# 50. T2 ABLATION

Run:

```text
T2 = 0.50
T2 = 0.65
T2 = 0.80
```

Keep T1 fixed.

Save:

```text
artifacts/evaluation/ablations/t2.csv
```

---

# 51. RESOLUTION ABLATION

Run:

```text
224
322
504
```

Measure:

```text
R@1
R@5
R@10
GPU memory
feature extraction latency
reranking latency
end-to-end latency
```

Save:

```text
artifacts/evaluation/ablations/resolution.csv
```

---

# 52. RUNTIME BENCHMARK

Create:

```text
scripts/benchmark.py
```

Measure separately:

```text
preprocessing
backbone forward
global feature extraction
FAISS search
QKV extraction
local patch selection
MNN
reranking
end-to-end
```

Use:

```text
warmup
multiple repeated measurements
mean
median
p95
```

Record hardware:

```text
GPU
CUDA version
PyTorch version
batch size
resolution
checkpoint
```

Never claim that these runtime numbers reproduce the paper unless the hardware and procedure are equivalent.

---

# 53. MEMORY BENCHMARK

Measure:

```text
model memory
gallery feature memory
FAISS index memory
peak inference VRAM
local reranking memory
```

Also report theoretical global feature storage:

```text
N × D × bytes_per_element
```

For float32:

```text
4 bytes per value
```

---

# 54. FEATURE CACHING STRATEGY

Do NOT precompute local features for the entire gallery by default.

Default scalable behavior:

```text
Precompute gallery global features
        ↓
FAISS
        ↓
retrieve top-K
        ↓
compute local features only for:
    query
    top-K candidates
```

This avoids unnecessary local-feature storage and computation.

---

# 55. VISUALIZATION

Create:

```text
scripts/visualize.py
```

Generate:

1. global top-10 retrieval
2. re-ranked top-10 retrieval
3. attention heatmap
4. selected local patches
5. MNN correspondence visualization

The visualization should clearly show:

```text
query
candidate
selected patches
matching patches
MNN count
prediction label
ground-truth label
```

---

# 56. FAILURE ANALYSIS

Classify:

```text
global wrong → reranker correct
global correct → reranker wrong
both wrong
correct result exists in top-10 but not top-1
correct result exists in top-100 but not top-10
correct result absent from top-100
```

Save:

```text
artifacts/failure_analysis/
```

and:

```text
failure_cases.csv
```

Include:

```text
query_id
query_label
global_prediction
reranked_prediction
global_correct_rank
reranked_correct_rank
failure_category
MNN score information
```

---

# 57. TESTS

Create:

```text
tests/
```

Tests must cover:

## Dataset

* loading
* column existence
* label mapping
* deterministic split
* duplicate checks

## Model

* output dimensions
* CLS extraction
* global projection
* trainable layer configuration
* checkpoint loading

## Registers

* CLS excluded from patch features
* register tokens excluded
* patch count correct

## Local features

* Q/K/V shape
* V descriptor dimension
* attention score shape
* T1 filtering

## MNN

* mutuality
* threshold
* known synthetic examples

## Retrieval

* FAISS index
* metadata alignment
* top-K output

## Metrics

* Recall@1
* Recall@5
* Recall@10
* MRR

---

# 58. SYNTHETIC MNN TEST

Create a synthetic test in which:

```text
query feature A
candidate feature A
```

must produce a mutual match.

Then test:

```text
similarity < T2
```

must eliminate the pair.

Then test:

```text
one-way nearest neighbor
```

must not count as mutual.

This test must run without DINOv2.

---

# 59. SYNTHETIC RETRIEVAL TEST

Create synthetic gallery:

```text
gallery_1 → class A
gallery_2 → class A
gallery_3 → class B
gallery_4 → class C
```

Create a query whose embedding exactly matches gallery_1.

Expected:

```text
Recall@1 = 1
```

This validates the retrieval and metric stack independently of the neural network.

---

# 60. SMOKE TEST

Create:

```text
scripts/smoke_test.py
```

The smoke test must verify:

```text
dataset loading
↓
one training batch
↓
forward
↓
loss
↓
backward
↓
checkpoint save
↓
gallery indexing
↓
retrieval
↓
reranking
↓
evaluation
```

CPU mode must work where technically possible.

Run with:

```bash
python -m scripts.smoke_test
```

A smoke test result must never be confused with a full training result.

---

# 61. FULL EXPERIMENT ORCHESTRATION

Create:

```text
scripts/run_experiments.py
```

Example:

```bash
python -m scripts.run_experiments \
    --config configs/effovpr_vn_attractions.yaml
```

It should support stages:

```text
dataset audit
split generation
smoke test
DINOv2-ZS
EffoVPR-ZS
fine-tuning
gallery indexing
EffoVPR-G
EffoVPR-R
compact feature experiment
rerank-K ablation
layer ablation
facet ablation
T1 ablation
T2 ablation
resolution ablation
runtime benchmark
memory benchmark
failure analysis
visualization
report generation
```

Individual stages must also remain executable independently.

---

# 62. REPRODUCIBILITY

Every experiment must save:

```text
random seed
Python version
PyTorch version
CUDA version
GPU
dataset identifier
dataset revision if available locally
backbone identifier
checkpoint
image resolution
global dimension
rerank K
local layer
facet
T1
T2
batch size
git commit if available
timestamp
```

Save:

```text
artifacts/run_metadata.json
```

---

# 63. NO INTERNET RESEARCH REQUIREMENT

The implementation agent MUST NOT:

```text
search "EffoVPR GitHub"
search "EffoVPR implementation"
search "VN_Attractions details"
search "EigenPlaces defaults"
search "DINOv2 implementation tutorial"
```

to decide the implementation.

Use the specification and supplied paper.

The only external operation needed for the dataset is downloading/loading:

```text
PhamPhien/VN_Attractions
```

through the Hugging Face dataset library.

Do not perform exploratory web research.

---

# 64. NO FABRICATED RESULTS

This is mandatory.

Never write:

```text
R@1 = 95.4%
```

unless the experiment actually produced that result.

Never fabricate:

```text
dataset statistics
runtime
memory
training completion
ablation values
benchmark results
```

Use:

```text
N/A
not executed
failed
```

where appropriate.

Every reported metric must have an executable source.

---

# 65. PAPER RESULT VS CURRENT EXPERIMENT

The paper reports results on datasets such as:

```text
Pitts30k
Tokyo24/7
MSLS
Nordland
SF-XL
SF-Night
SF-Occlusion
SVOX
AmsterTime
```

Those published numbers are reference results from the paper.

Do not present them as results obtained from VN_Attractions.

Maintain this separation:

```text
PAPER REPORTED RESULTS
        ≠
OUR VN_ATTRACTIONS RESULTS
```

---

# 66. ORIGINAL PAPER EVALUATION DIFFERENCE

The original EffoVPR benchmark evaluates localization using geographic distance, commonly using:

```text
25 meter radius
```

VN_Attractions currently does not provide the required geographic coordinate information.

Therefore our current experiment evaluates:

```text
same-label retrieval
```

This must be explicitly mentioned in the final report.

---

# 67. CODE STRUCTURE

Use approximately:

```text
project/
│
├── configs/
│   └── effovpr_vn_attractions.yaml
│
├── src/
│   └── effovpr/
│       ├── data/
│       │   ├── dataset.py
│       │   ├── split.py
│       │   ├── audit.py
│       │   └── manifest.py
│       │
│       ├── models/
│       │   ├── effovpr.py
│       │   ├── dino_backbone.py
│       │   ├── dino_attention.py
│       │   ├── cosface.py
│       │   └── projection.py
│       │
│       ├── retrieval/
│       │   ├── index.py
│       │   ├── metadata.py
│       │   └── search.py
│       │
│       ├── reranking/
│       │   ├── local_features.py
│       │   ├── keypoint_selection.py
│       │   └── mnn.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── protocols.py
│       │   └── evaluator.py
│       │
│       ├── visualization/
│       │   └── matching.py
│       │
│       └── utils/
│           ├── seed.py
│           ├── logging.py
│           ├── device.py
│           └── checkpoint.py
│
├── scripts/
│   ├── audit_dataset.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── build_gallery_index.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── run_ablations.py
│   ├── visualize.py
│   ├── smoke_test.py
│   └── run_experiments.py
│
├── tests/
│
├── artifacts/
│
├── README.md
├── REPRODUCTION.md
└── EFFOVPR_IMPLEMENTATION_SPEC.md
```

Adapt this structure to the current repository where equivalent modules already exist.

Do not create duplicate implementations.

---

# 68. REPOSITORY AUDIT

Before modifying code:

1. inspect all existing files
2. identify current architecture
3. identify reusable components
4. identify incorrect/incomplete components
5. identify duplicated implementations
6. identify hardcoded assumptions
7. identify broken tests
8. identify existing dataset code
9. identify existing model code
10. identify existing retrieval code

Then implement the missing pieces.

Prefer refactoring over blindly creating duplicate modules.

---

# 69. ENGINEERING STANDARDS

Use:

* clear naming
* type hints where useful
* structured logging
* reusable modules
* configuration-driven experiments
* explicit error handling
* deterministic execution
* meaningful docstrings
* no hardcoded Windows paths
* no absolute machine-specific paths
* no hidden global state

Do not over-engineer.

The final repository should remain easy to understand.

---

# 70. REQUIRED README

`README.md` must explain:

1. What EffoVPR is
2. System architecture
3. Dataset
4. Dataset columns
5. Dataset split
6. Installation
7. Training
8. Index construction
9. Zero-shot inference
10. Fine-tuned inference
11. Evaluation
12. Ablations
13. Visualization
14. Testing
15. Limitations

Provide executable commands.

---

# 71. REQUIRED REPRODUCTION DOCUMENT

Create:

```text
REPRODUCTION.md
```

It must include:

## 71.1 Paper-to-code mapping

Example:

```text
Paper:
CLS global representation

Implementation:
src/effovpr/models/effovpr.py
```

Repeat this for:

```text
CLS
projection
CosFace
last-five-layer fine-tuning
Q/K/V extraction
attention score
T1
V descriptors
MNN
T2
re-ranking
```

## 71.2 Dataset mapping

Explain:

```text
Original EffoVPR:
geographic cells

Current experiment:
VN_Attractions labels
```

## 71.3 Experimental results

Use only actual measured results.

## 71.4 Ablation results

Use only actually executed experiments.

## 71.5 Runtime/memory

Use actual measurements.

## 71.6 Limitations

Explicitly mention:

* absence of GPS information
* label-based evaluation
* differences from original VPR datasets
* hardware differences
* implementation assumptions

---

# 72. EXPECTED FINAL RESULT TABLE

Generate:

| Method     |  Dim | R@1 | R@5 | R@10 | MRR |
| ---------- | ---: | --: | --: | ---: | --: |
| DINOv2-ZS  | 1024 |     |     |      |     |
| EffoVPR-ZS | 1024 |     |     |      |     |
| EffoVPR-G  | 1024 |     |     |      |     |
| EffoVPR-G  |  256 |     |     |      |     |
| EffoVPR-G  |  128 |     |     |      |     |
| EffoVPR-R  | 1024 |     |     |      |     |

Never fill empty cells with guesses.

---

# 73. EXPECTED ABLATION TABLES

Generate:

## Layer

| Layer | R@1 | R@5 | R@10 | Avg Local Features |
| ----- | --: | --: | ---: | -----------------: |

## Facet

| Facet | R@1 | R@5 | R@10 |
| ----- | --: | --: | ---: |

## Threshold

| T1 | T2 | R@1 | R@5 | R@10 |
| -: | -: | --: | --: | ---: |

## Re-ranking K

|  K | R@1 | R@5 | R@10 | Latency |
| -: | --: | --: | ---: | ------: |

## Resolution

| Resolution | R@1 | R@5 | R@10 | VRAM | Latency |
| ---------: | --: | --: | ---: | ---: | ------: |

---

# 74. FINAL VALIDATION CHECKLIST

Do not declare implementation complete until the following are checked:

```text
[ ] Repository audited
[ ] VN_Attractions loaded
[ ] Dataset has expected columns
[ ] Dataset audit generated
[ ] Class mapping generated
[ ] Deterministic split generated
[ ] Leakage checks passed
[ ] DINOv2 with registers loads
[ ] CLS extraction works
[ ] Registers excluded
[ ] Patch count is correct
[ ] Global feature works
[ ] 1024D projection works
[ ] 256D projection works
[ ] 128D projection works
[ ] CosFace works
[ ] Last-five-layer fine-tuning works
[ ] Checkpoint save/load works
[ ] Gallery index works
[ ] DINOv2-ZS evaluation works
[ ] EffoVPR-ZS evaluation works
[ ] EffoVPR-G evaluation works
[ ] EffoVPR-R evaluation works
[ ] Q/K/V extraction works
[ ] T1 works
[ ] V descriptors work
[ ] MNN works
[ ] T2 works
[ ] K=5 works
[ ] K=10 works
[ ] K=25 works
[ ] K=50 works
[ ] K=100 works
[ ] Layer ablation works
[ ] Facet ablation works
[ ] T1 ablation works
[ ] T2 ablation works
[ ] Resolution ablation works
[ ] Runtime benchmark works
[ ] Memory benchmark works
[ ] Failure analysis works
[ ] Visualization works
[ ] Unit tests pass
[ ] Smoke test passes
[ ] README complete
[ ] REPRODUCTION.md complete
[ ] No fabricated results
```

---

# 75. FINAL IMPLEMENTATION PRINCIPLE

Every major paper concept must have:

```text
Paper concept
      ↓
Mathematical interpretation
      ↓
Tensor shape
      ↓
Implementation
      ↓
Unit test
      ↓
Experiment
```

The canonical pipeline must remain:

```text
DINOv2 ViT-L/14 with registers
        ↓
CLS
        ↓
Global descriptor
        ↓
L2 normalization
        ↓
FAISS global retrieval
        ↓
Top-K
        ↓
Intermediate Q/K/V
        ↓
CLS-based attention score
        ↓
T1 patch selection
        ↓
V local descriptors
        ↓
Cosine similarity
        ↓
Mutual Nearest Neighbors
        ↓
T2 filtering
        ↓
MNN count
        ↓
Re-ranking
        ↓
Recall@K / MRR
```

Do not insert unrelated architectural components into this path.

---

# 76. FINAL DELIVERABLE CONDITION

The implementation is complete only when the repository provides:

```text
working code
+
trained checkpoint when resources allow
+
retrieval index
+
evaluation results
+
ablation results
+
visualizations
+
tests
+
documentation
```

The agent must distinguish among:

```text
FULL EXPERIMENT
LIMITED EXPERIMENT
SMOKE TEST
NOT EXECUTED
FAILED
```

Never present one as another.

---

# 77. REQUIRED FINAL AGENT REPORT

At the end of the implementation, respond with exactly:

## 1. Repository Audit

Describe:

* what existed
* what was reused
* what was refactored
* what was added
* what was removed

## 2. Architecture

Describe the final EffoVPR pipeline.

## 3. Dataset

Report actual runtime values:

```text
sample count
class count
train count
gallery count
validation query count
test query count
duplicate count
invalid image count
```

## 4. Training

Report:

```text
backbone
trainable layers
epochs
batch size
learning rates
global dimensions
best checkpoint
best validation R@1
```

## 5. Evaluation

Report actual:

```text
DINOv2-ZS
EffoVPR-ZS
EffoVPR-G
EffoVPR-R
```

with:

```text
R@1
R@5
R@10
MRR
```

## 6. Ablations

Report only experiments that actually ran.

## 7. Runtime / Memory

Report actual measurements.

## 8. Tests

Report:

```text
pytest
smoke test
```

## 9. Limitations

Explicitly explain:

* no GPS metadata
* label-based evaluation
* deviations from the original geographic VPR protocol
* hardware limitations
* implementation assumptions

## 10. Exact Reproduction Commands

Provide executable commands for:

```text
dataset audit
prepare split
training
index construction
evaluation
ablations
benchmark
visualization
tests
```

---

# 78. AUTONOMOUS EXECUTION RULE

Proceed autonomously.

Do not ask for confirmation for normal engineering choices.

Do not stop after writing code.

The expected behavior is:

```text
Inspect
→ Audit
→ Plan
→ Implement
→ Test
→ Smoke Test
→ Train
→ Index
→ Evaluate
→ Ablate
→ Benchmark
→ Visualize
→ Document
```

When a long-running experiment cannot be executed because of hardware/resource limitations:

* complete implementation
* run all feasible tests
* run a smaller smoke/limited experiment if useful
* clearly identify what was and was not executed
* never fabricate the missing results

The final goal is a reproducible EffoVPR implementation adapted specifically to:

```text
PhamPhien/VN_Attractions
```

without requiring external web research.
