# UniMAG: Multimodal Attributed Graph Pipelines

> **One repository, three pillars** — graph-centric learning, multimodal quality evaluation, and graph-aware creative generation (text & image).  
> This README consolidates and operationalizes the project guides into a single, runnable document with complete usage instructions.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data & Embedding Workflow](#data--embedding-workflow)
  - [Datasets](#datasets)
  - [Embedding Conversion](#embedding-conversion)
  - [Unified Loading](#unified-loading)
- [Graph-Centric Tasks](#graph-centric-tasks)
  - [Node Classification](#node-classification)
  - [Link Prediction](#link-prediction)
  - [Graph-Level Tasks](#graph-level-tasks)
- [Multimodal Quality Evaluation (QE)](#multimodal-quality-evaluation-qe)
  - [Modality Matching](#modality-matching)
  - [Modality Retrieval](#modality-retrieval)
  - [Fine-Grained Alignment](#fine-grained-alignment)
- [Creative Generation](#creative-generation)
  - [G2Text: Graph → Text](#g2text-graph--text)
  - [G2Image: Graph → Image](#g2image-graph--image)
  - [GT2Image: Text + Graph Context → Image](#gt2image-text--graph-context--image)
- [Configuration (YAML) Examples](#configuration-yaml-examples)
- [Metrics](#metrics)
- [Reproducibility & Logging](#reproducibility--logging)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

**UniMAG** provides end-to-end pipelines for **Multimodal Attributed Graphs (MAGs)**:

- Lightweight **GNN backbones** for **graph-centric** learning (node/edge/graph-level).
- **Multimodal quality evaluation**: matching, retrieval, and fine-grained alignment with **graph-aware** enhancement.
- **Creative generation**: graph → text, graph → image, and text + graph context → controllable image generation.

**Outputs**

- **Modality-level**: per-modality embeddings per node (e.g., image/text).
- **Entity-level**: fused node representation derived from its modalities and optionally its neighborhood.

---

## Key Features

- Standardized **embedding conversion** (factory + registry) and a **unified loader** for downstream tasks.
- Clean **task separation** with dedicated subpackages and **YAML-driven** experiments.
- Reproducible training/evaluation with **seed control**, consistent logging, and structured results.
- Built-in utilities for **PPR sampling**, **subgraph construction**, and **common metrics**.

---

## Repository Layout

```
UniMAG/
├─ configs/
│   ├─ nc_gcn.yaml                  # node classification (GCN)
│   ├─ lp_gat.yaml                  # link prediction (GAT)
│   ├─ qe_matching.yaml             # modality matching
│   ├─ qe_retrieval.yaml            # modality retrieval
│   ├─ qe_alignment.yaml            # phrase-region alignment
│   ├─ g2text.yaml                  # graph → text
│   ├─ g2i.yaml                     # graph → image
│   └─ GT2Image.yaml                # text + graph context → image
├─ src/
│   ├─ data/
│   │   ├─ embedding_converter/     # raw → npy feature pipeline (factory + registry)
│   │   └─ embedding_manager.py     # unified feature/graph loader for downstream tasks
│   ├─ graph_centric/
│   │   ├─ train_nc.py
│   │   ├─ train_lp.py
│   │   └─ eval.py
│   ├─ multimodal_centric/
│   │   ├─ qe/
│   │   │   ├─ matching.py
│   │   │   ├─ retrieval.py
│   │   │   └─ alignment.py         # spaCy + GroundingDINO + RoI features
│   │   ├─ g2text/
│   │   │   ├─ decoder.py           # multimodal decoder → soft prompts
│   │   │   └─ infer.py
│   │   ├─ g2i/
│   │   │   ├─ unet.py
│   │   │   └─ pipeline.py
│   │   └─ gt2image/
│   │       ├─ train.py  test.py  infer_pipeline.py
│   │       ├─ GraphQFormer.py  GraphAdapter.py
│   │       └─ dataset.py        # PPR sampling, subgraph loading
│   └─ utils/
│       ├─ graph_samplers.py     # PPR, neighbor search, subgraph assembly
│       └─ metrics.py            # Accuracy/F1/MRR/H@K/NDCG/mAP/Alignment
├─ data/
│   └─ <dataset_name>/           # images/, texts/, graphs/, splits/, labels/
├─ outputs/
│   ├─ ckpts/  logs/  results/
└─ README.md
```

---

## Installation

- **Python**: 3.10+ (recommended)
- **Core**: PyTorch, DGL *or* PyG (choose one)
- **Multimodal/Generation**: `transformers`, `accelerate`, `diffusers`, `torchvision`
- **QE Alignment**: `spacy` (with `en_core_web_sm`), `GroundingDINO`
- **Utils**: `numpy`, `scipy`, `pandas`, `tqdm`, `pyyaml`

```bash
conda create -n unimag python=3.10 -y
conda activate unimag

# core
pip install torch torchvision
pip install dgl               # or: pip install torch-geometric

# multimodal + generation
pip install transformers accelerate diffusers

# QE alignment extras
pip install spacy
python -m spacy download en_core_web_sm

# utilities
pip install numpy scipy pandas tqdm pyyaml
```

---

## Data & Embedding Workflow

### Datasets

Primary: <https://huggingface.co/datasets/enjun-collab/MMAG>  
Additional MAG baselines:  

- <https://huggingface.co/datasets/Sherirto/MAGB>  
- <https://huggingface.co/datasets/mm-graph-org/mm-graph>

**Common dataset components**

- `*-images.tar` / `*.tar.gz`: raw images (file name == node id)
- `*-raw-text.jsonl` / `*.csv`: `{id, text/title/description}`
- `node_mapping.pt`: raw IDs → graph indices
- `Graph.pt` / `nc_edges-nodeid.pt` / `*.pt`: graph structure
- `split.pt`, `labels*.pt`, `lp-edge-split.pt`: splits and labels

### Embedding Conversion

> Convert raw MAG data to `.npy` feature matrices with a **standard naming** scheme and **consistent dtype** (float32).

- **Factory + Registry**: plug-and-play encoders (e.g., CLIP, SigLIP, BLIP; other stable variants).  
- **Modalities**: `text`, `image`, and optionally `multimodal` fused features.  
- **Naming**: `{dataset_name}_{modality}_{encoder_name}_{dimension}d.npy`.

**Examples**

```bash
# Convert MAGB text CSV to unified JSONL (if needed)
python -m src.data.embedding_converter.utils.convert_magb_text_to_mmgraph \
  --in data/MAGB/books-nc.csv \
  --out data/mm-graph/books-nc-raw-text.jsonl

# Extract features with a registered encoder
python -m src.data.embedding_converter.run \
  --dataset books-nc \
  --modality text image \
  --encoder clip-vit-b32 \
  --outdir data/books-nc/features
```

### Unified Loading

Use `embedding_manager.py` in downstream code. It abstracts file paths and naming:

```python
from src.data.embedding_manager import load_node_features, load_graph_splits

x_text  = load_node_features(dataset="books-nc", modality="text",  encoder="clip-vit-b32")  # [N, d_t]
x_image = load_node_features(dataset="books-nc", modality="image", encoder="clip-vit-b32")  # [N, d_i]
splits  = load_graph_splits(dataset="books-nc")  # dict: train/val/test indices
```

---

## Graph-Centric Tasks

Lightweight GNNs for node/edge/graph objectives.

**Backbones** (from the consolidated guide)

- `MLP`, `GCN`, `GAT`, `GraphSAGE`, `RevGAT`, `MMGCN`, `MGAT`

**Losses**

- Node classification: **Cross-Entropy**
- Link prediction: **Binary Cross-Entropy** (with negative sampling)
- Self-supervised: **InfoNCE** (optional)

### Node Classification

```bash
python -m src.graph_centric.train_nc --config configs/nc_gcn.yaml
python -m src.graph_centric.eval     --config configs/nc_gcn.yaml --ckpt outputs/ckpts/nc_gcn.pt
```

### Link Prediction

```bash
python -m src.graph_centric.train_lp --config configs/lp_gat.yaml
python -m src.graph_centric.eval     --config configs/lp_gat.yaml --ckpt outputs/ckpts/lp_gat.pt
```

### Graph-Level Tasks

- Graph classification and community detection are supported via the same backbone design.  
- Provide `graph_splits` and use appropriate pooling (mean/sum/max) prior to classification layers.

**Metrics**: `Accuracy`, `F1-Macro`, `MRR`, `Hits@{1,10}`.

---

## Multimodal Quality Evaluation (QE)

Evaluate cross-modal embedding quality with or without graph context.

### Modality Matching

- **Traditional**: cosine / CLIP-like score between arbitrary image/text embeddings.  
- **Graph-aware**: fetch node + neighbors via the embedding manager; aggregate with a small GNN; compute cosine on **enhanced** embeddings.

```bash
python -m src.multimodal_centric.qe.matching --config configs/qe_matching.yaml
```

### Modality Retrieval

- **Traditional**: similarity matrix `query @ candidates.T` → rank.  
- **Graph-aware**: enhance query via 1-hop aggregation; rank against all other nodes.

```bash
python -m src.multimodal_centric.qe.retrieval --config configs/qe_retrieval.yaml
```

**Training Tips** (from the benchmark report)

- **InfoNCE**: pulls **same-node** cross-modal pairs together; pushes different nodes apart.
- **Symmetric InfoNCE** (retrieval): optimize **text→image** and **image→text** jointly; average the two losses for a **bidirectionally aligned** space.

### Fine-Grained Alignment

- **Offline**: build `(image, [(phrase, box), ...])` using spaCy (noun phrase extraction) + Grounding DINO (region proposals).  
- **Online**: extract region features (e.g., RoIAlign on the feature map) + phrase embeddings → similarity per `(phrase, box)`.  
- Graph-aware alignment uses **GNN-enhanced** feature maps.

```bash
python -m src.multimodal_centric.qe.alignment --config configs/qe_alignment.yaml
```

**Metrics**

- Matching: score distribution / threshold-F1  
- Retrieval: `Recall@K`, `mAP`, `NDCG`  
- Alignment: mean/max phrase-region similarity, coverage

---

## Creative Generation

### G2Text (Graph → Text)

Use a **light multimodal decoder** to map MAG features to **virtual tokens (soft prompts)** that steer a **frozen LLM**.

**Inputs**

- Node entity-level embedding (concatenate modalities; e.g., `768×3=2304`)
- Optional context embeddings (neighbors) and structure encoding (e.g., LPE)

**Multimodal Decoder**

- MLP + LayerNorm + Tanh to project concatenated features to the LLM hidden size
- Learnable positional encodings
- Outputs **virtual tokens** that are prepended to the LLM context

**LLM**

- Local **frozen** model (e.g., Qwen2.5-VL or similar). No gradient on LLM.

**Training**

- Labels: use provided references; if missing, weak labels can be generated from raw graph-text.
- Loss: **Cross-Entropy** (decoder-only training)
- Optimizer: **AdamW**

**Inference**

```python
with torch.no_grad():
    vtoks = decoder(node_and_context_emb)        # [num_virtual_tokens, hidden_size]
    text  = frozen_llm.generate_with_soft_prompt(vtoks, prompt_template)
```

**CLI**

```bash
python -m src.multimodal_centric.g2text.train --config configs/g2text.yaml
python -m src.multimodal_centric.g2text.infer --config configs/g2text.yaml --ckpt outputs/ckpts/g2text.pt
```

**Evaluation**

- BLEU/ROUGE/BERTScore + human preference (coherence, faithfulness to graph evidence).

---

### G2Image (Graph → Image)

Feed **precomputed embeddings** into a **diffusion U-Net** to synthesize images.

**Pipeline**

1. Map embeddings to initial conditioning in latent diffusion space.  
2. Inject conditioning via cross-attention or conditional concatenation during denoising.  
3. Decode latents to pixels (e.g., VAE decoder).

**Training Objective (noise reconstruction MSE)**
$$
\mathbb{E}_{t,\mathbf{x},\epsilon}\left[\lVert \epsilon - \epsilon_\theta(\mathbf{x}_t, t, c)\rVert^2\right]
$$
**CLI**

```bash
python -m src.multimodal_centric.g2i.train  --config configs/g2i.yaml
python -m src.multimodal_centric.g2i.infer  --config configs/g2i.yaml --ckpt outputs/ckpts/g2i.pt
```

**Evaluation**

- FID/KID, CLIP score; human study on semantic faithfulness.

---

### GT2Image (Text + Graph Context → Image)

Generate or reconstruct the **target node image** using a **text description** and **graph context** (neighbor images).

**Three Stages**

1. **Informative Neighbor Sampling**  
   - **PPR** for structural importance  
   - Re-rank neighbors by semantic similarity between neighbor images and the target text  
2. **Graph Encoding**  
   - Encode selected neighbors with **Graph-QFormer** (self-attention among neighbors)  
   - Optional **GraphAdapter** for feature alignment  
3. **Controllable Generation**  
   - Diffusion-based synthesis conditioned on **graph prompt** and **target text**

**Training Loss (supervised denoising)**
Let latent $ z \sim \mathrm{Enc}(x) $ of a real image be noised to $ z_t $ with noise $ \varepsilon \sim \mathcal{N}(0,I) $. Minimize
$$
\mathbb{E}_{z,c_T,c_G,\varepsilon,t}\left[\left\lVert \varepsilon - \varepsilon_\theta\!\big(z_t, t, h(c_T,c_G)\big)\right\rVert^2\right]
$$

where $ h(c_T,c_G) $ fuses text and graph conditions.

**Structure & CLI**

```
src/multimodal_centric/gt2image/
  train.py  test.py  infer_pipeline.py
  GraphQFormer.py  GraphAdapter.py
  dataset.py

python -m src.multimodal_centric.gt2image.train  --config configs/GT2Image.yaml
python -m src.multimodal_centric.gt2image.test   --config configs/GT2Image.yaml --ckpt outputs/ckpts/gt2image.pt
python -m src.multimodal_centric.gt2image.infer_pipeline --config configs/GT2Image.yaml --ckpt outputs/ckpts/gt2image.pt
```

**Evaluation**

- FID/KID, CLIP-text consistency, retrieval consistency vs. neighbor context.

---

## Configuration (YAML) Examples

```yaml
# configs/nc_gcn.yaml
task: node_classification          # link_prediction, matching, retrieval, alignment, g2text, g2image, gt2image
dataset:
  name: books-nc
  root: ./data/books-nc
  modalities: [image, text]
features:
  text_encoder: clip-vit-b32
  image_encoder: clip-vit-b32
model:
  backbone: gcn
  hidden_dim: 256
  num_layers: 2
  num_heads: 4            # if applicable (e.g., GAT)
train:
  batch_size: 256
  epochs: 100
  lr: 3.0e-4
  seed: 42
eval:
  metrics: [accuracy, f1_macro]
  topk: [1, 5, 10]
device: cuda
log_dir: ./outputs/logs
save_dir: ./outputs/ckpts
```

```yaml
# configs/qe_retrieval.yaml
task: retrieval
dataset:
  name: books-nc
  root: ./data/books-nc
features:
  text_encoder: clip-vit-b32
  image_encoder: clip-vit-b32
qe:
  use_graph_context: true       # enable MAG-specific enhancement
  loss: symmetric_infonce       # two-way alignment
train:
  batch_size: 512
  epochs: 20
  lr: 2.0e-4
eval:
  metrics: [recall@1, recall@5, recall@10, map, ndcg]
device: cuda
```

```yaml
# configs/g2text.yaml
task: g2text
dataset:
  name: Movies
  root: ./data/Movies
features:
  text_encoder: clip-vit-b32
  image_encoder: clip-vit-b32
model:
  decoder:
    num_virtual_tokens: 16
    hidden_size: 4096      # match LLM hidden dim
    dropout: 0.1
llm:
  name_or_path: Qwen2.5-VL-7B
  freeze: true
train:
  batch_size: 8
  epochs: 5
  lr: 1.0e-4
  weight_decay: 0.01
eval:
  metrics: [bleu, rouge, bertscore]
device: cuda
```

```yaml
# configs/GT2Image.yaml
task: gt2image
dataset:
  name: Reddit-M
  root: ./data/Reddit-M
features:
  image_encoder: clip-vit-b32
  text_encoder: clip-vit-b32
sampling:
  ppr_alpha: 0.15
  topk_neighbors: 16
graph_encoding:
  qformer_layers: 4
  adapter: true
gen:
  diffusion_steps: 50
  guidance_scale: 7.5
train:
  batch_size: 2
  epochs: 10
  lr: 1.0e-4
eval:
  metrics: [fid, kid, clip_score]
device: cuda
```

---

## Metrics

| Family         | Tasks                           | Metrics                                   |
| -------------- | ------------------------------- | ----------------------------------------- |
| Graph-centric  | NC / LP / Graph cls / Community | `Accuracy`, `F1-Macro`, `MRR`, `Hits@K`   |
| QE — Matching  | Image↔Text score                | cosine/CLIP-like, threshold-F1            |
| QE — Retrieval | Text→Image / Image→Text         | `Recall@K`, `mAP`, `NDCG`                 |
| QE — Alignment | Phrase↔Region (fine-grained)    | mean/max similarity, coverage             |
| Generation     | G2Text / G2Image / GT2Image     | BLEU/ROUGE/BERTScore, FID/KID, CLIP score |

---

## Reproducibility & Logging

- Pin **random seeds** in YAML and data loaders.  
- Save `{config, ckpt, metrics.json}` under `outputs/` per run.  
- For retrieval, report **mean ± std** across repeated runs.  
- For alignment, record exact spaCy/GroundingDINO versions.  
- Keep encoder versions fixed for stability (start with CLIP/SigLIP/BLIP families).

---

## Troubleshooting

- **Slow convergence / weak performance**  
  - Graph tasks: check normalization, depth (over-smoothing), and negative sampling.  
  - QE: ensure **symmetric InfoNCE** and sufficient negatives; validate neighbor enhancement.  
  - Generation: tune diffusion steps/lr; adjust number and dimension of virtual tokens (G2Text).

- **Out-of-memory (OOM)**  
  - Reduce batch size and diffusion steps; cap neighbor count in Graph-QFormer; use subgraph sampling.

- **Unstable encoders**  
  - Prefer robust encoders first; some very large multimodal encoders may be unstable on limited VRAM.

---

## License

Released under a permissive open-source license. 
