# UniMAG
**MultiModal Attribute Graph Pipelines**

The datasets are given in huggingface [https://huggingface.co/datasets/enjun-collab/MMAG](https://huggingface.co/datasets/enjun-collab/MMAG) .



## **Task Summary**

### Key Points to Note:

- **Finetune simple models (GCN, GAT) to support modality-level and entity-level outputs**

- Be mindful of the **supervised loss function** corresponding to each downstream task; we need to implement simple versions, e.g., cross-entropy for graph-centric classification tasks

- Multimodal-centric downstream tasks are more complex. Pay attention to class inheritance in the benchmark; the hierarchy is as follows:

  - Quality Evaluation

    - Modality Matching
    - Modality Retrieval
    - Modality Alignment

  - Creative Generation

    - G2Text
    - G2Image
    - GT2Image
    - GT2Text
    - G2TextImage

- Final Output

  - **Modality-level**: Embedding of each modality within each node
  - **Entity-level**: Representation of the entire node based on its modality embeddings

  

### 1. Graph-centric tasks 【Yushuo LI】:

  

  - **Semantic Level** - Node-level: Node classification, Node clustering
  - **Semantic Level** - Edge-level: Edge existence prediction, Edge classification
  - **Semantic Level** - Graph-level: Graph classification, Community detection (subgraphs)



#### **Task Arrangement**

​	**Following the MAGB benchmark for node classification and link prediction tasks.**

* **configs**: **config.yaml**: Specifies the task, model, dataset, random seed, device, and other global configurations.
* **task**: Contains training-related parameters for the task, such as the number of epochs and learning rate.
* **model**: Contains model-related parameters for training, such as hidden dimension size, number of layers, and number of heads.
* **dataset**: Contains dataset-related parameters, such as file path
* **src**: store different downstream tasks.

  

### Multimodal-centric tasks



#### 2. **Quality Evaluation Tasks**【Yilong ZUO】



- **Modality-level**: Modality Matching

  

  - Traditional: Input any image and text, get respective embeddings, return a matching score
  - MAG-specific: Given any image-text pair from a node in MAG, compute a matching score (e.g., CLIP-score) based on embeddings

  

- **Modality-level**: Modality Retrieval

  

  - Traditional: Given an image or text, obtain a query embedding and retrieve the most relevant item from a pool of candidates
  - MAG-specific: Use any image-text from a MAG node as a query and return relevant items from others

  

- **Modality-level**: Modality Alignment

  

  - Traditional: Fine-grained modality matching focusing on matching degree (detailed descriptions)
  - MAG-specific: Given any image-text pair from a MAG node, return fine-grained alignment details based on embeddings

##### Task Arrangement

The Quality Evaluation (QE) tasks are designed to assess the quality and alignment of multimodal embeddings. The implementation will reside in `src/multimodal_centric/qe/`.

-   **Core References**:
    
    -   **CLIP**: For baseline similarity metrics. [Radford, et al. (2021)](https://arxiv.org/abs/2103.00020).
    -   **UniGraph2**: For leveraging graph structure in MAG-specific tasks. [He, Y., et al. (2025)](https://arxiv.org/abs/2502.00806).
    -   **FG-CLIP**: For methodologies in fine-grained, region-based alignment. [Xie, C., et al. (2025)](https://arxiv.org/abs/2505.05071).
    
-   **Planned Project Architecture**:
    ```
    src/multimodal_centric/qe/
    │
    ├── README.md
    ├── matching.py
    ├── retrieval.py
    ├── alignment.py
    └── utils/
    ```

-   **Detailed Workflow**:

    **1. Modality Matching (`matching.py`)**
    | Approach | Input | Processing | Output |
    | :--- | :--- | :--- | :--- |
    | **Traditional** | An arbitrary image embedding and a text embedding. | Calculate the CLIP-score (typically `100 * cosine_similarity`) between the two embeddings. | A single matching score, reflecting context-free alignment. |
    | **MAG-specific** | A `node_id` and a `dataset_name`. | 1. Use `embedding_manager` to fetch the target node's features and its neighbors' features. <br> 2. Use a GNN layer to create a neighborhood-enhanced image embedding and a neighborhood-enhanced text embedding. <br> 3. Calculate the CLIP-score between these two **enhanced** embeddings. | A single matching score, reflecting context-aware alignment. |

    **2. Modality Retrieval (`retrieval.py`)**
    | Approach | Input | Processing | Output |
    | :--- | :--- | :--- | :--- |
    | **Traditional** | A query embedding (image or text) and a pool of candidate embeddings. | 1. Calculate the cosine similarity between the query embedding and all candidate embeddings. <br> 2. Rank candidates based on similarity scores. | A ranked list of candidate IDs. |
    | **MAG-specific** | A query `node_id` and a `dataset_name`. | 1. Fetch the query node's multimodal embedding. <br> 2. Use a GNN to aggregate features from the query node's 1-hop neighbors, creating a neighborhood-enhanced query embedding. <br> 3. Calculate similarity between the enhanced query embedding and all other node embeddings. | A ranked list of node IDs from the graph. |

    **3. Modality Alignment (`alignment.py`)**
    This task is divided into two stages:
    -   **Stage 1: Data Preprocessing (Offline)**: Use external tools (SpaCy, Grounding DINO) to create a benchmark dataset of `(image, [(phrase, box), ...])` mappings.
    -   **Stage 2: Evaluation (Online)**:
        | Approach | Input | Processing | Output |
        | :--- | :--- | :--- | :--- |
        | **Traditional** | A benchmark entry: `(image, [(phrase, box), ...])`. | 1. Get image feature map from **our** image encoder. <br> 2. For each `(phrase, box)` pair, extract region embedding (via RoIAlign) and phrase embedding, then compute similarity. | A list of `(phrase, box, alignment_score)` tuples, reflecting baseline alignment quality. |
        | **MAG-specific** | A `node_id` and its benchmark entry. | 1. Get the target node's feature map and its neighbors' feature maps. <br> 2. Use a GNN layer to create an **enhanced feature map**. <br> 3. Perform the same RoIAlign and similarity calculation on the **enhanced feature map**. | A list of `(phrase, box, alignment_score)` tuples, reflecting context-aware alignment quality. |



#### 3. **Creative Generation Tasks**



- MAG-specific【Modality/Semantic level】graph (image, text) + prompt → text 【G2Text】【Yaxin DENG】

  

  - Input: Current node embedding + (optional) contextual info around the node [subgraph]
  - Multimodal interaction: Integrate embedding with prompt template and input to the language model
  - Output: Summary and analysis of the node, including its attributes and neighborhood info


##### Task Arrangement

Reference paper: [[2310.07478\] Multimodal Graph Learning for Generative Tasks](https://arxiv.org/abs/2310.07478)

Design:
* Input:

  1.	Current Node Embedding: Directly use the entity-level embedding of the current node.
  2.	Node Context Information: Includes the entity-level embeddings of neighboring nodes and the graph structural information (generated using GNNs or LPE).

* Multimodal Interaction:

  1.	The current node and context embeddings can be directly concatenated with the prompt text and fed into a large language model (LLM).

  2.	Prefix Tuning: Use the node and context embeddings to generate a prefix, and fine-tune the LLM accordingly.

  1.	Evaluation: Use the LLM to score the generated node summaries and analyses.




- 【Modality/Semantic level】Image Annotation 【G2Image】【Zhenning ZHANG】

  

  - Traditional: Generate an explanatory image from a text
  - MAG-specific: Similar to G2Text, combine embeddings and prompt, then input to image generator




##### Task Arrangement

**Target Task**: Similar to *G2Text*, but aimed at generating images from Multimodal Attributed Graphs (MMAGs).

- **Reference**: [Multimodal Graph-to-Image Generation with Graph Context-Conditioned Diffusion Models (arXiv:2410.07157)](https://arxiv.org/abs/2410.07157?utm_source=chatgpt.com)
- **Motivation**: This task aims to generate images conditioned on structured multimodal graph data. The referenced method introduces a *Graph Context-Conditioned Diffusion Model* to handle the complexity of graph structures and multimodal attributes.
- **Key Contribution**: A novel encoder is proposed that adaptively transforms graph nodes into *graph prompts*, which effectively guide the denoising process in diffusion-based image generation.

##### **1. Input Data Preparation**

**Goal**: Prepare and combine embeddings from images, texts, and additional prompts for input into an image generator.

- **Image Embeddings**: Pre-extracted visual features from the image modality.
- **Text Embeddings**: Pre-extracted semantic embeddings from the text modality.
- **Prompts**: Optional textual cues that provide additional context. These can be handcrafted or generated using a large language model (LLM).

##### **2. Fusion of Embeddings and Prompts**

To simplify the integration process, we use a basic concatenation method to merge the embeddings from multiple modalities.

- **Concatenated Embeddings**: Combine the image embedding vector, text embedding vector, and the prompt embedding into a single unified vector.
  - Example: [image_embedding ; text_embedding ; prompt_embedding] This joint embedding serves as the input condition for the downstream generation model.
  

##### **3. Generation Model Design**

- **Model Choice**: Use a lightweight image generator (or possibly a generative LLM with visual capabilities) conditioned on the concatenated embeddings.

  - **Input**: Combined embedding vector (image + text + prompt)
  - **Output**: A generated image that reflects both the semantic content and visual cues of the input embeddings
  

##### **4. Loss Function**

- **To Be Explored**: At present, a standard image generation loss (e.g., pixel-wise reconstruction loss, adversarial loss) can be used. Further study on loss formulations specific to MMAGs is needed.

##### **5. Training and Evaluation**

- **Training**: Train the image generator (and optionally a discriminator) using the fused multimodal embeddings. GAN-based strategies may be employed if realism is a key goal.

- **Evaluation**: Image quality can be assessed via:

  - Human evaluation (subjective visual inspection)
  - Automated metrics (e.g., FID, CLIPScore)
  - LLM-based assessment (e.g., prompt-image alignment scoring using GPT-4V or similar)

  


- MAG-specific【Modality-level】graph (image) + text [prompt] → image 【GT2Image】【Sicheng LIU】

  

  - Input: Embeddings of current node (multiple images) + (optional) context [subgraph] + text description
  - Multimodal interaction: Use image/text embeddings as graph/text prompt tokens for diffusion model
  - Output: Image satisfying the textual instruction
  - Belongs to the same category as traditional **image editing or reconstruction**

  

##### Task Arrangement

Given a multimodal attributed graph:

- **Input**:
  - The complete graph dataset (subgraphs must be partitioned manually).
  - The **text description** of a specific target node (serving as the generation prompt).
  - The **image embeddings** of all other nodes **except** the target node.
- **Objective**: Generate an image for the target node, conditioned on its text and the multimodal context from its neighbors.

The GT2Image task directly aligns with InstructG2I, which consists of three key modules:

1. **Informative Neighbor Sampling**
   - Uses a **Personalized PageRank (PPR)** algorithm to identify structurally important neighbors of the target node.
   - These neighbors are **re-ranked** based on the semantic similarity between their images and the target node’s textual prompt.
2. **Graph Encoding**
   - The selected neighbor images are encoded using a visual encoder (e.g., CLIP).
   - A **Graph-QFormer** module enables self-attention-based interaction among neighbor embeddings to capture richer graph-aware multimodal context.
3. **Controllable Generation**
   - A diffusion-based model generates the image, conditioned on the encoded graph prompt and target text.

###### **Planned Project Architecture**

```
project/
│
├── utils/
│   └── GT2Image_utils.py               # Utility functions for data processing and support
│
├── src/
│   └── GT2Image/                       # Main task directory
│       ├── README.md                  # Task introduction and usage guide
│       ├── main.py                    # Entry point
│       ├── test.py                    # Evaluation script
│       ├── train.py                   # Training script
│       ├── GraphAdapter.py            # Optional intermediate graph processing module
│       ├── GraphQFormer.py            # Core module for Step 2: Graph encoding
│       ├── infer_pipeline.py          # Inference workflow
│       ├── customized_sd_pipeline.py  # Custom Stable Diffusion generation pipeline
│       ├── customized_sd_pipeline_multi.py # Multi-instance variant of the above
│       └── dataset.py                 # Subgraph sampling, loading, and graph construction
│
└── config/
    └── GT2Image.yaml                  # Training and evaluation configuration
```












- 【Modality-level】Story Generation 【GT2Text】

  

  - Traditional: Generate a story from one or more images
  - MAG-specific: Similar to GT2Image — locate a sequence of images in MAG to create graph prompt tokens, combine with text prompt to generate the story

  

- 【Modality-level】Fusion Response 【G2TextImage】

  

  - Traditional: Model generates both text and image as part of the response
  - MAG-specific: Given a query (text prompt), retrieve relevant modality info from MAG and organize a multimodal response

  

- Summary

  

  - G2Text: Textual annotation for node/subgraph
  - G2Image: Visual annotation for node/subgraph
  - GT2Image: Image editing or reconstruction
  - GT2Text: Story generation
  - G2TextImage: Multimodal response generation





## Dataset Description

Below datasets can be found in either https://huggingface.co/datasets/Sherirto/MAGB or  https://huggingface.co/datasets/mm-graph-org/mm-graph .

### books-lp

A **book preference graph** for link‐prediction: predict which books a user will like together.  
#### Common evaluation indicators
    MRR、H@1、H@10
#### Component file description
* `books-lp-images.tar`: Compressed raw book‐cover images (file name = node ID).  
* `books-lp-raw-text.jsonl`: JSONL of `{ "id":…, "text":… }` with book titles/descriptions.  
* `node_mapping.pt`: Dict aligns raw node IDs to graph indices.  
* `lp-edge-split.pt`: Train/validation/test edge‐index tensors for link‐prediction.

---

### books-nc
A **book category graph** for node‐classification: assign each book to one of several genres.  
#### Common evaluation indicators
    Accuracy、F1‐Macro
#### Component file description
* `books-nc-images.tar`: Compressed raw book‐cover images.  
* `books-nc-raw-text.jsonl`: JSONL of `{ "id":…, "text":… }` with book titles/descriptions.  
* `node_mapping.pt`: Dict aligns raw node IDs to graph indices.  
* `nc_edges-nodeid.pt`: Undirected edge list tensor for classification.  
* `labels-w-missing.pt`: LongTensor of node labels (–1 = unlabeled).  
* `split.pt`: Dict of train/validation/test node indices (6/1/3 split).

---

### cloth-copurchase
An **apparel co‐purchase graph** for link‐prediction: predict which clothing items are likely to be purchased together.  
#### Common evaluation indicators
    MRR、H@1、H@10
#### Component file description
* `cloth-images.tar`: Compressed raw clothing‐item images.  
* `cloth-raw-text.jsonl`: JSONL of `{ "id":…, "text":… }` with item titles/descriptions.  
* `node_mapping.pt`: Dict aligns raw node IDs to graph indices.  
* `lp-edge-split.pt`: Train/validation/test edge‐index tensors for link‐prediction.

---

### ele-fashion
A **fashion‐item graph** for node‐classification: classify each fashion product into style categories.  
#### Common evaluation indicators
    Accuracy、F1‐Macro
#### Component file description
* `ele-fashion-images.tar`: Compressed raw fashion‐item images.  
* `ele-fashion-raw-text.jsonl`: JSONL of `{ "id":…, "text":… }` with item descriptions.  
* `node_mapping.pt`: Dict aligns raw node IDs to graph indices.  
* `nc_edges-nodeid.pt`: Undirected edge list tensor for classification.  
* `labels-w-missing.pt`: LongTensor of node labels (–1 = unlabeled).  
* `split.pt`: Dict of train/validation/test node indices.

---

### Grocery
A **grocery‐item graph** for link‐prediction: predict which grocery products are frequently bought together.  
#### Common evaluation indicators
    MRR、H@1、H@10
#### Component file description
* `Grocery.csv`: CSV of `{ id, title, description }` for each product.  
* `GroceryImages.tar.gz`: Compressed raw product images.  
* `GroceryGraph.pt`: DGLGraph object containing adjacency and feature placeholders.

---

### mm-codex-m
A **knowledge graph completion dataset**: predict missing entity–relation–entity triples in a medium‐scale KG.  
#### Common evaluation indicators
    MRR、H@1、H@10
#### Component file description
* `mm-codex-m/`: Directory containing `Graph.pt`, raw‐text JSONL, `node_mapping.pt` and split files for the medium CoDEx graph.  
* `codex-images.tar`: Compressed raw entity images.

---

### mm-codex-s
A **knowledge graph completion dataset**: predict missing triples in a small‐scale KG.  
#### Common evaluation indicators
    MRR、H@1、H@10
#### Component file description
* `mm-codex-s/`: Directory containing `Graph.pt`, raw‐text JSONL, `node_mapping.pt` and split files for the small CoDEx graph.  
* `codex-images.tar`: Compressed raw entity images.

---

### Movies
A **movie attribute graph** for node‐classification and link‐prediction: classify movie genres and recommend related movies.  
#### Common evaluation indicators
    Accuracy、F1‐Macro(classification)  
    MRR、H@1、H@10(link‐prediction)  
#### Component file description
* `Movies.csv`: CSV of `{ id, title, description }` for each movie.  
* `MoviesImages.tar.gz`: Compressed raw movie‐poster images.  
* `MoviesGraph.pt`: DGLGraph object containing adjacency and feature placeholders.

---

### Reddit-M
A **social-media post graph** for node‐classification and link‐prediction: classify post categories and predict co‐comment interactions.  
#### Common evaluation indicators
    Accuracy、F1‐Macro(classification)  
    MRR、H@1、H@10(link‐prediction)  
#### Component file description
* `RedditM.csv`: CSV of `{ id, title, description }` for each post.  
* `RedditMImages_parta`: Archive of raw avatar images (part A).  
* `RedditMGraph.pt`: Serialized DGLGraph object.

---

### Reddit-S
A **social-media post graph** for link‐prediction: predict co‐comment relationships in a small Reddit graph.  
#### Common evaluation indicators
    MRR、H@1、H@10
#### Component file description
* `RedditS.csv`: CSV of `{ id, title, description }` for each post.  
* `RedditSImages.tar.gz`: Compressed raw avatar images.  
* `RedditSGraph.pt`: Serialized DGLGraph object.

---

### sports-copurchase
A **sports equipment co‐purchase graph** for link‐prediction: predict which sports items are frequently bought together.  
#### Common evaluation indicators
    MRR、H@1、H@10
#### Component file description
* `sports-images.tar`: Compressed raw sports‐equipment images.  
* `sports-raw-text.jsonl`: JSONL of `{ "id":…, "text":… }` with item descriptions.  
* `node_mapping.pt`: Dict aligns raw node IDs to graph indices.  
* `lp-edge-split.pt`: Train/validation/test edge‐index tensors for link‐prediction.

---

### Toys
A **toy product graph** for node‐classification and link‐prediction: classify toy categories and recommend related toys.  
#### Common evaluation indicators
    Accuracy、F1‐Macro(classification)  
    MRR、H@1、H@10(link‐prediction)  
#### Component file description
* `Toys.csv`: CSV of `{ id, title, description }` for each toy.  
* `ToysImages.tar.gz`: Compressed raw toy images.  
* `ToysGraph.pt`: DGLGraph object containing adjacency and feature placeholders.
<<<<<<< HEAD

- - 
