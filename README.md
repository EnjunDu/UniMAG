# UniMAG
**MultiModal Attribute Graph Pipelines**

The datasets are given in huggingface [https://huggingface.co/datasets/enjun-collab/MMAG](https://huggingface.co/datasets/enjun-collab/MMAG) .



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

  

### Graph-centric tasks 【Yushuo LI】:

  

  - **Semantic Level** - Node-level: Node classification, Node clustering
  - **Semantic Level** - Edge-level: Edge existence prediction, Edge classification
  - **Semantic Level** - Graph-level: Graph classification, Community detection (subgraphs)

  

### Multimodal-centric tasks



#### **Quality Evaluation Tasks**【Yilong ZUO】



- **Modality-level**: Modality Matching

  

  - Traditional: Input any image and text, get respective embeddings, return a matching score
  - MAG-specific: Given any image-text pair from a node in MAG, compute a matching score (e.g., CLIP-score) based on embeddings

  

- **Modality-level**: Modality Retrieval

  

  - Traditional: Given an image or text, obtain a query embedding and retrieve the most relevant item from a pool of candidates
  - MAG-specific: Use any image-text from a MAG node as a query and return relevant items from others

  

- **Modality-level**: Modality Alignment

  

  - Traditional: Fine-grained modality matching focusing on matching degree (detailed descriptions)
  - MAG-specific: Given any image-text pair from a MAG node, return fine-grained alignment details based on embeddings

  



#### **Creative Generation Tasks**



- MAG-specific【Modality/Semantic level】graph (image, text) + prompt → text 【G2Text】【Yaxin DENG】

  

  - Input: Current node embedding + (optional) contextual info around the node [subgraph]
  - Multimodal interaction: Integrate embedding with prompt template and input to the language model
  - Output: Summary and analysis of the node, including its attributes and neighborhood info

  

- 【Modality/Semantic level】Image Annotation 【G2Image】【Zhenning ZHANG】

  

  - Traditional: Generate an explanatory image from a text
  - MAG-specific: Similar to G2Text, combine embeddings and prompt, then input to image generator

  

- MAG-specific【Modality-level】graph (image) + text [prompt] → image 【GT2Image】【Sicheng LIU】

  

  - Input: Embeddings of current node (multiple images) + (optional) context [subgraph] + text description
  - Multimodal interaction: Use image/text embeddings as graph/text prompt tokens for diffusion model
  - Output: Image satisfying the textual instruction
  - Belongs to the same category as traditional **image editing or reconstruction**

  

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

