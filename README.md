# UniMAG
**MultiModal Attribute Graph Pipelines**

The datasets are given in huggingface [https://huggingface.co/datasets/enjun-collab/MMAGs](https://huggingface.co/datasets/enjun-collab/MMAG) .



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
