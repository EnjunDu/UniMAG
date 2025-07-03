# Embedding Converter

This folder contains scripts for converting raw multimodal data (e.g., text, image, audio) into vector embeddings. These embeddings serve as the input for downstream tasks in the benchmark pipeline.

---

## Accessing the Embeddings

The logic within this module is focused on the **creation and management** of feature embeddings.

If your goal is to **use** these pre-computed embeddings in your own models or experiments, you do not need to interact with the code in this directory. Instead, please refer to the documentation for the `EmbeddingManager`, which provides a simple, high-level API for accessing all features.

**> See: [Documentation for the `EmbeddingManager`](../utils/README.md)**
