# Embedding Converter

This folder contains scripts for converting raw multimodal data (e.g., text, image, audio) into vector embeddings. These embeddings serve as the input for downstream tasks in the benchmark pipeline.

---

## Accessing the Embeddings

The logic within this module is focused on the **creation and management** of feature embeddings.

If your goal is to **use** these pre-computed embeddings in your own models or experiments, you do not need to interact with the code in this directory. Instead, please refer to the documentation for the `EmbeddingManager`, which provides a simple, high-level API for accessing all features.

> **See: [Documentation for the `EmbeddingManager`](../utils/README.md)**

---

## How to Add a New Encoder

The `embedding_converter` is designed for easy, "plug-and-play" extension. If you want to integrate a new model to generate embeddings, you only need to create a new encoder class, and the system will automatically discover and register it.

### Core Concepts

-   **`BaseEncoder`**: An abstract class that defines the required interface for all encoders. Any new encoder **must** inherit from it and implement its abstract methods.
-   **`EncoderFactory`**: A factory that uses a decorator (`@EncoderFactory.register`) to find and register encoder classes.
-   **Automatic Discovery**: The `embedding_converter/encoders/__init__.py` file automatically scans its directory and imports all Python files, triggering the registration of any decorated encoder classes. **You do not need to manually edit this file.**

### Step 1: Create the Encoder File

Create a new Python file in the `embedding_converter/encoders/` directory. For example, `my_encoder.py`.

### Step 2: Implement the Encoder Class

In your new file, define a class that inherits from `BaseEncoder` and implement the required methods.

**Example: A New Multimodal Encoder `MyEncoder`**

```python
# In: embedding_converter/encoders/my_encoder.py

import torch
import numpy as np
from PIL import Image
from typing import List
from transformers import AutoModel, AutoProcessor

# Import the necessary base class and the factory
from ..base_encoder import BaseEncoder
from ..encoder_factory import EncoderFactory

# Register the encoder with a unique name. This name is used in config files.
@EncoderFactory.register("my_encoder")
class MyEncoder(BaseEncoder):
    """
    An example of a new, custom multimodal encoder.
    """
    def _load_model(self, **kwargs):
        """
        Load the model and processor from Hugging Face.
        This is called automatically by the BaseEncoder's __init__.
        """
        # The `self.model_name`, `self.cache_dir`, and `self.device` attributes
        # are inherited from the base class and populated from the config file.
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            **kwargs
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        print(f"MyEncoder model '{self.model_name}' loaded to {self.device}.")

    def get_native_embedding_dim(self) -> int:
        """Return the model's native embedding dimension."""
        return self.model.config.hidden_size

    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode a batch of texts."""
        # Note: The computation happens on `self.device` (e.g., GPU).
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        
        # The final output is always a numpy array on the CPU.
        return embeddings.cpu().numpy()

    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """Encode a batch of images."""
        images = [Image.open(path).convert("RGB") for path in image_paths if path]
        if not images:
            return np.array([]).reshape(0, self.get_native_embedding_dim())
            
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            
        return embeddings.cpu().numpy()

    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        """Encode a batch of image-text pairs."""
        # This is a placeholder. The actual implementation depends heavily on the model.
        # For this example, we'll just return the text embeddings.
        return self.encode_text(texts, **kwargs)

```

### Step 3: Use the New Encoder in a Config File

You can now use your new encoder by referencing its registered name (`"my_encoder"`) in a YAML configuration file. The structure must follow the `pipeline_settings`, `encoder_settings`, `dataset_settings` format.

**Example `config.yaml`:**

```yaml
# 1. Pipeline Settings
pipeline_settings:
  force_reprocess: true
  batch_size: 16

# 2. Encoder Settings
encoder_settings:
  # Use the unique name you registered with the factory
  encoder_type: "my_encoder"
  
  # The Hugging Face path for the model you want to load
  model_name: "openai/clip-vit-base-patch32"
  
  # Specify the device, e.g., "cuda:0", "cpu"
  device: "cuda:0"
  
  # (Optional) Specify target dimensions to trigger dimension reduction
  # target_dimensions:
  #   text: 768
  #   image: 768

# 3. Dataset Settings
dataset_settings:
  datasets_to_process:
    - "books-nc-50" # A small sample dataset for testing
  modalities_to_process:
    - "text"
    - "image"
```

### Advanced: Creating an Encoder with Dimension Reduction

If you want your encoder to support different output dimensions, you can create a second registered class that inherits from your base encoder and uses the `DimensionReducer` utility.

**Example:**

```python
# In the same file: embedding_converter/encoders/my_encoder.py

from ..utils.dimension_reducer import DimensionReducer
from typing import Literal

@EncoderFactory.register("my_encoder_with_dim")
class MyEncoderWithDimension(MyEncoder):
    """
    An extended version of MyEncoder that supports dimension reduction.
    """
    def __init__(self, target_dimension: int, reduction_method: Literal["linear", "pca"] = "linear", **kwargs):
        # Initialize the base encoder first
        super().__init__(**kwargs)
        self.target_dimension = target_dimension
        
        # Get the native dimension from the base class
        model_native_dim = super().get_native_embedding_dim()
        
        # Initialize the reducer
        self.dimension_reducer = DimensionReducer(
            input_dim=model_native_dim,
            output_dim=target_dimension,
            method=reduction_method
        )

    def _apply_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        # The reducer expects a 2D array. Handle empty cases.
        if embeddings.ndim == 2 and embeddings.shape[0] > 0:
            return self.dimension_reducer.transform(embeddings)
        return embeddings

    # Override the encoding methods to apply the reduction
    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        native_embeddings = super().encode_text(texts, **kwargs)
        return self._apply_reduction(native_embeddings)
    
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        native_embeddings = super().encode_image(image_paths, **kwargs)
        return self._apply_reduction(native_embeddings)

    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        native_embeddings = super().encode_multimodal(texts, image_paths, **kwargs)
        return self._apply_reduction(native_embeddings)
        
    # Override this to report the new target dimension
    def get_native_embedding_dim(self) -> int:
        return self.target_dimension
```

To use this version, you would set `encoder_type: "my_encoder_with_dim"` in your config and provide the `target_dimensions` for the modalities you want to reduce. The `main.py` pipeline will automatically handle passing the correct parameters.
