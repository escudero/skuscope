"""DINO embeddings: a visual fingerprint for each product crop."""

import numpy as np
import torch
from PIL import Image

from skuscope.config import EMBEDDERS

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pads instead of center-cropping, so tall bottles keep their cap and base."""
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), tuple(int(m * 255) for m in MEAN))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def preprocess(image: Image.Image) -> torch.Tensor:
    image = pad_to_square(image.convert("RGB")).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1))


class Embedder:
    """Turns product images into L2-normalized vectors, so dot product = cosine similarity."""

    def __init__(self, name: str = "dinov2-small", device: str | None = None):
        from transformers import AutoModel

        self.name = name
        self.device = device or pick_device()
        self.model = AutoModel.from_pretrained(EMBEDDERS[name]).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([preprocess(im) for im in images[start : start + batch_size]])
            cls = self.model(pixel_values=batch.to(self.device)).pooler_output
            chunks.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
        if not chunks:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.concatenate(chunks)
