"""CLIP prompt adherence: does the picture show what was asked for?

This is the only automatic metric here, and it measures exactly one thing - agreement
between the image and the words. It says nothing about whether the image is *good*. A
plausible, well-composed photograph of the wrong subject scores badly; a mangled six-
fingered mess of the right subject scores well. Every table in the README that uses it is
answering "was the prompt followed", never "was the output pretty", and the step-count
result would be dishonest if it were read the other way.

Being cross-modal, the numbers live around 0.20-0.35 rather than near 1. CLIP's image and
text encoders share a space but not a region of it, so the useful comparison is between two
of these scores, never between one of them and an image-to-image similarity.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from PIL import Image

MODEL = os.environ.get("SCORE_MODEL", "openai/clip-vit-base-patch32")


@lru_cache(maxsize=1)
def _clip():
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    return model, CLIPProcessor.from_pretrained(MODEL), device


@torch.inference_mode()
def adherence(images: list[Image.Image], prompts: list[str]) -> list[float]:
    """Cosine between each image and its prompt. Same length in, same length out."""
    if len(images) != len(prompts):
        raise ValueError("images and prompts must line up")
    if not images:
        return []

    model, processor, device = _clip()
    batch = processor(text=prompts, images=images, return_tensors="pt",
                      padding=True, truncation=True).to(device)

    # The full forward, not get_image_features/get_text_features: those two have changed
    # return type across transformers releases (they hand back the raw vision output rather
    # than the projected embedding in some), whereas CLIPOutput.image_embeds and
    # .text_embeds have always been the projected, comparable vectors.
    output = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"],
                   pixel_values=batch["pixel_values"])
    image_vectors, text_vectors = output.image_embeds, output.text_embeds

    image_vectors = image_vectors / image_vectors.norm(dim=-1, keepdim=True)
    text_vectors = text_vectors / text_vectors.norm(dim=-1, keepdim=True)
    return (image_vectors * text_vectors).sum(-1).float().cpu().tolist()


def score(image: Image.Image, prompt: str) -> float:
    return adherence([image], [prompt])[0]


def release() -> None:
    _clip.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
