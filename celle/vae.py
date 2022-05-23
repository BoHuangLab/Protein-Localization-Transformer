import os
import urllib
from tqdm import tqdm
from math import sqrt, log
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import importlib

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

# constants

CACHE_PATH = os.path.expanduser("~/.cache/dalle")

# helpers methods


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def load_model(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location=torch.device("cpu"))


def map_pixels(x, eps=0.1):
    return (1 - 2 * eps) * x + eps


def unmap_pixels(x, eps=0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)

def make_contiguous(module):
    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())


# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()

        assert(vqgan_model_path is not None)

        model_path = vqgan_model_path
        config_path = vqgan_config_path

        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        state = torch.load(model_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state, strict=False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        # f as used in https://github.com/CompVis/taming-transformers#overview-of-pretrained-models
        f = (
            config.model.params.ddconfig.resolution
            / config.model.params.ddconfig.attn_resolutions[0]
        )
        self.num_layers = int(log(f) / log(2))
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)

        self._register_external_parameters()

    def get_codebook_indices(self, img):
        b = img.shape[0]
        # img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, "b h w -> b (h w)", b=b)
        return rearrange(indices, "(b n) -> b n", b=b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes=self.num_tokens).float()
        z = (
            one_hot_indices @ self.model.quantize.embed.weight
            if self.is_gumbel
            else (one_hot_indices @ self.model.quantize.embedding.weight)
        )

        z = rearrange(z, "b (h w) c -> b c h w", h=int(sqrt(n)))
        img = self.model.decode(z)

        # img = (img.clamp(-1.0, 1.0) + 1) * 0.5
        return img

    def forward(self, img, optimizer_idx=1):
        return self.model.training_step(img, optimizer_idx=optimizer_idx)
