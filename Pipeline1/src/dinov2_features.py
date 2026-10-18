"""Frozen official DINOv2 Small inference with deterministic preprocessing/cache."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import PIL
from PIL import Image, ImageOps
import torch


MODEL_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODEL_NAME = "dinov2_vits14"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
EMBEDDING_DIM = 384


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_image(path):
    """Keep the full field of view; padding becomes zero after normalization."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = INPUT_SIZE / max(width, height)
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(resized_size, Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    canvas = np.empty((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    canvas[:] = MEAN
    left = (INPUT_SIZE - resized_size[0]) // 2
    top = (INPUT_SIZE - resized_size[1]) // 2
    canvas[top:top + resized_size[1], left:left + resized_size[0]] = pixels
    normalized = (canvas - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy())


def load_backbone(cache_root):
    """Use the pinned official Hub entry point and its pretrained checkpoint."""
    cache_root = Path(cache_root)
    torch.hub.set_dir(str(cache_root / "torch_hub"))
    local_repo = cache_root / "torch_hub" / f"facebookresearch_dinov2_{MODEL_COMMIT}"
    if local_repo.is_dir():
        model = torch.hub.load(str(local_repo), MODEL_NAME, source="local", pretrained=True)
    else:
        model = torch.hub.load(
            f"facebookresearch/dinov2:{MODEL_COMMIT}", MODEL_NAME,
            pretrained=True, trust_repo=True, skip_validation=True,
        )
    model.eval().requires_grad_(False)
    assert model.embed_dim == EMBEDDING_DIM
    device = select_device()
    model = model.to(device)
    weights = cache_root / "torch_hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
    config = {
        "model": MODEL_NAME, "official_repository": "facebookresearch/dinov2",
        "source_commit": MODEL_COMMIT, "weights_sha256": sha256_file(weights),
        "representation": "forward_features.x_norm_clstoken", "embedding_dimension": EMBEDDING_DIM,
        "input_size": INPUT_SIZE, "exif_transpose": True, "color": "RGB",
        "resize": "aspect-preserving, round dimensions, Pillow bicubic",
        "padding": "centered, floating-point ImageNet mean RGB", "crop": False,
        "normalization_mean": MEAN, "normalization_std": STD, "dtype": "float32",
        "random_augmentation": False, "torch_version": torch.__version__,
        "pillow_version": PIL.__version__, "numpy_version": np.__version__,
        "device": device.type, "extractor_source_sha256": sha256_file(__file__),
    }
    return model, device, config


class EmbeddingCache:
    """Content-addressed embeddings shared by the three dataset views.

    A model/preprocessing/version signature selects the cache directory. Each
    image is identified by its file-content hash, so unchanged original copies
    share one vector. Saved split files separately retain filenames and labels.
    """

    def __init__(self, cache_root, model, device, config):
        self.model, self.device, self.config = model, device, config
        self.config_json = json.dumps(config, sort_keys=True)
        self.signature = hashlib.sha256(self.config_json.encode()).hexdigest()
        self.directory = Path(cache_root) / "embeddings" / self.signature
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "content_embeddings.npz"
        self.vectors = {}
        if self.path.is_file():
            with np.load(self.path, allow_pickle=False) as saved:
                assert saved["config_json"].item() == self.config_json
                features = saved["features"]
                assert features.shape == (len(saved["hashes"]), EMBEDDING_DIM)
                assert np.isfinite(features).all()
                self.vectors = dict(zip(saved["hashes"].tolist(), features, strict=True))

    def save(self):
        hashes = list(self.vectors)
        temporary = self.directory / "content_embeddings.tmp.npz"
        np.savez_compressed(temporary, hashes=np.asarray(hashes),
                            features=np.stack([self.vectors[key] for key in hashes]),
                            config_json=np.asarray(self.config_json))
        temporary.replace(self.path)

    def extract(self, records, split_name, batch_size=8):
        """Decode at most two full images concurrently; infer small batches only."""
        started = time.perf_counter()
        paths = [Path(row["path"]) for row in records]
        hashes = [sha256_file(path) for path in paths]
        missing = {}
        for key, path in zip(hashes, paths, strict=True):
            if key not in self.vectors:
                missing.setdefault(key, path)
        pending = list(missing.items())
        print(f"{split_name}: {len(records)} rows, {len(pending)} new embeddings, "
              f"{len(records) - len(pending)} reused.", flush=True)
        assert not self.model.training and not any(p.requires_grad for p in self.model.parameters())
        with ThreadPoolExecutor(max_workers=2) as pool, torch.inference_mode():
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                images = list(pool.map(preprocess_image, [path for _, path in batch]))
                inputs = torch.stack(images).to(self.device)
                output = self.model.forward_features(inputs)["x_norm_clstoken"]
                assert output.shape == (len(batch), EMBEDDING_DIM)
                assert not output.requires_grad
                features = output.cpu().numpy().astype(np.float32, copy=True)
                assert np.isfinite(features).all()
                for (key, _), vector in zip(batch, features, strict=True):
                    self.vectors[key] = vector
                self.save()
                done = min(start + batch_size, len(pending))
                if done % 64 == 0 or done == len(pending):
                    print(f"{split_name}: inferred {done}/{len(pending)}; "
                          f"elapsed {time.perf_counter() - started:.1f}s", flush=True)
        features = np.stack([self.vectors[key] for key in hashes])
        assert features.shape == (len(records), EMBEDDING_DIM) and np.isfinite(features).all()
        output_path = self.directory / f"{split_name}.npz"
        np.savez_compressed(
            output_path, features=features, hashes=np.asarray(hashes),
            filenames=np.asarray([row["filename"] for row in records]),
            labels=np.asarray([row["label"] for row in records], dtype=np.int64),
            source_ids=np.asarray([row["source_id"] for row in records]),
            config_json=np.asarray(self.config_json),
        )
        stats = {"split": split_name, "image_count": len(records),
                 "new_embeddings": len(pending), "reused_embeddings": len(records) - len(pending),
                 "elapsed_seconds": time.perf_counter() - started,
                 "batch_size": batch_size, "device": self.device.type,
                 "embedding_dimension": EMBEDDING_DIM}
        return features, stats
