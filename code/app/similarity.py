import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm


PrintFn = Callable[[str, str], None]


def print_fn(msg: str, fn_name: str = "") -> None:
    """
    Custom print function to handle printing in a specific format.
    """
    print(f"({fn_name}) {msg}" if fn_name else msg)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_path_from_yaml(
    yaml_file_path: Path, print_fn: PrintFn = print_fn
) -> list[str] | str | None:
    def print_(msg: str):
        print_fn(msg, fn_name="get_train_path_from_yaml")

    try:
        with open(yaml_file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        base_path = data.get("path", "")
        train_paths = data.get("train", [])

        if isinstance(train_paths, str):
            train_paths = [train_paths]

        full_train_paths: list[str] = []
        for train_path in train_paths:
            if base_path:
                full_path = os.path.join(base_path, train_path)
            else:
                full_path = train_path

            full_train_paths.append(full_path)

        if len(full_train_paths) == 1:
            return full_train_paths[0]
        else:
            return full_train_paths

    except Exception as e:
        print_(
            f"Error reading YAML {yaml_file_path}: {e}",
            fn_name="get_train_path_from_yaml",
        )
        return None


def load_images_from_folder(
    folder_path: str, max_images: int = 50, seed: int = 42
) -> list[str]:
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_paths.sort()

    if len(image_paths) > max_images:
        np.random.seed(seed)
        image_paths = np.random.choice(image_paths, max_images, replace=False).tolist()
        image_paths.sort()

    return image_paths


def load_images_from_yaml_paths(
    yaml_paths: str | list[str],
    max_images: int = 50,
    seed: int = 42,
    print_fn: PrintFn = print_fn,
) -> list[str]:
    def print_(msg: str):
        print_fn(msg, fn_name="load_images_from_yaml_paths")

    all_image_paths: list[str] = []

    if isinstance(yaml_paths, str):
        yaml_paths = [yaml_paths]

    yaml_paths.sort()

    images_per_path = max_images // len(yaml_paths)

    for i, path in enumerate(yaml_paths):
        if os.path.exists(path):
            try:
                path_seed = seed + i * 100
                folder_images = load_images_from_folder(
                    path, max_images=images_per_path, seed=path_seed
                )
                all_image_paths.extend(folder_images)
                print_(f"Loaded {len(folder_images)} images from: {path}")
            except Exception as e:
                print_(f"Error loading from {path}: {e}")
        else:
            print_(f"Warning: Path not found: {path}")

    all_image_paths.sort()
    print_(f"Total images loaded: {len(all_image_paths)}")
    return all_image_paths


def extract_embeddings(
    model: nn.Module,
    image_paths: list[str],
    batch_size: int,
    transform: transforms.Compose,
    print_fn: PrintFn = print_fn,
):
    def print_(msg: str):
        print_fn(msg, fn_name="extract_embeddings")

    model.eval()
    embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                tensor = transform(image)
                batch_tensors.append(tensor)
            except Exception as e:
                print_(f"Error loading image {path}: {e}")
                continue

        if batch_tensors:
            batch = torch.stack(batch_tensors)

            with torch.no_grad():
                features = model(batch)
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                embeddings.append(features.cpu())

    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)


def compute_similarity(
    embeddings_A: torch.Tensor, embeddings_B: torch.Tensor, seed: int = 42
) -> dict[str, float] | dict[str, str]:
    if len(embeddings_A) == 0 or len(embeddings_B) == 0:
        return {"error": "One of the embedding sets is empty"}

    centroid_A = torch.mean(embeddings_A, dim=0)
    centroid_B = torch.mean(embeddings_B, dim=0)
    centroid_sim = F.cosine_similarity(centroid_A, centroid_B, dim=0)

    sample_size = min(50, len(embeddings_A), len(embeddings_B))

    torch.manual_seed(seed)
    sample_A = embeddings_A[torch.randperm(len(embeddings_A))[:sample_size]]

    torch.manual_seed(seed + 1)
    sample_B = embeddings_B[torch.randperm(len(embeddings_B))[:sample_size]]

    pairwise_sim = F.cosine_similarity(
        sample_A.unsqueeze(1), sample_B.unsqueeze(0), dim=2
    )
    pairwise_mean = torch.mean(pairwise_sim)

    combined_sim = 0.7 * centroid_sim + 0.3 * pairwise_mean

    return {
        "centroid_similarity": centroid_sim.item(),
        "pairwise_mean": pairwise_mean.item(),
        "combined_similarity": combined_sim.item(),
    }


def check_domain_similarity(
    current_domain_id: int,
    model: nn.Module,
    transform: transforms.Compose,
    similarity_threshold: float,
    max_images: int,
    batch_size: int,
    yaml_fn: Callable[[int], tuple[Path, Path]],
    seed: int = 1,
    print_fn: PrintFn = print_fn,
):
    from typing import cast

    def print_(msg: str):
        print_fn(msg, fn_name="check_domain_similarity_for_client")

    if current_domain_id == 0:
        print_("No similarity check needed for domain 0.")
        return False

    set_seed(seed)

    print_(f"\n{'=' * 60}")
    print_(
        f"SIMILARITY CHECK: Domain {current_domain_id - 1} vs Domain {current_domain_id}"
    )
    print_(f"{'=' * 60}")

    yaml_prev, yaml_curr = yaml_fn(current_domain_id)

    print_(f"Check: {yaml_prev} vs {yaml_curr}")

    if not yaml_prev.exists() or not yaml_curr.exists():
        if not yaml_prev.exists():
            print_(f"Warning: Prev YAML file {yaml_prev} not found.")
        if not yaml_curr.exists():
            print_(f"Warning: Curr YAML file {yaml_curr} not found.")
        return False

    print_("1. Reading YAML files ...")
    train_paths_prev = get_train_path_from_yaml(yaml_prev, print_fn=print_fn)
    train_paths_curr = get_train_path_from_yaml(yaml_curr, print_fn=print_fn)

    if train_paths_prev is None or train_paths_curr is None:
        if train_paths_prev is None:
            print_(f"Error loading prev YAML: {yaml_prev}")
        if train_paths_curr is None:
            print_(f"Error loading curr YAML: {yaml_curr}")
        return False

    print_("2. Loading images...")
    print_("Dataset A:")
    image_prev = load_images_from_yaml_paths(
        train_paths_prev, max_images=max_images, seed=seed, print_fn=print_fn
    )

    print_("\nDataset B:")
    image_curr = load_images_from_yaml_paths(
        train_paths_curr, max_images=max_images, seed=seed + 1000, print_fn=print_fn
    )

    if len(image_prev) == 0 or len(image_curr) == 0:
        if len(image_prev) == 0:
            print_("No images found in Dataset A.")
        if len(image_curr) == 0:
            print_("No images found in Dataset B.")
        return False

    print_("3. Extracting embeddings...")
    print_("Computing Dataset A...")
    embeddings_prev = extract_embeddings(
        model, image_prev, batch_size=batch_size, transform=transform, print_fn=print_fn
    )

    print_("Computing Dataset B...")
    embeddings_curr = extract_embeddings(
        model, image_curr, batch_size=batch_size, transform=transform, print_fn=print_fn
    )

    print_("4. Computing similarity...")
    similarities = compute_similarity(embeddings_prev, embeddings_curr, seed=seed)

    if "error" in similarities:
        error = cast(dict[str, str], similarities)["error"]
        print_(f"Error calculating similarity: {error}")
        return False

    similarities = cast(dict[str, float], similarities)

    combined_sim = similarities["combined_similarity"]
    print_(f"Similarity pairwise: {similarities['pairwise_mean']:.4f}")
    print_(f"Similarity centroids: {similarities['centroid_similarity']:.4f}")
    print_(f"Similarity combined: {similarities['combined_similarity']:.4f}")

    similarity_threshold_reached = combined_sim > similarity_threshold

    if similarity_threshold_reached:
        print_(f"SIMILARITY {combined_sim:.4f} > {similarity_threshold}.")
    else:
        print_(f"SIMILARITY {combined_sim:.4f} <= {similarity_threshold}.")

    print_(f"{'=' * 60}\n")
    return similarity_threshold_reached
