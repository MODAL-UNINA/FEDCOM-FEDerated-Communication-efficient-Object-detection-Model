from typing import Any
from pathlib import Path

import torch

FRAMEWORKS = ["Baseline", "FCL", "FEDCOM"]
SCENARIOS = [1, 2, 3]
DOMAIN_IDS = [0, 1, 2]


def validate_framework(framework: Any, frameworks: list[str] = FRAMEWORKS) -> str:
    if framework not in frameworks:
        raise ValueError(
            f"Unsupported framework: {framework}. Supported values are 'Baseline', 'FCL', or 'FEDCOM'."
        )
    return framework


def validate_model(
    model_name: Any,
    base_model_path: Path,
    not_exists_ok: bool = False,
) -> str:
    """
    Validates the model name. It must be a string ending with '.yaml'.
    If `not_exists_ok` is True, the model may not exist in the specified directory.
    """
    if not isinstance(model_name, str):
        raise ValueError(f"Invalid model name: {model_name}.")

    if not model_name.endswith(".yaml"):
        raise ValueError(f"Model name must be a YAML file, got: {model_name}.")

    if not not_exists_ok and not (base_model_path / model_name).exists():
        raise FileNotFoundError(
            f"Model {model_name} does not exist in the base model directory."
        )

    return model_name


def validate_scenario(scenario: Any, scenarios: list[int] = SCENARIOS) -> int:
    if scenario not in scenarios:
        raise ValueError(
            f"Invalid scenario: {scenario}. It must be a non-negative integer."
        )
    return scenario


def validate_domain_id(domain_id: Any, domain_ids: list[int] = DOMAIN_IDS) -> int:
    if domain_id not in domain_ids:
        raise ValueError(
            f"Invalid domain_id: {domain_id}. Supported values are 0, 1, or 2."
        )

    return domain_id


def validate_server_address(server_address: Any) -> str:
    if not isinstance(server_address, str) or not server_address:
        raise ValueError(f"Invalid server address: {server_address}.")

    # Let FLWR validate it later
    return server_address


def validate_gpu_id(gpu_id: Any) -> int:
    if not isinstance(gpu_id, int):
        raise TypeError("GPU ID must be an integer.")

    if gpu_id < 0:
        raise ValueError(
            f"Invalid GPU ID: {gpu_id}. It must be a non-negative integer."
        )

    # check if the GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your PyTorch installation or GPU setup."
        )

    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU ID: {gpu_id}. It exceeds the number of available GPUs: {torch.cuda.device_count() - 1}."
        )

    return gpu_id


def validate_max_num_clients(
    max_num_clients: Any, default_max_num_clients: int = 3
) -> int:
    if max_num_clients != default_max_num_clients:
        raise ValueError(
            f"Invalid max_num_clients: {max_num_clients}. It must be {default_max_num_clients}."
        )

    return max_num_clients


def validate_rounds(rounds: Any) -> int:
    if not isinstance(rounds, int) or rounds <= 0:
        raise ValueError(f"Invalid rounds: {rounds}. It must be a positive integer.")

    return rounds
