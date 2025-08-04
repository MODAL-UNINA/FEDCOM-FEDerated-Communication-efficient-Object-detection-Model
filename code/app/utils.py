from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

FloatType = np.float32
FloatArray = NDArray[FloatType]


def get_n_freezing_layers(domain_id: int) -> int:
    if domain_id == 0:
        n_freezing_layers = 0
    elif domain_id == 1:
        n_freezing_layers = 4
    elif domain_id == 2:
        n_freezing_layers = 7
    else:
        raise ValueError(
            f"Unsupported domain_id: {domain_id}. Supported values are 0, 1, or 2."
        )

    return n_freezing_layers


def get_effective_n_freezing_layers(domain_id: int, use_trainable_only: bool) -> int:
    if use_trainable_only:
        return get_n_freezing_layers(domain_id)
    else:
        return 0


def get_freeze_filter_layers(n_freezing_layers: int, inner: bool = False) -> list[str]:
    """
    Returns a list of layer names to freeze based on the number of freezing layers.
    """
    freeze = [
        f"model.model.{x}." if not inner else f"model.{x}."
        for x in range(n_freezing_layers)
    ]
    return freeze


def get_state_dict(params_dict: Iterable[tuple[str, FloatArray]]) -> OrderedDict:
    return OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})


def get_state_dict_all_parameters(
    parameters: list[FloatArray], model: nn.Module
) -> OrderedDict:
    return get_state_dict(zip(model.state_dict().keys(), parameters))


def get_state_dict_trainable_parameters(
    parameters: list[FloatArray], model: nn.Module, n_freezing_layers: int
) -> OrderedDict:
    freeze = get_freeze_filter_layers(n_freezing_layers)

    trainable_params_dict = OrderedDict(
        zip(
            [
                k
                for k in model.state_dict().keys()
                if not any(k.startswith(x) for x in freeze)
            ],
            parameters,
        )
    )

    remaining_params_dict = OrderedDict(
        [
            (k, v)
            for k, v in model.state_dict().items()
            if k not in [k for k, _ in trainable_params_dict.items()]
        ]
    )

    state_dict = OrderedDict(
        {k: v for k, v in remaining_params_dict.items()}
        | {k: torch.from_numpy(v) for k, v in trainable_params_dict.items()}
    )

    assert len(state_dict) == len(model.state_dict()), (
        "Mismatch in number of parameters after setting trainable parameters."
    )

    assert list(model.state_dict().keys()) == list(state_dict.keys()), (
        "Keys mismatch after setting trainable parameters."
    )

    return state_dict


def set_parameters(model: nn.Module, parameters: list[FloatArray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = get_state_dict(params_dict)
    model.eval()
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model: nn.Module) -> list[FloatArray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
