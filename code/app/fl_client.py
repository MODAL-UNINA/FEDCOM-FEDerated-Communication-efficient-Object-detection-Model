import csv
import os
from pathlib import Path
from typing import Callable

import yaml
from flwr.client import NumPyClient
from ultralytics import YOLO

from .utils import (
    get_freeze_filter_layers,
    get_state_dict_all_parameters,
    get_state_dict_trainable_parameters,
)


def freeze_layer(N_freezing_layers):
    def callback(trainer):
        model = trainer.model
        num_freeze = N_freezing_layers
        freeze = get_freeze_filter_layers(num_freeze, inner=True)
        for k, v in model.named_parameters():
            v.requires_grad = True
            if any(k.startswith(x) for x in freeze):
                v.requires_grad = False

    return callback


class FLClient(NumPyClient):
    def __init__(
        self,
        domain_id: int,
        num_epochs: int,
        batch_size: int,
        model_path: Path,
        n_train_freezing_layers: int,
        n_comm_freezing_layers: int,
        train_model_name: str,
        data_yaml: str,
        runs_path: Path,
        print_fn: Callable[[str], None] = print,
        csv_path: Path | None = None,
        cumulative_data_path: Path | None = None,
        cumulative_data_name: str | None = None,
        cumulative_csv_path: Path | None = None,
    ):
        self.model = YOLO(model_path)

        self.model_path = model_path
        self.domain_id = domain_id
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.data_yaml = data_yaml
        self.runs_path = runs_path
        self.print_fn = print_fn
        self.n_train_freezing_layers = n_train_freezing_layers
        self.n_comm_freezing_layers = n_comm_freezing_layers
        self.train_model_name = train_model_name
        self.csv_path = csv_path
        self.cumulative_data_path = cumulative_data_path
        self.cumulative_data_name = cumulative_data_name
        self.cumulative_csv_path = cumulative_csv_path

        if domain_id not in [0, 1, 2]:
            raise ValueError(
                f"Unsupported domain_id: {domain_id}. Supported values are 0, 1, or 2."
            )

    def _setup_layer_freezing(self):
        if self.n_train_freezing_layers > 0:
            self.print_fn(f"Freezing first {self.n_train_freezing_layers} layers ...")
            if len(self.model.callbacks["on_train_start"]) > 0:
                self.model.callbacks["on_train_start"] = []
            self.model.add_callback(
                "on_train_start", freeze_layer(self.n_train_freezing_layers)
            )
        else:
            self.print_fn("Training without freezing layers..")

    def _get_all_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _get_initial_parameters(self):
        return self._get_all_parameters()

    def _get_trainable_parameters(self, n_freezing_layers):
        freeze = get_freeze_filter_layers(n_freezing_layers)
        return [
            val.cpu().numpy()
            for k, val in self.model.state_dict().items()
            if not any(k.startswith(x) for x in freeze)
        ]

    def _get_parameters(self, n_freezing_layers):
        if n_freezing_layers > 0:
            return self._get_trainable_parameters(n_freezing_layers)

        return self._get_all_parameters()

    def _get_train_parameters(self):
        return self._get_parameters(self.n_train_freezing_layers)

    def _get_comm_parameters(self):
        return self._get_parameters(self.n_comm_freezing_layers)

    def get_parameters(self, config):
        if "current_round" not in config or config["current_round"] == 0:
            print(
                "This is the initialization round of training. Returning initial parameters."
            )
            return self._get_initial_parameters()

        if "comm_parameters" in config and config["comm_parameters"]:
            print(
                f"Returning trainable parameters with "
                f"{self.n_comm_freezing_layers} communication frozen layers."
            )
            return self._get_comm_parameters()

        if "train_parameters" in config and config["train_parameters"]:
            print(
                f"Returning trainable parameters with "
                f"{self.n_train_freezing_layers} train frozen layers."
            )
            return self._get_train_parameters()

        return self._get_all_parameters()

    def _set_parameters(self, parameters, server_round):
        if self.n_comm_freezing_layers > 0 and server_round > 1:
            state_dict = get_state_dict_trainable_parameters(
                parameters, self.model, self.n_comm_freezing_layers
            )
        else:
            state_dict = get_state_dict_all_parameters(parameters, self.model)

        self.model.eval()
        self.model.load_state_dict(state_dict, strict=True)

    def _count_train_images(self, yaml_path):
        with open(yaml_path, "r") as file:
            dataset_yaml = yaml.safe_load(file)

        path = Path(dataset_yaml["path"])

        num_images = 0
        for train_dir in dataset_yaml["train"]:
            train_images = len(
                [
                    f
                    for f in (path / train_dir).iterdir()
                    if f.name.endswith((".jpg", ".png", ".jpeg"))
                ]
            )
            assert train_images > 0, (
                f"No training images found in {path / train_dir}. "
                "Please check the dataset YAML file."
            )
            num_images += train_images
        return num_images

    def fit(self, parameters, config):
        self.print_fn(f"Received config: {config}")

        if "only_communication" in config and config["only_communication"]:
            self.print_fn(
                f"Only communication mode."
                f" Returning {self.n_comm_freezing_layers} communication frozen layers."
            )
            return parameters, 0, {}

        len_dataset = self._count_train_images(self.data_yaml)

        current_round = config["current_round"]

        self.print_fn(f"=== TRAINING ROUND {current_round} ===")

        self.print_fn(
            f"{self.n_comm_freezing_layers} | {len(parameters)} parameters received"
        )

        self._set_parameters(parameters, current_round)

        self._setup_layer_freezing()
        self.print_fn(
            f"Training with {self.n_train_freezing_layers} frozen layers and "
            f"{self.n_comm_freezing_layers} communication frozen layers..."
        )

        results_1 = self.model.train(
            data=self.data_yaml,
            project=self.runs_path,
            batch=self.batch_size,
            epochs=self.num_epochs,
            name=self.train_model_name,
            exist_ok=True,
        )

        class_names = list(results_1.names.values())

        updated_parameters = self._get_comm_parameters()
        self.print_fn(
            f"Training completed. Sending {len(updated_parameters)} parameters to the server"
        )

        # if self.post_train_fn is not None:
        #     self.post_train_fn(self.model, results_1, current_round, config["total_rounds"])

        # Saving training results
        if self.csv_path is not None:
            csv_path = self.csv_path
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(["Class", "mAP50", "mAP50-95"])

                # Overall/All classes row
                writer.writerow(
                    ["all", f"{results_1.box.map50:.4f}", f"{results_1.box.map:.4f}"]
                )

                # Per-class metrics
                for i, class_name in enumerate(class_names):
                    map50_class = (
                        results_1.box.ap50[i] if i < len(results_1.box.ap50) else 0.0
                    )
                    map50_95_class = (
                        results_1.box.ap[i] if i < len(results_1.box.ap) else 0.0
                    )
                    writer.writerow(
                        [class_name, f"{map50_class:.4f}", f"{map50_95_class:.4f}"]
                    )

        # TODO move domain_id out of the client
        if self.domain_id > 0:
            if current_round == config["total_rounds"]:
                if (
                    self.cumulative_data_path is not None
                    and self.cumulative_data_name is not None
                ):
                    results_2 = self.model.val(
                        data=self.cumulative_data_path,
                        name=self.cumulative_data_name,
                        batch=self.batch_size,
                        exist_ok=True,
                        project=self.runs_path,
                        # split="test",
                    )

                    class_names = list(results_2.names.values())

                    if self.cumulative_csv_path is not None:
                        csv_path = self.cumulative_csv_path
                        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                        with open(csv_path, "w", newline="") as f:
                            writer = csv.writer(f)

                            # Header
                            writer.writerow(["Class", "mAP50", "mAP50-95"])

                            # Overall/All classes row
                            writer.writerow(
                                [
                                    "all",
                                    f"{results_2.box.map50:.4f}",
                                    f"{results_2.box.map:.4f}",
                                ]
                            )

                            # Per-class metrics
                            for i, class_name in enumerate(class_names):
                                map50_class = (
                                    results_2.box.ap50[i]
                                    if i < len(results_2.box.ap50)
                                    else 0.0
                                )
                                map50_95_class = (
                                    results_2.box.ap[i]
                                    if i < len(results_2.box.ap)
                                    else 0.0
                                )
                                writer.writerow(
                                    [
                                        class_name,
                                        f"{map50_class:.4f}",
                                        f"{map50_95_class:.4f}",
                                    ]
                                )

        return updated_parameters, len_dataset, {}
