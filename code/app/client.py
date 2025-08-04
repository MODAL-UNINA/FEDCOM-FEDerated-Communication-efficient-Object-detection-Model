import time
import sys
from pathlib import Path

import flwr as fl
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from codecarbon import EmissionsTracker

from app.fl_client import FLClient
from app.similarity import check_domain_similarity
from app.utils import get_n_freezing_layers


def run_client(
    framework: str,
    scenario: int,
    domain_id: int,
    client_id: int,
    gpu_id: int,
    model_name: str,
    server_address: str,
    max_num_clients: int,
    num_epochs: int,
    batch_size: int,
    similarity_threshold: float,
    max_images: int,
    root_path: Path,
):
    if client_id < 0 or client_id >= max_num_clients:
        raise ValueError(
            f"Client ID {client_id} does not exist for id {client_id}. Closing."
        )

    check_similarity = framework != "Baseline"

    n_train_freezing_layers = (
        get_n_freezing_layers(domain_id) if framework != "Baseline" else 0
    )
    n_comm_freezing_layers = (
        get_n_freezing_layers(domain_id) if framework == "FEDCOM" else 0
    )

    node_name = f"client_{client_id}"
    node_name_desc = f"Client {client_id}"

    log_file = (
        root_path
        / "Results"
        / "logs"
        / f"logs_{framework}_scenario_{scenario}_domain_{domain_id}_{node_name}.txt"
    )
    if log_file.exists():
        log_file.unlink()  # Remove the existing log file

    def print_fn(msg: str, fn_name: str = "main"):
        """Prints messages with a prefix."""
        print(f"[{node_name_desc}] (main) {msg}")
        with open(log_file, "a") as f:
            f.write(f"[{node_name_desc}] (main) {msg}\n")

    print_fn(
        f"Starting Federated Learning - Framework {framework}, Scenario {scenario}, Domain {domain_id}"
    )
    print_fn(f"- Server address: {server_address}")
    print_fn(f"- GPU ID: {gpu_id}")

    data_dir = root_path / "configs" / f"scenario_{scenario}"

    if check_similarity:
        if domain_id > 0:
            print_fn("Checking similarity between the domains ...")

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # 2. Feature extractor: EfficientNet-B7 without the classifier
            net = models.efficientnet_b7(
                weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1
            )
            feature_extractor = nn.Sequential(
                *list(net.children())[:-1]
            )  # up to the avg pool layer
            # feature_extractor.eval()

            def yaml_fn(domain_id: int) -> tuple[Path, Path]:
                continual_scenario_path = data_dir / "continual"
                return (
                    continual_scenario_path
                    / f"domain_{domain_id - 1}_client_{client_id}_not_aug.yaml",
                    continual_scenario_path
                    / f"domain_{domain_id}_client_{client_id}_not_aug.yaml",
                )

            should_shutdown = check_domain_similarity(
                current_domain_id=domain_id,
                model=feature_extractor,
                transform=transform,
                similarity_threshold=similarity_threshold,
                max_images=max_images,
                batch_size=batch_size,
                yaml_fn=yaml_fn,
                seed=1,
                print_fn=print_fn,
            )

            # Exit if the similarity threshold is exceeded
            if should_shutdown:
                print_fn(
                    f"🔴 CLIENT {client_id} SHUTTING DOWN DUE TO SIMILARITY "
                    f"EXCEEDING THE THRESHOLD"
                )
                sys.exit(0)

            del net
            del feature_extractor
            del transform

    data_yaml = (
        data_dir
        / ("cumulative" if framework == "Baseline" else "continual")
        / f"domain_{domain_id}_{node_name}.yaml"
    )

    print_fn(f"Configuration loaded for Client {client_id}: {data_yaml}")

    results_dir = root_path / "Results" / framework / f"scenario_{scenario}"

    emissions_dir = results_dir / "Federated_emissions" / node_name
    if not emissions_dir.exists():
        emissions_dir.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=f"domain_{domain_id}",
        output_dir=f"{emissions_dir}",
        gpu_ids=[gpu_id],
        tracking_mode="machine",
    )

    base_model_path = root_path / "Base_Model"

    # Model and configuration setup
    saved_models_path = root_path / "Models"
    models_dir = saved_models_path / framework / f"scenario_{scenario}"
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

    if framework == "Baseline":
        model_path = base_model_path / model_name
    else:
        if domain_id == 0:
            model_path = base_model_path / model_name
        else:
            model_path = (
                models_dir / f"global_model_domain_{domain_id - 1}_small_aug.pt"
            )

    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    orig_csv_dir = results_dir / "csv"
    if not orig_csv_dir.exists():
        orig_csv_dir.mkdir(parents=True, exist_ok=True)

    orig_client_csv_dir = orig_csv_dir / node_name
    if not orig_client_csv_dir.exists():
        orig_client_csv_dir.mkdir(parents=True, exist_ok=True)

    train_model_name = (
        f"train__{framework}__scenario_{scenario}__domain_{domain_id}__{node_name}"
    )

    csv_path = orig_client_csv_dir / f"eval_current_domain_{domain_id}_small.csv"
    cumulative_csv_path = (
        orig_client_csv_dir / f"eval_cumulative_domain_{domain_id}_small.csv"
    )

    cumulative_data_path = (
        data_dir / "cumulative" / f"domain_{domain_id}_{node_name}.yaml"
    )
    cumulative_data_name = (
        f"client_model_test__{node_name}__cumulative__domain_{domain_id}"
    )

    stats_dir = results_dir / "stats"
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    client = FLClient(
        domain_id=domain_id,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_path=model_path,
        n_train_freezing_layers=n_train_freezing_layers,
        n_comm_freezing_layers=n_comm_freezing_layers,
        train_model_name=train_model_name,
        data_yaml=data_yaml,
        csv_path=csv_path,
        runs_path=root_path / "Results" / "runs",
        print_fn=print_fn,
        cumulative_data_path=cumulative_data_path,
        cumulative_data_name=cumulative_data_name,
        cumulative_csv_path=cumulative_csv_path,
    ).to_client()

    start_time = time.time()

    tracker.start()

    fl.client.start_client(
        server_address=server_address,
        client=client,
    )

    emissions = tracker.stop()
    print_fn(f"CO₂ Emissions: {emissions} kg CO₂")

    end_time = time.time()
    training_time = end_time - start_time

    # Saving time in file txt
    time_file_path = stats_dir / f"time__{node_name}__domain_{domain_id}__small.txt"
    with open(time_file_path, "w") as f:
        f.write(f"Training time: {training_time:.2f} seconds\n")

    emissions_file_path = (
        stats_dir / f"emissions__{node_name}__domain_{domain_id}__small.txt"
    )
    with open(emissions_file_path, "w") as f:
        f.write(f"CO₂ Emissions: {emissions} kg CO₂\n")
