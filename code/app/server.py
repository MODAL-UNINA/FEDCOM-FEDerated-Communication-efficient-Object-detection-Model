import time
from pathlib import Path

import flwr as fl
from codecarbon import EmissionsTracker
from flwr.common import ndarrays_to_parameters
from ultralytics import YOLO

from app.fl_server import FLServer
from app.utils import get_parameters, get_n_freezing_layers
from app.validations import (
    validate_domain_id,
    validate_framework,
    validate_gpu_id,
    validate_max_num_clients,
    validate_model,
    validate_rounds,
    validate_scenario,
    validate_server_address,
)


def run_server(
    framework: str,
    scenario: int,
    domain_id: int,
    gpu_id: int,
    model_name: str,
    server_address: str,
    max_num_clients: int,
    rounds_per_domain: int,
    root_path: Path,
):
    base_model_path = root_path / "Base_Model"

    framework = validate_framework(framework)
    model_name = validate_model(model_name, base_model_path=base_model_path)
    scenario = validate_scenario(scenario)
    domain_id = validate_domain_id(domain_id)
    server_address = validate_server_address(server_address)
    gpu_id = validate_gpu_id(gpu_id)
    max_num_clients = validate_max_num_clients(max_num_clients)
    rounds_per_domain = validate_rounds(rounds_per_domain)

    n_comm_freezing_layers = (
        get_n_freezing_layers(domain_id) if framework == "FEDCOM" else 0
    )

    node_name = "server"
    node_name_desc = "Server"

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
    print_fn(f"- Rounds per domain: {rounds_per_domain}")
    print_fn(f"- Server address: {server_address}")
    print_fn(f"- GPU ID: {gpu_id}")

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

    def fit_config(server_round: int):
        """Generate training configuration for each round."""
        # Create the configuration dictionary
        config = {
            "current_round": server_round,
            "total_rounds": rounds_per_domain,
        }
        return config

    def evaluate_global(
        server_round: int,
        parameters,
        config: dict,
    ) -> None:
        """Evaluate the model using the server's test dataset."""
        from app.utils import (
            get_state_dict_all_parameters,
            get_state_dict_trainable_parameters,
        )

        print_fn(f"Server round: {server_round}")
        if server_round == 0:
            print_fn(
                "This is the initialization round of training. Skipping evaluation."
            )
            return None

        if server_round == rounds_per_domain:
            print_fn(f"This is the last round of training: {server_round}")
            print_fn("Skipping evaluation in final round")
            return None

        if domain_id == 0:
            # Load the model
            model = YOLO(model_path)
            print_fn("Using initial model configuration for domain 0")
        else:
            # model_path = "client_model_after_training.pt"
            runs_path = root_path / "Results" / "runs"

            prev_model_paths = [
                runs_path
                / f"train__{framework}__scenario_{scenario}__domain_{domain_id}__client_{client_id}"
                / "weights"
                / "last.pt"
                for client_id in range(max_num_clients)
            ]
            prev_model_path = [f for f in prev_model_paths if f.exists()][0]
            if not prev_model_path.exists():
                raise FileNotFoundError(f"Model file {prev_model_path} does not exist.")
            model = YOLO(prev_model_path)
            print_fn("Using updated global model from previous domain")

        # Set the parameters saved in the previous domain
        state_dict = (
            get_state_dict_trainable_parameters(
                parameters, model, n_comm_freezing_layers
            )
            if n_comm_freezing_layers > 0
            else get_state_dict_all_parameters(parameters, model)
        )
        model.eval()
        model.load_state_dict(state_dict, strict=True)

        model.save(f"{models_dir}/global_model_domain_{domain_id}_small_aug.pt")

    init_params = None

    print_fn(f"\n{'=' * 50}")
    print_fn(f"Starting training for DOMAIN {domain_id}")
    print_fn(f"{'=' * 50}")

    if framework != "Baseline" and domain_id > 0:
        assert model_path.name.endswith(".pt"), "Model file is not a .pt file."
        prev_model = YOLO(model_path)
        prev_model.eval()
        init_params = ndarrays_to_parameters(get_parameters(prev_model))
        print_fn(f"Loaded init weights from {model_path}")

    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3 if domain_id == 0 else 2,
        min_available_clients=3 if domain_id == 0 else 2,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_global,
        initial_parameters=init_params,
    )

    stats_dir = results_dir / "stats"
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    communication_file_path = stats_dir / f"communication_time_domain_{domain_id}.txt"
    fit_round_file_path = stats_dir / f"fit_round_time_domain_{domain_id}.txt"

    communication_file_path.parent.mkdir(parents=True, exist_ok=True)
    if communication_file_path.exists():
        with open(communication_file_path, "w") as f:
            f.write("")

    fit_round_file_path.parent.mkdir(parents=True, exist_ok=True)
    if fit_round_file_path.exists():
        with open(fit_round_file_path, "w") as f:
            f.write("")

    custom_server = FLServer(
        client_manager=client_manager,
        strategy=strategy,
        communication_file_path=communication_file_path,
        fit_round_file_path=fit_round_file_path,
        print_fn=print_fn,
    )

    start_time = time.time()

    tracker.start()

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds_per_domain),
        server=custom_server,
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

    # Average communication time calculation
    def compute_avg_comm_time(path, out_file):
        times = []
        with open(path, "r") as f:
            for line in f:
                if "Communication time for round" in line:
                    try:
                        time_str = line.strip().split(":")[-1].strip().split()[0]
                        times.append(float(time_str))
                    except (IndexError, ValueError):
                        continue

        if times:
            avg = sum(times) / len(times)
            print_fn(
                f"Average communication time over {len(times)} rounds: {avg:.6f} seconds"
            )
            with open(out_file, "w") as f:
                f.write(
                    f"Average communication time over {len(times)} rounds: {avg:.6f} seconds\n"
                )
        else:
            print_fn("No valid communication times found.")

    compute_avg_comm_time(
        communication_file_path,
        stats_dir / f"avg_communication_time_domain_{domain_id}.txt",
    )

    def parse_comm_times(filepath):
        import re

        pattern = r"Communication time for round (\d+): ([\d.]+) seconds"
        times = {}
        with open(filepath, "r") as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    round_num = int(match.group(1))
                    time_val = float(match.group(2))
                    times[round_num] = time_val
        return times

    def parse_fit_times(filepath):
        import re

        pattern = r"Fit round (\d+) completed in ([\d.]+) seconds"
        times = {}
        with open(filepath, "r") as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    round_num = int(match.group(1))
                    time_val = float(match.group(2))
                    times[round_num] = time_val
        return times

    def compute_comm_ratios(comm_file, fit_file, output_file):
        comm_times = parse_comm_times(comm_file)
        fit_times = parse_fit_times(fit_file)

        lines = []
        header = "Round | Communication (s) | Fit Total (s) | Ratio (%)"
        lines.append(header)
        lines.append("-" * len(header))

        ratios = []
        comms = []
        fits = []
        for rnd in sorted(fit_times.keys()):
            comm = comm_times.get(rnd)
            fit = fit_times.get(rnd)
            if comm is not None and fit > 0:
                ratio = comm / fit
                ratios.append(ratio)
                lines.append(f"{rnd:5d} | {comm:17.6f} | {fit:13.6f} | {ratio:6f}")
                comms.append(comm)
                fits.append(fit)
            else:
                lines.append(f"{rnd:5d} |     MISSING DATA")

        if ratios:
            # avg_ratio = sum(ratios) / len(ratios)
            avg_ratio = sum(comms) / sum(fits)
            lines.append(
                f"\nAverage communication ratio across rounds: {avg_ratio:.6f}"
            )

        for line in lines:
            print_fn(line)

        with open(output_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    compute_comm_ratios(
        communication_file_path,
        fit_round_file_path,
        stats_dir / f"communication_ratios_domain_{domain_id}.txt",
    )
