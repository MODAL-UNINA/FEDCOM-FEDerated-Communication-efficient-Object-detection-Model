from pathlib import Path

from app.args import parse_args
from app.startup import setup_environment

args = parse_args(is_server=True)

framework: str = args.framework
model_name: str = args.model_name
scenario: int = args.scenario
domain_id: int = args.domain_id
server_address: str = args.server_address
gpu_id: int = args.gpu_id
max_num_clients: int = args.max_num_clients
rounds_per_domain: int = args.rounds_per_domain

root_dir = Path().cwd().parent

setup_environment(
    root_dir=root_dir,
    framework=framework,
    scenario=scenario,
    domain_id=domain_id,
    gpu_id=gpu_id,
    node_name="server",
)

if __name__ == "__main__":
    from app.server import run_server

    run_server(
        framework=framework,
        scenario=scenario,
        domain_id=domain_id,
        gpu_id=gpu_id,
        model_name=model_name,
        server_address=server_address,
        max_num_clients=max_num_clients,
        rounds_per_domain=rounds_per_domain,
        root_path=root_dir,
    )
