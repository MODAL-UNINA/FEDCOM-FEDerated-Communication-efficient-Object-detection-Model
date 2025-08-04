from pathlib import Path

from app.args import parse_args
from app.startup import setup_environment

args = parse_args(is_server=False)

framework: str = args.framework
model_name: str = args.model_name
num_epochs: int = args.epochs
server_address: str = args.server_address
gpu_id: int = args.gpu_id
similarity_threshold: float = args.similarity_threshold
max_images: int = args.max_images
batch_size: int = args.batch_size
client_id: int = args.client_id
domain_id: int = args.domain_id
scenario: int = args.scenario
max_num_clients: int = args.max_num_clients

root_dir = Path().cwd().parent

setup_environment(
    root_dir=root_dir,
    framework=framework,
    scenario=scenario,
    domain_id=domain_id,
    gpu_id=gpu_id,
    node_name=f"client_{client_id}",
)

if __name__ == "__main__":
    from app.client import run_client

    run_client(
        framework=framework,
        scenario=scenario,
        domain_id=domain_id,
        client_id=client_id,
        gpu_id=gpu_id,
        model_name=model_name,
        server_address=server_address,
        num_epochs=num_epochs,
        batch_size=batch_size,
        similarity_threshold=similarity_threshold,
        max_images=max_images,
        max_num_clients=max_num_clients,
        root_path=root_dir,
    )
