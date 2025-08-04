import argparse
from .exec_utils import is_interactive


def parse_args(is_server: bool = True):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--framework", type=str, default="FEDCOM", help="Use only trainable layers"
    )
    parser.add_argument(
        "--model-name", type=str, default="yolo12s_upd.yaml", help="Model name to use"
    )
    parser.add_argument("--scenario", type=int, default=3, help="Current scenario")
    parser.add_argument("--domain-id", type=int, default=1, help="Current domain ID")
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8081",
        help="Address of the server",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU number to use")
    parser.add_argument(
        "--max-num-clients",
        type=int,
        default=3,
        help="Maximum expected number of clients",
    )
    if is_server:
        parser.add_argument(
            "--rounds-per-domain",
            type=int,
            default=51,
            help="Number of rounds per domain",
        )
    else:
        parser.add_argument("--client-id", type=int, default=0, help="Client ID")
        parser.add_argument(
            "--similarity-threshold",
            type=float,
            default=0.75,
            help="Threshold for similarity shutdown",
        )
        parser.add_argument(
            "--max-images", type=int, default=20, help="Max images for similarity check"
        )
        parser.add_argument("--epochs", type=int, default=5, help="Epochs per round")
        parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    if is_interactive():
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args
