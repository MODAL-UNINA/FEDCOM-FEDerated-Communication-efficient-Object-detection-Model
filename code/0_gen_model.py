import argparse
from pathlib import Path

import yaml

from app.exec_utils import is_interactive
from app.startup import setup_environment
from app.validations import validate_model

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--model-config",
    type=str,
    default="yolo12s.yaml",
    help="Starting model configuration file",
)
parser.add_argument(
    "--output-model-config",
    type=str,
    default="yolo12s_upd.yaml",
    help="Model configuration file to generate",
)
parser.add_argument("--gpu-id", type=int, default=0, help="GPU number to use")

if is_interactive():
    args, _ = parser.parse_known_args()
else:
    args = parser.parse_args()

root_dir = Path().cwd().parent

setup_environment(root_dir, "Baseline", 1, 0, args.gpu_id, "server")


def main():
    from ultralytics.nn.tasks import yaml_model_load

    base_model_path = Path("Base_Model")

    # This file may not exist, as it can be obtained from Ultralytics
    model_config = validate_model(
        args.model_config, base_model_path=base_model_path, not_exists_ok=True
    )

    # This file may not exist, so we allow it
    output_model_config = validate_model(
        args.output_model_config, base_model_path=base_model_path, not_exists_ok=True
    )

    # Model and configuration setup

    output_model_path = base_model_path / output_model_config

    if Path(output_model_path).exists():
        raise FileExistsError(
            f"Model {output_model_path} already exists. "
            f"Please remove if you need to regenerate it."
        )

    config = yaml_model_load(model_config)
    config["nc"] = 3
    del config["yaml_file"]

    with open(output_model_path, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    main()
