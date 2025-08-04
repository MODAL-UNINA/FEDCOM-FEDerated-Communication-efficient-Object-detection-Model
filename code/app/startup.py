import os
from pathlib import Path


def setup_environment(
    root_dir: Path,
    framework: str,
    scenario: int,
    domain_id: int,
    gpu_id: int,
    node_name: str,
):
    ultralytics_path = (
        root_dir
        / "Results"
        / "Ultralytics"
        / framework
        / f"scenario_{scenario}"
        / f"domain_{domain_id}"
        / node_name
    )

    config_path = ultralytics_path / "settings"

    os.environ["YOLO_CONFIG_DIR"] = str(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # from ultralytics.utils import SettingsManager

    # manager = SettingsManager()
    # manager["runs_dir"] = str(ultralytics_path / "runs")

    # del manager
    # del SettingsManager
