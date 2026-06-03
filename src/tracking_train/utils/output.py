import os
import shutil
from datetime import datetime


def unique_output_dir(config):
    configured_run_dir = config.get("output", {}).get("run_dir")
    if configured_run_dir:
        os.makedirs(configured_run_dir, exist_ok=True)
        return configured_run_dir

    # Generate a unique directory name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(config["output"]["base_path"], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def copy_config_to_output(config_path, output_dir):
    """
    Copies the configuration file to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, os.path.basename(config_path))
    if os.path.abspath(config_path) == os.path.abspath(destination):
        return
    shutil.copy(config_path, output_dir)
