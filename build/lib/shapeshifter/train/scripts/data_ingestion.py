import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
)

if __name__ == "__main__":
    base_dir = "opt.ml.processing"

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            f"{base_dir}.main.run_pipeline",
            "--run_type",
            "train",
            "--flow_type",
            "data_pipeline",
            "--start_date",
            None,
            "--end_date",
            None,
        ]
    )
