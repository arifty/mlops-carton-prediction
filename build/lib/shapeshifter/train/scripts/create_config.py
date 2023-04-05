import argparse
import json

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file-s3-path", type=str, dest="config_file")
    parser.add_argument("--training-date", type=str, dest="training_date")
    parser.add_argument("--encoders-s3-path", type=str, dest="encoders_s3_path")
    parser.add_argument("--model-location", type=str, dest="model_location")
    parser.add_argument(
        "--baseline-data-constraints", type=str, dest="baseline_data_constraints"
    )
    parser.add_argument(
        "--baseline-data-statistics", type=str, dest="baseline_data_statistics"
    )

    args = parser.parse_args()

    config_file = args.config_file.split("/")
    file_name = config_file[-1]

    config_data = {}

    config = {
        "training_date": args.training_date,
        "encoders_s3_path": args.encoders_s3_path,
        "model_location": args.model_location,
        "baseline_data_constraints": args.baseline_data_constraints,
        "baseline_data_statistics": args.baseline_data_statistics,
    }

    for key, value in config.items():
        config_data[key] = value

    with open(f"{base_dir}/{config_file[-2]}/{file_name}", "w") as f:
        json.dump(config_data, f, indent=6)
