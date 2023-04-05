import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-date", type=str, dest="training_date")
    parser.add_argument("--encoders-s3-path", type=str, dest="encoders_s3_path")
    parser.add_argument("--model-location", type=str, dest="model_location")

    args = parser.parse_args()

    config = {
        "training_date": args.training_date,
        "encoders_s3_path": args.encoders_s3_path,
        "model_location": args.model_location,
    }

    out_file = open("/opt/ml/processing/config/train_config.json", "w")
    json.dump(config, out_file, indent=6)
    out_file.close()
