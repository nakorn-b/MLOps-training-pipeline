import argparse

from pipeline.flow import retrain, train


def main():
    parser = argparse.ArgumentParser(description="Run the train or retrain pipeline.")
    parser.add_argument("--config", required=True, help="path to the experiment YAML")
    parser.add_argument("--retrain", action="store_true", help="fine-tune the current champion")
    parser.add_argument("--dataset-uri", help="retrain only: override data.uri from the YAML")
    args = parser.parse_args()

    if args.dataset_uri and not args.retrain:
        parser.error("--dataset-uri requires --retrain")

    if args.retrain:
        retrain(args.config, args.dataset_uri)
    else:
        train(args.config)


if __name__ == "__main__":
    main()
