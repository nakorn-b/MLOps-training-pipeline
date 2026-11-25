import argparse

from pipeline.flow import retrain, train


def main():
    parser = argparse.ArgumentParser(description="Run the train or retrain pipeline.")
    parser.add_argument("--config", required=True, help="path to the experiment YAML")
    parser.add_argument("--retrain", action="store_true", help="fine-tune an existing model")
    parser.add_argument("--model-uri", help="retrain only: MLflow model to fine-tune; overrides model.uri from the YAML")
    parser.add_argument("--dataset-uri", help="retrain only: override data.uri from the YAML")
    args = parser.parse_args()

    # Error if retrain and not provide model_uri or dataset_uri
    if args.model_uri and not args.retrain:
        parser.error("--model-uri requires --retrain")
    if args.dataset_uri and not args.retrain:
        parser.error("--dataset-uri requires --retrain")

    if args.retrain:
        retrain(args.config, model_uri=args.model_uri, dataset_uri=args.dataset_uri)
    else:
        train(args.config)


if __name__ == "__main__":
    main()
