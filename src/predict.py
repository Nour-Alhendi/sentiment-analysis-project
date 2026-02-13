import argparse
from typing import Any
import numpy as np
from numpy.typing import NDArray
from joblib import load


def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    return load(model_path)


# Other functions will go here


def main(model_path: str, input_texts: list[str]) -> None:
    """
    Load the trained model and make predictions on input texts.
    """
    model = load_model(model_path)

    predictions: NDArray[np.int_] = model.predict(input_texts)

    for text, pred in zip(input_texts, predictions):
        label = "Positive" if pred == 1 else "Negative"
        print(f"{label}: {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sentiment.joblib")
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    args = parser.parse_args()
    main(model_path=args.model, input_texts=args.text)