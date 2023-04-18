from typing import Type, Any
import yaml
import torch


def get_classification_label(pred: int) -> str:
    with open("./demo/images/imagenet_labels.yaml", "r") as file:
        labels = yaml.safe_load(file)
    return f"{labels[pred].title()} ({pred})"
