from abc import ABC, abstractmethod
from typing import Type, Any
from dataclasses import dataclass
import inspect

import pandas as pd
import torch
import torchvision.transforms as T
import PIL
from PIL.Image import Image


class Instance(ABC):
    def __init__(self, data: Any, target: int) -> None:
        self.data = data
        self.target = target
        self.preprocessed = None

    @abstractmethod
    def preview(self) -> Any:
        pass

    @abstractmethod
    def preprocess(self) -> Any:
        pass


class ImageInstance(Instance):
    def __init__(self, data: Any, target: int):
        super().__init__(data, target)
        self.preprocessed = self.preprocess()

    def preview(self) -> Image:
        preview_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
            ]
        )
        return preview_transform(PIL.Image.open(self.data))

    def preprocess(self) -> torch.Tensor:
        preprocess_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.preprocessed = preprocess_transform(PIL.Image.open(self.data)).unsqueeze(0)
        return self.preprocessed


class TextInstance(Instance):
    def __init__(self, data: Any, target: int):
        super().__init__(data, target)


class TabularInstance(Instance):
    def __init__(self, data: Any, target: int):
        super().__init__(data, target)


def create_instance(instance_type: str) -> Instance:
    instance_map = {
        "Image": ImageInstance,
        "Text": TextInstance,
        "Tabular": TabularInstance,
    }
    instance_class = instance_map.get(instance_type)
    if instance_class:
        return instance_class
    else:
        raise ValueError(f"Unsupported instance type: {instance_type}")


def get_instance_type(instance: Instance) -> str:
    match instance.__class__.__name__:
        case "ImageInstance":
            return "Image"
        case "TextInstance":
            return "Text"
        case "TabularInstance":
            return "Tabular"
        case _:
            raise ValueError(f"Unsupported instance type: {type(instance)}")