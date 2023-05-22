from abc import ABC, abstractmethod
from typing import Type, Any
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
import PIL
from PIL.Image import Image

from utils.image import imagenet_preprocess


class Instance(ABC):
    def __init__(self, data: Any, target: int) -> None:
        self.data = data
        self.target = target

    @abstractmethod
    def preprocess(self) -> Any:
        pass


class ImageInstance(Instance):
    def __init__(self, data: Any, target: int):
        super().__init__(data, target)
        self.image_array = self.load_as_nparray()

    def load_as_nparray(self) -> Image:
        img_arr = np.array(PIL.Image.open(self.data))
        return img_arr

    def preprocess(self) -> torch.Tensor:
        return imagenet_preprocess(self.image_array)


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
