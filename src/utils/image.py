import requests

import torch
import torchvision.transforms as T
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class_names = None


def get_imagenet_classname(index: int) -> str:
    """Retrieves the ImageNet class name for a given label index.

    Parameters
    ----------
    index: int
        The label index for the ImageNet class (0 ~ 999).

    Returns
    -------
    str
        The ImageNet class name corresponding to the given index.

    Raises
    ------
    IndexError
        If the index is out of range.
    """
    global class_names

    if class_names is None:
        response = requests.get(
            "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        )
        class_names = response.json()
        class_names = {
            int(key): " ".join(value[1].split("_")).title() for key, value in class_names.items()
        }

    try:
        class_name = class_names[index]
        return f"{class_name} ({index})"
    except IndexError:
        raise IndexError("Invalid index. Index must be between 0 and 999.")


def resize_crop(image: np.array) -> np.array:
    """Resizes and crops the input image.

    Parameters
    ----------
    image : np.array
        Input image as a numpy array of shape (H, W, C).

    Returns
    -------
    np.array
        Resized and cropped image as a numpy array of shape (224, 224, C).
    """
    resize_crop = A.Compose(
        [
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
        ]
    )
    return resize_crop(image=image)["image"]


def imagenet_preprocess(image: np.array) -> torch.Tensor:
    """Preprocesses the input image according to the ImageNet standards.

    Parameters
    ----------
    image : np.array
        Input image of shape (H, W, C), or batch of images of shape (B, H, W, C).

    Returns:
    torch.Tensor
        Preprocessed image as a torch.Tensor of shape (B, C, H, W).
    """
    resize_crop_if_necessary = lambda x: x if x.shape[-2] == x.shape[-3] == 224 else resize_crop(x)
    preprocess_transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    image = resize_crop_if_necessary(image)

    if image.ndim == 3:
        return preprocess_transform(image=image)["image"].unsqueeze(0)
    else:
        images_list = np.split(image, image.shape[0], axis=0)
        preprocessed_images_list = [
            preprocess_transform(image=image.squeeze())["image"] for image in images_list
        ]
        return torch.stack(preprocessed_images_list, dim=0)
