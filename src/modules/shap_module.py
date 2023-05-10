from typing import Tuple, Dict, List, TypeAlias

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import streamlit as st
from skimage.segmentation import slic, mark_boundaries

import shap

import sys, os
import io

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from explainer import ImageExplainer, TextExplainer, TabularExplainer
from model import Model
from instance import Instance

OptionKey: TypeAlias = str
OptionValue: TypeAlias = int | str | bool


class SHAPImageExplainer(ImageExplainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)

    def set_options(
        self,
    ) -> Dict[OptionKey, OptionValue]:
        options: Dict[OptionKey, OptionValue] = {}

        label_generation = st.radio(
            "Choose how the labels for explanation would be generated",
            ["Automatic pseudolabel generation", "Manual label designation"],
            horizontal=True,
        )
        if label_generation == "Manual label designation":
            options["labels"] = [
                int(i)
                for i in st.text_input(
                    "Labels you wish to be explained",
                    help="Split each label with a comma",
                ).split(",")
            ]
            options["top_labels"] = None
        else:
            options["top_labels"] = st.number_input(
                "Number of labels (with the highest prediction probabilities) to produce explanations for",
                min_value=1,
                max_value=1000,
                value=5,
                step=1,
            )

        options["nsamples"] = st.number_input(
            "**nsamples**: Maximum number of features present in the explanation",
            min_value=1,
            value=1000,
            step=1,
        )

        options["n_segments"] = st.number_input(
            "**n_segments**: The (approximate) number of labels in the segmented image.",
            value=50,
            step=1,
        )

        options["compactness"] = st.number_input(
            "**compactness**: Balances color proximity and space proximity.",
            help="Higher values give more weight to space proximity, making superpixel shapes more square/cubic.",
            value=10.0,
        )

        options["sigma"] = st.number_input(
            "**sigma**: Width of Gaussian smoothing kernel for pre-processing for each dimension of the image.",
            value=0,
        )

        self.options = options
        return options

    def explain(self) -> List[Tuple[Image.Image, int]]:
        if self.options["top_labels"]:
            _, targets = torch.topk(
                self.model.predict(self.instance.preprocess()), k=self.options["top_labels"]
            )
            targets = targets.tolist()[0]
        else:
            targets = self.options["labels"]

        original_img_arr = np.array(self.instance.preview())

        # https://h1ros.github.io/posts/explain-the-prediction-for-imagenet-using-shap/
        def segment_image() -> np.array:
            segments_slic = slic(
                original_img_arr,
                n_segments=self.options["n_segments"],
                compactness=self.options["compactness"],
                sigma=self.options["sigma"],
            )
            return segments_slic

        segments_slic = segment_image()

        def batcher(x: np.array) -> torch.Tensor:
            return torch.stack(
                tuple(
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                        T.ToTensor()(i)
                    )
                    for i in x
                ),
                dim=0,
            )

        def mask_image(
            coalition: np.array, segmentation: np.array, image: np.array, background=None
        ) -> torch.Tensor:
            if background is None:
                background = image.mean((0, 1))

            # Create an empty 4D array
            out = np.zeros((coalition.shape[0], image.shape[0], image.shape[1], image.shape[2]))

            for i in range(coalition.shape[0]):
                out[i, :, :, :] = image
                for j in range(coalition.shape[1]):
                    if coalition[i, j] == 0:
                        out[i][segmentation == j, :] = background

            return batcher(out)

        def predict_coalition(coalition: np.array) -> torch.Tensor:
            return (
                self.model.predict(mask_image(coalition, segments_slic, original_img_arr).double())
                .cpu()
                .detach()
                .numpy()
            )

        def fill_segmentation(shap_values: np.array, segmentation: np.array) -> np.array:
            out = np.zeros(segmentation.shape)
            for i in range(len(shap_values)):
                out[segmentation == i] = shap_values[i]
            return out

        explainer = shap.KernelExplainer(predict_coalition, np.zeros((1, 50)))
        shap_values: np.array = explainer.shap_values(
            np.ones((1, 50)), nsamples=self.options["nsamples"]
        )

        colors = []
        for l in np.linspace(1.0, 0.1, 256):
            colors.append((245 / 255, 39 / 255, 87 / 255, l))
        for l in np.linspace(0.1, 1.0, 256):
            colors.append((24 / 255, 196 / 255, 93 / 255, l))
        my_cmap = LinearSegmentedColormap.from_list("shap", colors)

        max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])

        exp_label_list = [None] * len(targets)

        for i, target in enumerate(targets):
            buf = io.BytesIO()
            m = fill_segmentation(shap_values[target][0], segments_slic)
            plt.axis("off")
            plt.imshow(mark_boundaries(original_img_arr, segments_slic))
            plt.imshow(m, cmap=my_cmap, vmin=-max_val, vmax=max_val)
            plt.savefig(buf, format="png", transparent=True, bbox_inches="tight")
            plt.clf()
            exp_label_list[i] = (Image.open(buf), target)

        return exp_label_list
