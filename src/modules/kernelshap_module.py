from typing import Tuple, Dict, List, TypeAlias

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torchvision.transforms as T
import streamlit as st
from skimage.segmentation import slic, mark_boundaries

import shap

import sys, os
import logging
import warnings


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from explainer import ImageExplainer, TextExplainer, TabularExplainer
from model import Model
from instance import Instance
from utils.image import resize_crop, imagenet_preprocess

OptionKey: TypeAlias = str
OptionValue: TypeAlias = int | str | bool


class KernelSHAPImageExplainer(ImageExplainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)
        logging.getLogger("shap").disabled = True

    def set_explainer_options(
        self,
    ) -> Dict[OptionKey, OptionValue]:
        options = self.options

        options["nsamples"] = st.number_input(
            "**nsamples**: Number of times to re-evaluate the model when explaining each prediction.",
            min_value=1,
            value=2048,
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

        original_img_arr = self.instance.preview()

        # https://h1ros.github.io/posts/explain-the-prediction-for-imagenet-using-shap/
        def segment_image(img_arr: np.array) -> np.array:
            segments_slic = slic(
                img_arr,
                n_segments=self.options["n_segments"],
                compactness=self.options["compactness"],
                sigma=self.options["sigma"],
            )
            return segments_slic

        segments_slic = segment_image(original_img_arr)

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

            return out

        def predict_coalition(coalition: np.array) -> torch.Tensor:
            masked = mask_image(coalition, segments_slic, original_img_arr)
            masked_preprocessed = imagenet_preprocess(masked)
            prediction = self.model.predict(masked_preprocessed.double())

            return prediction.cpu().detach().numpy()

        def fill_segmentation(shap_values: np.array, segmentation: np.array) -> np.array:
            out = np.zeros(segmentation.shape)
            for i in range(len(shap_values)):
                out[segmentation == i] = shap_values[i]
            return out

        explainer = shap.KernelExplainer(
            predict_coalition, np.zeros((1, self.options["n_segments"]))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values: np.array = explainer.shap_values(
                np.ones((1, self.options["n_segments"])), nsamples=self.options["nsamples"]
            )  # Shape:

        # Get the original seismic colormap
        cmap_data = plt.cm.seismic_r(np.arange(plt.cm.seismic_r.N))
        # Swap the blue and green channels
        cmap_data[:, [1, 2]] = cmap_data[:, [2, 1]]
        # Create the modified colormap
        seismic_g = ListedColormap(cmap_data)

        # For colourmap normalisation
        max_val = np.max([np.max(np.abs(shap_values[i])) for i in range(len(shap_values))])
        sigma = np.std(shap_values)

        exp_label_list = [None] * len(targets)

        for i, target in enumerate(targets):
            m = fill_segmentation(shap_values[target][0], segments_slic)
            background = Image.fromarray(
                (
                    mark_boundaries(original_img_arr / 255, segments_slic, color=(1, 1, 1)) * 255
                ).astype(np.uint8)
            ).convert("RGBA")

            shap_colormap = seismic_g(plt.Normalize(vmin=-max_val, vmax=max_val)(m)) * 255
            adjust_alpha = lambda x: np.clip(255 * (1 - 0.8 * np.exp(-((x / sigma) ** 2))), 0, 200)
            shap_colormap[:, :, 3] = adjust_alpha(m)

            shap_colormap = Image.fromarray(shap_colormap.astype(np.uint8)).convert("RGBA")
            background.paste(shap_colormap, (0, 0), shap_colormap)

            exp_label_list[i] = (background, target)

        return exp_label_list
