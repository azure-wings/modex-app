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
            "**nsamples**: Maximum number of features present in the explanation",
            min_value=1,
            value=500,
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

        original_img_arr = resize_crop(self.instance.image_array)

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

            return imagenet_preprocess(out)

        def predict_coalition(coalition: np.array) -> torch.Tensor:
            prediction = self.model.predict(
                mask_image(coalition, segments_slic, original_img_arr).double()
            )

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
            )

        # Get the original seismic colormap
        cmap_data = plt.cm.seismic_r(np.arange(plt.cm.seismic_r.N))
        # Swap the blue and green channels
        cmap_data[:, [1, 2]] = cmap_data[:, [2, 1]]
        # Create the modified colormap
        seismic_g = ListedColormap(cmap_data)

        # For colourmap normalisation
        max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])

        exp_label_list = [None] * len(targets)

        for i, target in enumerate(targets):
            m = fill_segmentation(shap_values[target][0], segments_slic)
            background = Image.fromarray(
                (
                    mark_boundaries(original_img_arr / 255, segments_slic, color=(1, 1, 1)) * 255
                ).astype(np.uint8)
            ).convert("RGBA")
            shap_colormap = seismic_g(plt.Normalize(vmin=-max_val, vmax=max_val)(m)) * 255

            # Set white colormaps transparent, and vivid colormaps opaque
            avg_brightness = np.sum(shap_colormap[:, :, :3], axis=2) / 3
            # Modify the alpha channel based on the brightness of each pixel
            high_brightness_all_channels = np.all(shap_colormap[:, :, :3] > 250, axis=2)
            shap_colormap[high_brightness_all_channels, 3] = 0

            high_brightness_some_channels = (
                np.any(shap_colormap[:, :, :3] > 127, axis=2) & ~high_brightness_all_channels
            )
            shap_colormap[high_brightness_some_channels, 3] = (
                0.5 + 0.5 * np.tanh(30 - 31 * avg_brightness[high_brightness_some_channels] / 255)
            ) * 255
            shap_colormap[:, :, 3] = np.clip(shap_colormap[:, :, 3], 0, 200)

            shap_colormap = Image.fromarray(shap_colormap.astype(np.uint8)).convert("RGBA")
            background.paste(shap_colormap, (0, 0), shap_colormap)

            exp_label_list[i] = (background, target)

        return exp_label_list
