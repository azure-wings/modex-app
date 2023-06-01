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


class PartitionSHAPImageExplainer(ImageExplainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)
        logging.getLogger("shap").disabled = True
        self.explainer_options = dict()

    def set_explainer_options(
        self,
    ) -> Dict[OptionKey, OptionValue]:
        options, explainer_options = self.options, self.explainer_options

        masker_names = {
            0: {
                "display_name": "Inpaint (Navier-Stokes)",
                "internal_name": "inpaint_ns",
            },
            1: {
                "display_name": "Inpaint (Telea)",
                "internal_name": "inpaint_telea",
            },
            2: {
                "display_name": "Box Blur",
                "internal_name": "blur",
            },
        }
        masker_type = st.radio(
            "**masker_type**: Masking options to use when generating coalitions.",
            options=[0, 1, 2],
            index=0,
            format_func=lambda x: masker_names[x]["display_name"],
            horizontal=True,
        )
        if masker_type == 2:
            kernel_size = st.number_input(
                "**kernel_size**: Size of the box blur kernel. (Uses square kernel)",
                min_value=1,
                max_value=224,
                step=1,
            )
            options[
                "masker"
            ] = f"{masker_names[masker_type]['internal_name']}({kernel_size},{kernel_size})"
        else:
            options["masker"] = masker_names[masker_type]["internal_name"]

        explainer_options["max_evals"] = st.number_input(
            "**max_evals**: Number of times to re-evaluate the model when explaining each prediction.",
            min_value=1,
            value=1000,
            step=1,
        )

        explainer_options["batch_size"] = st.number_input(
            "**batch_size**: Number of masked images to be evaluated at once.",
            min_value=1,
            value=64,
            step=1,
        )

        fixed_context = st.text_input(
            "**fixed_context**: (Optional) Masking technqiue used to build partition tree with options of '0', '1' or 'None'.",
            value="None",
            help="'None' is the best option to generate meaningful results, but it is relatively slower than '0' or '1' because it generates a full partition tree.",
        )
        if fixed_context not in {"0", "1", "None"}:
            raise ValueError(f"Unknown option for fixed_context: {fixed_context}")
        explainer_options["fixed_context"] = fixed_context if fixed_context != "None" else None

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

        def _predict(arr: np.array) -> np.array:
            arr = arr.copy()
            preprocessed = imagenet_preprocess(arr)
            prediction = self.model.predict(preprocessed)

            return prediction.cpu().detach().numpy()

        masker = shap.maskers.Image(self.options["masker"], original_img_arr.shape)

        explainer = shap.Explainer(_predict, masker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanation = explainer(original_img_arr[np.newaxis, ...], **self.explainer_options)

        # List unpacking. Original explanation.values is a list of length 1
        explanation.values = explanation.values[0]
        # Move axis: (224, 224, 3, 1000) -> (1000, 224, 224, 3)
        explanation.values = np.moveaxis(explanation.values, -1, 0)
        shap_values = [np.sum(explanation.values[target], axis=-1) for target in targets]

        # Get the original seismic colormap
        cmap_data = plt.cm.seismic_r(np.arange(plt.cm.seismic_r.N))
        # Swap the blue and green channels
        cmap_data[:, [1, 2]] = cmap_data[:, [2, 1]]
        # Create the modified colormap
        seismic_g = ListedColormap(cmap_data)

        # For colourmap normalisation
        max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])
        sigma = np.std(shap_values)

        exp_label_list = [None] * len(targets)

        for i, target in enumerate(targets):
            background = Image.fromarray(original_img_arr.astype(np.uint8)).convert("RGBA")

            shap_colormap = (
                seismic_g(plt.Normalize(vmin=-max_val, vmax=max_val)(shap_values[i])) * 255
            )
            adjust_alpha = lambda x: np.clip(255 * (1 - 0.8 * np.exp(-((x / sigma) ** 2))), 0, 200)
            shap_colormap[:, :, 3] = adjust_alpha(shap_values[i])

            shap_colormap = Image.fromarray(shap_colormap.astype(np.uint8)).convert("RGBA")
            background.paste(shap_colormap, (0, 0), shap_colormap)

            exp_label_list[i] = (background, target)

        return exp_label_list
