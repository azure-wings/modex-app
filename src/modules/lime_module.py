from typing import Tuple, Dict, List, TypeAlias

import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import streamlit as st
from skimage.segmentation import mark_boundaries

from lime.explanation import Explanation
from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from explainer import ImageExplainer, TextExplainer, TabularExplainer
from model import Model
from instance import Instance

OptionKey: TypeAlias = str
OptionValue: TypeAlias = int | str | bool


class LIMEImageExplainer(ImageExplainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)
        self.display_options = {}

    def set_options(
        self,
    ) -> Tuple[Dict[OptionKey, OptionValue], Dict[OptionKey, OptionValue]]:
        options: Dict[OptionKey, OptionValue] = {}
        display_options: Dict[OptionKey, OptionValue] = {}
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

        options["num_features"] = st.number_input(
            "Maximum number of features present in the explanation",
            min_value=1,
            value=100000,
            step=1,
        )
        options["num_samples"] = st.number_input(
            "Size of the neighbourhood to learn the linear model",
            min_value=1,
            value=1000,
            step=1,
        )
        options["batch_size"] = st.number_input(
            "Batch size for perturbed prediction",
            min_value=1,
            value=64,
            step=1,
        )

        superpixels = st.selectbox(
            "Which superpixels to take depending on how they contribute to the prediction of the label",
            ["Positive only", "Negative only", "Both"],
            index=2,
        )

        display_options["positive_only"] = True if superpixels == "Positive only" else False

        display_options["negative_only"] = True if superpixels == "Negative only" else False

        display_options["hide_rest"] = (
            True
            if st.radio(
                "Whether to make the non-explanation part of the return image gray",
                ["True", "False"],
                index=1,
                horizontal=True,
            )
            == "True"
            else False
        )

        display_options["num_features"] = st.number_input(
            "Number of superpixels to include in explanation",
            min_value=1,
            value=5,
            step=1,
        )

        self.options = options
        self.display_options = display_options
        return options, display_options

    def explain(self) -> List[Tuple[Image.Image, int]]:
        def batcher(x: np.array) -> torch.Tensor:
            return torch.stack(
                tuple(
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                        T.ToTensor()(i)
                    )
                    # self.instance.preprocess(i)
                    for i in x
                ),
                dim=0,
            )

        def predictor(x: np.array) -> torch.Tensor:
            return self.model.predict(batcher(x).to(self.model.device))

        explanation = LimeImageExplainer().explain_instance(
            np.array(self.instance.preview()),
            lambda x: predictor(x).detach().cpu().numpy()
            if torch.sum(predictor(x)) == 1
            else F.softmax(predictor(x)).detach().cpu().numpy(),
            **self.options
        )

        labels = (
            explanation.top_labels if hasattr(explanation, "top_labels") else self.options["labels"]
        )

        image_mask_list = [
            explanation.get_image_and_mask(label, **self.display_options) for label in labels
        ]

        return [
            (
                Image.fromarray((mark_boundaries(image / 255.0, mask) * 255).astype(np.uint8)),
                labels[i],
            )
            for i, (image, mask) in enumerate(image_mask_list)
        ]


class LIMETextExplainer(TextExplainer):
    pass


class LIMETabularExplainer(TabularExplainer):
    pass
