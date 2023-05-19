from typing import Dict, List, TypeAlias

from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import streamlit as st

from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients, Occlusion
from zennit.composites import (
    EpsilonPlus,
    EpsilonPlusFlat,
    EpsilonGammaBox,
    EpsilonAlpha2Beta1,
    EpsilonAlpha2Beta1Flat,
)
from zennit.image import imgify

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from explainer import ImageExplainer
from model import Model
from instance import Instance

OptionKey: TypeAlias = str
OptionValue: TypeAlias = int | str | bool


class LRPImageExplainer(ImageExplainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)
        self.preprocessed = self.instance.preprocess()

    def set_explainer_options(self) -> Dict[OptionKey, OptionValue]:
        options: Dict[OptionKey, OptionValue] = self.options

        options["composite"] = st.selectbox(
            "**Composite**: Rule to use for relevance calculation",
            (
                "EpsilonPlus",
                "EpsilonPlusFlat",
                "EpsilonGammaBox",
                "EpsilonAlpha2Beta1",
                "EpsilonAlpha2Beta1Flat",
            ),
            key="lrp_composite",
        )
        options["attributor"] = st.selectbox(
            "**Attributor**: Attributor to use",
            ("Gradient", "SmoothGrad", "IntegratedGradients", "Occlusion"),
            key="lrp_attributor",
        )
        options["cmap"] = st.selectbox(
            "**Colourmap**: Colourmap scheme to use for heatmap visualisation",
            (
                "coldnhot",
                "hot",
                "cold",
                "gray",
                "wred",
                "wblue",
                "bwr",
                "france",
                "seismic",
                "coolio",
                "coleus",
            ),
            key="lrp_colourmap",
        )
        options["symmetricity"] = (
            True
            if st.radio(
                "**Symmetricity**: Whether or not to make the heatmap symmetric",
                ["True", "False"],
                horizontal=True,
                key="lrp_symmetricity",
            )
            == "True"
            else False
        )

        self.options = options
        return options

    def explain(self) -> List[Image.Image]:
        if self.options["top_labels"]:
            _, targets = torch.topk(
                self.model.predict(self.instance.preprocess()), k=self.options["top_labels"]
            )
            targets = targets.tolist()[0]
        else:
            targets = self.options["labels"]

        match self.options["composite"]:
            case "EpsilonPlus":
                composite = EpsilonPlus()
            case "EpsilonPlusFlat":
                composite = EpsilonPlusFlat()
            case "EpsilonGammaBox":
                low, high = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(
                    torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]).to(self.model.device)
                )
                composite = EpsilonGammaBox(low=low, high=high)
            case "EpsilonAlpha2Beta1":
                composite = EpsilonAlpha2Beta1()
            case "EpsilonAlpha2Beta1Flat":
                composite = EpsilonAlpha2Beta1Flat()

        match self.options["attributor"]:
            case "Gradient":
                attributor = Gradient(model=self.model.model, composite=composite)
            case "SmoothGrad":
                attributor = SmoothGrad(model=self.model.model, composite=composite)
            case "IntegratedGradients":
                attributor = IntegratedGradients(model=self.model.model, composite=composite)
            case "Occlusion":
                attributor = Occlusion(model=self.model.model, composite=composite)

        exp_label_list = [None] * len(targets)
        for i, target in enumerate(targets):
            _, attribution = attributor(
                self.instance.preprocess().double().to(self.model.device),
                torch.eye(1000)[[target]].double().to(self.model.device),
            )
            relevance = attribution.sum(1).cpu()
            exp_label_list[i] = (
                imgify(
                    relevance,
                    symmetric=self.options["symmetricity"],
                    cmap=self.options["cmap"],
                ),
                target,
            )

        return exp_label_list
