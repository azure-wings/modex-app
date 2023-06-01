from typing import Dict, TypeAlias, Any
from abc import ABC, abstractmethod

import streamlit as st

from model import Model
from instance import Instance


OptionKey: TypeAlias = str
OptionValue: TypeAlias = int | str | bool


class Explainer(ABC):
    def __init__(self, model: Model, instance: Instance):
        self.model = model
        self.instance = instance
        self.options = dict()

    @abstractmethod
    def set_base_options(self) -> Dict[OptionKey, OptionValue]:
        pass

    @abstractmethod
    def set_explainer_options(self) -> Dict[OptionKey, OptionValue]:
        pass

    @abstractmethod
    def explain(self) -> Any:
        pass


class ImageExplainer(Explainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)

    def set_base_options(self) -> Dict[OptionKey, OptionValue]:
        options = self.options

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

        st.write("---")

        return options


class TextExplainer(Explainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)


class TabularExplainer(Explainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)


class ExplainerFactory(ABC):
    @abstractmethod
    def create_lrp_explainer(self) -> Explainer:
        pass

    @abstractmethod
    def create_lime_explainer(self) -> Explainer:
        pass

    @abstractmethod
    def create_kernelshap_explainer(self) -> Explainer:
        pass

    @abstractmethod
    def create_partitionshap_explainer(self) -> Explainer:
        pass


from modules.lrp_module import LRPImageExplainer
from modules.lime_module import (
    LIMEImageExplainer,
    LIMETextExplainer,
    LIMETabularExplainer,
)
from modules.kernelshap_module import KernelSHAPImageExplainer
from modules.partitionshap_module import PartitionSHAPImageExplainer


class ImageExplainerFactory(ExplainerFactory):
    def create_lrp_explainer(self) -> LRPImageExplainer:
        return LRPImageExplainer

    def create_lime_explainer(self) -> LIMEImageExplainer:
        return LIMEImageExplainer

    def create_kernelshap_explainer(self) -> KernelSHAPImageExplainer:
        return KernelSHAPImageExplainer

    def create_partitionshap_explainer(self) -> PartitionSHAPImageExplainer:
        return PartitionSHAPImageExplainer


class TextExplainerFactory(ExplainerFactory):
    def create_lrp_explainer(self) -> None:
        raise ValueError("Text explanation is not supported for LRP method")

    def create_lime_explainer(self) -> LIMETextExplainer:
        return LIMETextExplainer

    def create_kernelshap_explainer(self) -> None:
        raise NotImplementedError()

    def create_partitionshap_explainer(self) -> None:
        raise NotImplementedError()


class TabularExplainerFactory(ExplainerFactory):
    def create_lrp_explainer(self) -> None:
        raise ValueError("Tabular explanation is not supported for LRP method")

    def create_lime_explainer(self) -> LIMETabularExplainer:
        return LIMETabularExplainer

    def create_kernelshap_explainer(self) -> None:
        raise NotImplementedError()

    def create_partitionshap_explainer(self) -> None:
        raise NotImplementedError()


def create_explainer(explainer_type: str, method: str) -> Explainer:
    creator_map = {
        ("Image", "LRP"): ImageExplainerFactory().create_lrp_explainer,
        ("Image", "LIME"): ImageExplainerFactory().create_lime_explainer,
        ("Text", "LIME"): TextExplainerFactory().create_lime_explainer,
        ("Tabular", "LIME"): TabularExplainerFactory().create_lime_explainer,
        ("Image", "KernelSHAP"): ImageExplainerFactory().create_kernelshap_explainer,
        ("Image", "PartitionSHAP"): ImageExplainerFactory().create_partitionshap_explainer,
    }

    creator = creator_map.get((explainer_type, method))
    if creator:
        return creator
    else:
        raise ValueError(f"Invalid explainer type: {explainer_type} or method: {method}")
