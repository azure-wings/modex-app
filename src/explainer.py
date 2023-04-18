from typing import Dict, TypeAlias, Any
from abc import ABC, abstractmethod

from model import Model
from instance import Instance

OptionKey: TypeAlias = str
OptionValue: TypeAlias = int | str | bool


class Explainer(ABC):
    def __init__(self, model: Model, instance: Instance):
        self.model = model
        self.instance = instance
        self.options = {}

    @abstractmethod
    def set_options(self) -> Dict[OptionKey, OptionValue]:
        pass

    @abstractmethod
    def explain(self) -> Any:
        pass


class ImageExplainer(Explainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)


class TextExplainer(Explainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)


class TabularExplainer(Explainer):
    def __init__(self, model: Model, instance: Instance):
        super().__init__(model, instance)


class ExplainerFactory(ABC):
    @abstractmethod
    def create_lrp_explainer(self) -> ImageExplainer:
        pass

    @abstractmethod
    def create_lime_explainer(self) -> TextExplainer:
        pass

    # @abstractmethod
    # def create_shap_explainer(self) -> TabularExplainer:
    #     pass


from modules.lrp_module import LRPImageExplainer
from modules.lime_module import (
    LIMEImageExplainer,
    LIMETextExplainer,
    LIMETabularExplainer,
)


class ImageExplainerFactory(ExplainerFactory):
    def create_lrp_explainer(self) -> LRPImageExplainer:
        return LRPImageExplainer

    def create_lime_explainer(self) -> LIMEImageExplainer:
        return LIMEImageExplainer


class TextExplainerFactory(ExplainerFactory):
    def create_lrp_explainer(self) -> None:
        raise ValueError("Text explanation is not supported for LRP method")

    def create_lime_explainer(self) -> LIMETextExplainer:
        return LIMETextExplainer


class TabularExplainerFactory(ExplainerFactory):
    def create_lrp_explainer(self) -> None:
        raise ValueError("Tabular explanation is not supported for LRP method")

    def create_lime_explainer(self) -> LIMETabularExplainer:
        return LIMETabularExplainer


def create_explainer(explainer_type: str, method: str) -> Explainer:
    creator_map = {
        ("Image", "LRP"): ImageExplainerFactory().create_lrp_explainer,
        ("Image", "LIME"): ImageExplainerFactory().create_lime_explainer,
        ("Text", "LIME"): TextExplainerFactory().create_lime_explainer,
        ("Tabular", "LIME"): TabularExplainerFactory().create_lime_explainer,
    }

    creator = creator_map.get((explainer_type, method))
    if creator:
        return creator
    else:
        raise ValueError(
            f"Invalid explainer type: {explainer_type} or method: {method}"
        )
