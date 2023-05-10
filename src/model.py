from typing import Any
import torch
from torch import nn


class Model:
    def __init__(self, checkpoint: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(checkpoint)

    def load_model(self, checkpoint: str) -> nn.Module:
        model = torch.load(
            checkpoint,
            map_location=torch.device(self.device),
        ).double()
        model.eval()
        return model

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        input_data = input_data.double().to(self.device)
        with torch.no_grad():
            output = self.model(input_data)
        return output
