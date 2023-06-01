from typing import Any
import torch
from torch import nn
import torch.nn.functional as F


class Model:
    def __init__(self, checkpoint: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(checkpoint)

    def load_model(self, checkpoint: str) -> nn.Module:
        model = torch.load(checkpoint).float().to(self.device)
        model.eval()
        return model

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        input_data = input_data.float().to(self.device)
        with torch.no_grad():
            output = self.model(input_data)

        # Apply softmax if prediction is logits
        if torch.mean(torch.abs(torch.sum(output, dim=1) - 1), dim=0) > 1e-5:
            output = F.softmax(output, dim=1)

        return output
