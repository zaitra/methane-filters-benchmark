import numpy as np
import torch
from typing import Dict
import pytorch_lightning as pl
from kornia.morphology import dilation as kornia_dilation
from kornia.morphology import erosion as kornia_erosion
import onnxruntime as ort

def pred_classification(pred_binary:torch.Tensor) -> torch.Tensor:
    n_pixels = (10 * np.prod(tuple(pred_binary.shape[-2:]))) / (64 ** 2)
    return (torch.sum(pred_binary, dim=(-1, -2)) > n_pixels).long()  # (B, 1)

def differences(y_pred_binary: torch.Tensor, y_gt:torch.Tensor) -> torch.Tensor:
    return 2 * y_pred_binary.long() + (y_gt == 1).long()

def binary_opening(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    eroded = torch.clamp(kornia_erosion(x.float(), kernel), 0, 1) > 0
    return torch.clamp(kornia_dilation(eroded.float(), kernel), 0, 1) > 0

def normalize_x(params_inputs, x):
    offsets, factors, clip_min, clip_max = params_inputs["offset"], params_inputs["factor"], params_inputs["clip"][0], params_inputs["clip"][1]
    return torch.clamp((x-offsets) / factors, clip_min, clip_max).float()

class Mag1cBaseline(pl.LightningModule):
    """
    Uses mag1c output with morphological operations as a baseline
    """

    def __init__(self, mag1c_threshold: float = 500.0, normalizer_params = {'offset': 0, 'factor': 1750, 'clip': (0, 2)}):
        super().__init__()
        self.mag1c_threshold = mag1c_threshold
        self.element_stronger = torch.nn.Parameter(torch.from_numpy(np.array([[0, 1, 0],
                                                                              [1, 1, 1],
                                                                              [0, 1, 0]])).float(), requires_grad=False)

        # 'mag1c': {'offset': 0, 'factor': 1750, 'clip': (0, 2)},
        self.normalizer_params = normalizer_params
        self.session = ort.InferenceSession("/home/jherec/methane-filters-benchmark/linknet_mag1c-sas.onnx")
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #mag1c = x[:, self.band_mag1c:(self.band_mag1c + 1)]
        return x

    def apply_threshold(self, pred: torch.Tensor, threshold) -> torch.Tensor:
        mag1c_thresholded = (pred > threshold)

        # https://programtalk.com/python-more-examples/kornia.morphology.dilation.bool/
        return binary_opening(mag1c_thresholded, self.element_stronger).long()

    def batch_with_preds(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = batch.copy()
        pred = self(batch["input"])

        batch["input_norm"] = normalize_x(self.normalizer_params, batch["input"])
        batch["output_norm"] = batch["output"]

        batch["prediction"] = pred  # torch.sigmoid((pred - self.mag1c_threshold) / 250)

        batch["pred_binary"] = self.apply_threshold(pred, self.mag1c_threshold)
        batch["differences"] = differences(batch["pred_binary"], batch["output_norm"].long())

        batch["pred_classification"] = pred_classification(batch["pred_binary"])

        return batch
    
    def batch_with_preds_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = batch.copy()

        batch["input_norm"] = batch["input"].numpy().astype(np.float32)
        batch["output_norm"] = batch["output"]
        batch["prediction"] = torch.sigmoid(torch.tensor(self.session.run(None, {self.input_name: batch["input_norm"]})[0]))

        batch["pred_binary"] = batch["prediction"] > 0.5
        batch["differences"] = differences(batch["pred_binary"], batch["output_norm"].long())

        batch["pred_classification"] = pred_classification(batch["pred_binary"])

        return batch
    

