import torch
import torch.nn as nn
from torchaudio.sox_effects import apply_effects_tensor


class SoxEffects(nn.Module):
    """Transform waveform tensors."""
    def __init__(
        self,
        sample_rate: int,
        sil_threshold: float,
        sil_duration: float,
    ):
        super().__init__()
        self.effects = [
            ["channels", "1"],  # convert to mono
            ["rate", f"{sample_rate}"],  # resample
            [
                "silence",
                "1",
                f"{sil_duration}",
                f"{sil_threshold}%",
                "-1",
                f"{sil_duration}",
                f"{sil_threshold}%",
            ],  # remove silence throughout the file
        ]

    def forward(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        wav : (C, T)
        """
        # 無音区間の切り取り
        wav, _ = apply_effects_tensor(wav, sample_rate, self.effects)
        return wav