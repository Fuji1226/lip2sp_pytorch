import torch
import torch.nn.functional as F


class STFTLoss:
    def __init__(self, n_fft, hop_length, win_length, device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length, device=device)

    def stft(self, x):
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        stft = torch.view_as_real(stft)

        real = stft[..., 0]
        imag = stft[..., 1]
        spec_mag = torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)
        return spec_mag
    
    def SpectralConvergenceLoss(self, x_mag, x_pred_mag):
        return torch.norm(x_pred_mag - x_mag, p="fro") / torch.norm(x_pred_mag, p="fro")

    def LogSTFTMagnitudeLoss(self, x_mag, x_pred_mag):
        return F.l1_loss(torch.log(x_pred_mag), torch.log(x_mag))

    def calc_loss(self, x, x_pred):
        x_mag = self.stft(x)
        x_pred_mag = self.stft(x_pred)

        sc_loss = self.SpectralConvergenceLoss(x_mag, x_pred_mag)
        lm_loss = self.LogSTFTMagnitudeLoss(x_mag, x_pred_mag)

        return sc_loss, lm_loss


class MultiResolutionSTFTLoss:
    def __init__(self, n_fft_list, hop_length_list, win_length_list, device):
        stft_losses = []
        for n_fft, hop_length, win_length in zip(n_fft_list, hop_length_list, win_length_list):
            stft_losses.append(
                STFTLoss(n_fft, hop_length, win_length, device)
            )
        self.stft_losses = stft_losses

    def calc_loss(self, x, x_pred):
        """
        x, x_pred : (B, 1, T)
        """
        x = x.squeeze(1)
        x_pred = x_pred.squeeze(1)

        sc_loss = 0
        lm_loss = 0
        for stft_loss in self.stft_losses:
            sc_l, lm_l = stft_loss.calc_loss(x, x_pred)
            sc_loss += sc_l
            lm_loss += lm_l

        sc_loss /= len(self.stft_losses)
        lm_loss /= len(self.stft_losses)

        return sc_loss + lm_loss