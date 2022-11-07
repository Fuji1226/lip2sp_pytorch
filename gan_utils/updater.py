import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))


import torch
from torch import nn
from torch.autograd import Variable
from loss import MaskedLoss


def update_discriminator(model_d, optimizer_d, x, y, y_hat,
                         mask, phase, eps=1e-20):
    
    T = mask.sum().item()

    # Real
    D_real = model_d(y)
    real_correct_count = ((D_real > 0.5).float() * mask).sum().item()

    # Fake
    D_fake = model_d(y_hat)
    fake_correct_count = ((D_fake < 0.5).float() * mask).sum().item()

    # Loss
    loss_real_d = -(torch.log(D_real + eps) * mask).sum() / T
    loss_fake_d = -(torch.log(1 - D_fake + eps) * mask).sum() / T
    loss_d = loss_real_d + loss_fake_d

    if phase == "train":
        loss_d.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model_d.parameters(), 1.0)
        optimizer_d.step()

    return loss_d.item(), loss_fake_d.item(), loss_real_d.item(),\
        real_correct_count, fake_correct_count


def update_generator(model_g, model_d, optimizer_g,
                     x, y, y_hat,
                     adv_w, lengths, mask, phase,
                     mse_w=None, eps=1e-20):
    T = mask.sum().item()

    criterion = MaskedLoss()

    # MSELoss
    loss_mse = criterion(y_hat, y, mask=mask)

    # Adversarial loss
    if adv_w > 0:
        loss_adv = -(torch.log(model_d(
            y_hat) + eps) * mask).sum() / T
    else:
        loss_adv = Variable(y.data.new(1).zero_())

    # MSE + MGE + ADV loss
    # try to decieve discriminator
    loss_g = mse_w * loss_mse + adv_w * loss_adv
    if phase == "train":
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(model_g.parameters(), 1.0)
        optimizer_g.step()

    return loss_mse.item(), loss_adv.item(), loss_g.item()
