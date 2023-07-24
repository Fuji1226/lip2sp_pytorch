import torch


def feature_matching_loss(fmaps_real_list, fmaps_pred_list):
    loss = 0
    for fmaps_real, fmaps_pred in zip(fmaps_real_list, fmaps_pred_list):
        for fmap_real, fmap_pred in zip(fmaps_real, fmaps_pred):
            loss += torch.mean(torch.abs(fmap_real - fmap_pred))
    return loss


def discriminator_loss(out_real_list, out_pred_list):
    loss = 0
    for out_real, out_pred in zip(out_real_list, out_pred_list):
        loss_real = torch.mean((1 - out_real) ** 2)
        loss_pred = torch.mean(out_pred ** 2)
        loss += (loss_real + loss_pred)
    return loss


def generator_loss(out_pred_list):
    loss = 0
    for out_pred in out_pred_list:
        loss += torch.mean((1 - out_pred) ** 2)
    return loss