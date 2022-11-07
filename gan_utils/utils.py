import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

from model.SimpleGAN.models import SimpleConv
from .updater import update_generator

from utils import make_pad_mask, make_pad_mask_for_loss

def make_descriminator(in_dim, device):
    model_d = SimpleConv(in_dim)
    return model_d.to(device)



def train_warmup_discriminator(base_model, model_d, dataloader, optimizer_d, mixing_prob, training_method, epoch, ckpt_time, device):
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    epoch_loss_feat_add = 0
    iter_cnt = 0
    all_iter = len(dataloader)
    
    base_model.train()
    model_d.train()
    
    for batch in dataloader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)
        
        output, dec_output, feat_add_out = base_model(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob) 
        B, C, T = output.shape
        
        loss_mse, loss_adv, loss_g = update_generator(model_d, optimizer_d, feature, output, mask)