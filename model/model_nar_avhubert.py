import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))
sys.path.append(str(Path('~/lip2sp_pytorch/av_hubert/fairseq').expanduser()))
sys.path.append(str(Path('~/lip2sp_pytorch/av_hubert/avhubert').expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

import avhubert_utils as avhubert_utils
from fairseq import checkpoint_utils
import hubert_asr

from net import ResNet3D, ResNet3DVTP, ResNet3DRemake
from nar_decoder import ResTCDecoder, LinearDecoder
from rnn import GRUEncoder
from transformer_remake import Encoder
from grad_reversal import GradientReversal
from classifier import SpeakerClassifier
from resnet18 import ResNet18
from conformer.encoder import ConformerEncoder


class Lip2SP_NAR(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_inner_channels, which_res,
        rnn_n_layers, rnn_which_norm, trans_n_layers, trans_n_head, trans_pos_max_len,
        conf_n_layers, conf_n_head, conf_feedforward_expansion_factor,
        dec_n_layers, dec_kernel_size,
        n_speaker, spk_emb_dim,
        which_encoder, which_decoder, where_spk_emb, use_spk_emb,
        dec_dropout, res_dropout, rnn_dropout, is_large, adversarial_learning, reduction_factor):
        super().__init__()
        self.where_spk_emb = where_spk_emb
        self.adversarial_learning = adversarial_learning
        inner_channels = int(res_inner_channels * 8)

        # self.encoder = model

        # if use_spk_emb:
        #     self.gr_layer = GradientReversal(1.0)
        #     self.classfier = SpeakerClassifier(
        #         in_channels=inner_channels,
        #         hidden_channels=inner_channels,
        #         n_speaker=n_speaker,
        #     )
        #     self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        # if which_decoder == 'restc':
        #     self.decoder = ResTCDecoder(
        #         cond_channels=inner_channels,
        #         out_channels=out_channels,
        #         inner_channels=inner_channels,
        #         n_layers=dec_n_layers,
        #         kernel_size=dec_kernel_size,
        #         dropout=dec_dropout,
        #         reduction_factor=reduction_factor,
        #     )
        # elif which_decoder == 'linear':
        #     self.decoder = LinearDecoder(
        #         in_channels=inner_channels,
        #         out_channels=out_channels,
        #         reduction_factor=reduction_factor,
        #     )

    def forward(self, lip, lip_len, spk_emb=None):
        """
        lip : (B, C, H, W, T)
        lip_len : (B,)
        spk_emb : (B, C)
        """
        lip = lip.permute(0, 1, 3, 4, 2)    # (B, C, T, H, W)
        # feature, _ = self.encoder.extract_finetune(source={'video': lip, 'audio': None}, padding_mask=None, output_layer=None)



def make_model(cfg, device):
    model = Lip2SP_NAR(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_inner_channels=cfg.model.res_inner_channels,
        which_res=cfg.model.which_res,
        rnn_n_layers=cfg.model.rnn_n_layers,
        rnn_which_norm=cfg.model.rnn_which_norm,
        trans_n_layers=cfg.model.trans_enc_n_layers,
        trans_n_head=cfg.model.trans_enc_n_head,
        trans_pos_max_len=int(cfg.model.fps * cfg.model.input_lip_sec),
        conf_n_layers=cfg.model.conf_n_layers,
        conf_n_head=cfg.model.conf_n_head,
        conf_feedforward_expansion_factor=cfg.model.conf_feed_forward_expansion_factor,
        dec_n_layers=cfg.model.tc_n_layers,
        dec_kernel_size=cfg.model.tc_kernel_size,
        n_speaker=len(cfg.train.speaker),
        spk_emb_dim=cfg.model.spk_emb_dim,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        where_spk_emb=cfg.train.where_spk_emb,
        use_spk_emb=cfg.train.use_spk_emb,
        dec_dropout=cfg.train.dec_dropout,
        res_dropout=cfg.train.res_dropout,
        rnn_dropout=cfg.train.rnn_dropout,
        is_large=cfg.model.is_large,
        adversarial_learning=cfg.train.adversarial_learning,
        reduction_factor=cfg.model.reduction_factor,
    )
    return model.to(device)


def make_avhubert():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = '/home/minami/av_hubert_data/base_vox_iter5.pt'
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(models[0], 'decoder'):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")
    return model
avhubet = make_avhubert()

import hydra
@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    print('here')
# def main():

    # B = 8
    # C = 1
    # T = 250
    # H = 88
    # W = 88
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # model.train()

    # for i in range(10):
    #     x = torch.rand(B, C, T, H, W).to(device)
    #     padding_mask = torch.zeros(B, T).to(device)
    #     for i in range(padding_mask.shape[0]):
    #         padding_mask[i][-int(5 * (i + 1)):] = 1
        
    #     feature, _ = model.extract_finetune(
    #         source={'video': x, 'audio': None}, 
    #         padding_mask=padding_mask, 
    #     )
    #     y = torch.rand_like(feature)

    #     loss = torch.mean((feature - y) ** 2)

    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    #     print(feature.shape)
    #     print(loss)


if __name__ == '__main__':
    main()
    