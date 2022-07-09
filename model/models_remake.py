"""
最終的なモデル
"""
import os
import sys
import glob

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from model.net import ResNet3D
    from model.transformer_taguchi import Postnet, Encoder, Decoder
    # from model.transformer_remake import Postnet, Encoder, Decoder
    from model.conformer.encoder import Conformer_Encoder
    from hparams import create_hparams
    from model.glu_remake import GLU
except:
    from net import ResNet3D
    from transformer_taguchi import Postnet, Encoder, Decoder
    # from transformer_remake import Postnet, Encoder, Decoder
    from conformer.encoder import Conformer_Encoder
    from hparams import create_hparams
    from glu_remake import GLU


class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers,
        d_model, n_layers, n_head, dec_n_layers, dec_d_model,
        glu_inner_channels, glu_layers, glu_kernel_size,
        pre_in_channels, pre_inner_channels, post_inner_channels, post_n_layers,
        n_position, max_len, which_encoder, which_decoder, apply_first_bn,
        dropout=0.1, reduction_factor=2, use_gc=False, input_layer_dropout=False, diag_mask=False):
        super().__init__()
        assert d_model % n_head == 0
        assert which_encoder == "transformer" or "conformer"
        assert which_decoder == "transformer" or "glu"
        self.max_len = max_len
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels

        if apply_first_bn:
            self.first_batch_norm = nn.BatchNorm3d(in_channels)

        self.ResNet_GAP = ResNet3D(in_channels, d_model, res_layers, input_layer_dropout, dropout)

        # encoder
        if self.which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=n_layers, 
                n_head=n_head, 
                d_model=d_model, 
                n_position=n_position, 
                reduction_factor=reduction_factor, 
                dropout=dropout,
            )
        elif self.which_encoder == "conformer":
            self.encoder = Conformer_Encoder(
                encoder_dim=d_model, num_layers=n_layers, num_attention_heads=n_head, reduction_factor=reduction_factor
            )

        # decoder
        if self.which_decoder == "transformer":
            self.decoder = Decoder(
                dec_n_layers=dec_n_layers, 
                n_head=n_head, 
                dec_d_model=dec_d_model, 
                pre_in_channels=pre_in_channels, 
                pre_inner_channels=pre_inner_channels, 
                out_channels=out_channels, 
                n_position=n_position, 
                reduction_factor=reduction_factor, 
                dropout=dropout, 
                use_gc=use_gc,
                diag_mask=diag_mask,
            )
        elif self.which_decoder == "glu":
            self.decoder = GLU(
                glu_inner_channels, 
                out_channels,
                pre_in_channels, 
                pre_inner_channels,
                reduction_factor, 
                glu_layers,
                kernel_size=glu_kernel_size,
                dropout=dropout,
            )

        self.postnet = Postnet(out_channels, post_inner_channels, out_channels, post_n_layers)

    def forward(self, lip, prev=None, data_len=None, max_len=None, gc=None, training_method=None, mixing_prob=None, visualize=False, epoch=None, iter_cnt=None):
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # print("\n##### input #####")
        # print("----- lip -----")
        # print(f"lip.grad = {lip.grad}")
        # print(f"lip.requires_grad = {lip.requires_grad}")
        # print(f"lip.is_leaf = {lip.is_leaf}")
        # print("----- prev -----")
        # print(f"prev.grad = {prev.grad}")
        # print(f"prev.requires_grad = {prev.requires_grad}")
        # print(f"prev.is_leaf = {prev.is_leaf}")

        # encoder
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)

        # print("\n##### after resnet ######")
        # print("----- lip_feature -----")
        # print(f"lip_feature.grad = {lip_feature.grad}")
        # print(f"lip_feature.requires_grad = {lip_feature.requires_grad}")
        # print(f"lip_feature.is_leaf = {lip_feature.is_leaf}")
        
        if self.which_encoder == "transformer":
            enc_output = self.encoder(lip_feature, data_len, self.max_len)    # taguchi : (B, C, T), remake : (B, T, C)
        elif self.which_encoder == "conformer":
            enc_output = self.encoder(lip_feature, data_len, self.max_len)    # (B, T, C) 
        
        # print("\n##### after encoder ######")
        # print("----- enc_output -----")
        # print(f"enc_output.grad = {enc_output.grad}")
        # print(f"enc_output.requires_grad = {enc_output.requires_grad}")
        # print(f"enc_output.is_leaf = {enc_output.is_leaf}")
        # print("----- prev -----")
        # print(f"prev.grad = {prev.grad}")
        # print(f"prev.requires_grad = {prev.requires_grad}")
        # print(f"prev.is_leaf = {prev.is_leaf}")

        # decoder
        # 学習時
        B = enc_output.shape[0]
        T = enc_output.shape[-1]
        D = self.out_channels
        if prev is not None:
            if training_method == "tf":
                dec_output = self.decoder_forward(enc_output, prev, data_len, self.max_len)

            elif training_method == "ss":
                if visualize:
                    plot_data = prev.to('cpu').detach().numpy().copy()
                    plot_data = plot_data.transpose(0, -1, -2)
                    plot_data = plot_data.reshape(B, -1, D)
                    plot_data = plot_data.transpose(0, -1, -2)
                    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                    img = librosa.display.specshow(
                        data=plot_data[0],
                        x_axis='time',
                        y_axis='mel',
                        sr=16000,
                        fmax=7600,
                        fmin=0,
                        n_fft=640,
                        hop_length=160,
                        win_length=640,
                        ax=ax[0],
                        cmap='viridis'
                    )
                    ax[0].set(title="target melspectrogram")
                    ax[0].label_outer()

                with torch.no_grad():
                    dec_output = self.decoder_forward(enc_output, prev, data_len, self.max_len)

                # print("\n----- dec_output for scheduled sampling -----")
                # print(f"dec_output.grad = {dec_output.grad}")
                # print(f"dec_output.requires_grad = {dec_output.requires_grad}")
                # print(f"dec_output.is_leaf = {dec_output.is_leaf}")
                # print("----- prev -----")
                # print(f"prev.grad = {prev.grad}")
                # print(f"prev.requires_grad = {prev.requires_grad}")
                # print(f"prev.is_leaf = {prev.is_leaf}")

                    # mixing_prob分だけtargetを選択し，それ以外をdec_outputに変更することで混ぜる
                    mixing_prob = torch.zeros_like(prev) + mixing_prob
                    judge = torch.bernoulli(mixing_prob)
                    mixed_prev = torch.where(judge == 1, prev, dec_output)

                # print("\n----- dec_output for scheduled sampling after mixing -----")
                # print(f"dec_output.grad = {dec_output.grad}")
                # print(f"dec_output.requires_grad = {dec_output.requires_grad}")
                # print(f"dec_output.is_leaf = {dec_output.is_leaf}")
                # print("----- prev -----")
                # print(f"prev.grad = {prev.grad}")
                # print(f"prev.requires_grad = {prev.requires_grad}")
                # print(f"prev.is_leaf = {prev.is_leaf}")

                if visualize:
                    plot_data = mixed_prev.to('cpu').detach().numpy().copy()
                    plot_data = plot_data.transpose(0, -1, -2)
                    plot_data = plot_data.reshape(B, -1, D)
                    plot_data = plot_data.transpose(0, -1, -2)
                    librosa.display.specshow(
                        data=plot_data[0],
                        x_axis='time',
                        y_axis='mel',
                        sr=16000,
                        fmax=7600,
                        fmin=0,
                        n_fft=640,
                        hop_length=160,
                        win_length=640,
                        ax=ax[1],
                        cmap='viridis'
                    )
                    ax[1].set(title="mixed melspectrogram")
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    ax[1].label_outer()
                    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
                    plt.savefig(save_path / f"ss_mel_epoch{epoch}_iter{iter_cnt}.png")

                # 混ぜたやつでもう一回計算させる
                dec_output = self.decoder_forward(enc_output, mixed_prev, data_len, self.max_len)
                # print("\n----- dec_output after scheduled sampling -----")
                # print(f"dec_output.grad = {dec_output.grad}")
                # print(f"dec_output.requires_grad = {dec_output.requires_grad}")
                # print(f"dec_output.is_leaf = {dec_output.is_leaf}")
        # 推論時
        else:
            dec_output = self.decoder_inference(enc_output, visualize=visualize)

        # postnet
        out = self.postnet(dec_output) 
        # print("\n----- out -----")
        # print(f"out.grad = {out.grad}")
        # print(f"out.requires_grad = {out.requires_grad}")
        # print(f"out.is_leaf = {out.is_leaf}")
        return out, dec_output, enc_output

    def decoder_forward(self, enc_output, prev=None, data_len=None, max_len=None, mode="training"):
        """
        学習時の処理
        """
        print("----- start forward -----")
        # print(f"enc_output = {enc_output}")
        if self.which_decoder == "transformer":
            dec_output = self.decoder(enc_output, prev, data_len, self.max_len, mode=mode)

        elif self.which_decoder == "glu":
            dec_output = self.decoder(enc_output, prev, mode=mode)

        return dec_output

    def decoder_inference(self, enc_output, mode="inference", visualize=False):
        """
        推論時の処理
        """
        print("----- start inference -----")
        dec_outputs = []
        max_decoder_time_steps = enc_output.shape[-1]   # (B, C, T)なので

        # print(f"enc_output = {enc_output}")

        if self.which_decoder == "transformer":
            for t in range(max_decoder_time_steps):
                if t == 0:
                    dec_output = self.decoder(enc_output, mode=mode)
                else:
                    dec_output = self.decoder(enc_output, dec_outputs[-1], mode=mode)
                dec_outputs.append(dec_output)

                if visualize:
                    plot_data = torch.cat(dec_outputs, dim=-1)
                    plot_data = plot_data.to('cpu').detach().numpy().copy()
                    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
                    img = librosa.display.specshow(
                        data=plot_data[0],
                        x_axis='time',
                        y_axis='mel',
                        sr=16000,
                        fmax=7600,
                        fmin=0,
                        n_fft=640,
                        hop_length=160,
                        win_length=640,
                        ax=ax,
                        cmap='viridis'
                    )
                    ax.set(title="generate melspectrogram")
                    ax.label_outer()
                    save_path = Path("~/lip2sp_pytorch/data_check/inference").expanduser()
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(save_path / f"generate_mel_{t}.png")

        elif self.which_decoder == "glu":
            for t in range(max_decoder_time_steps):
                if t == 0:
                    dec_output = self.decoder(enc_output[:, :, t].unsqueeze(-1), mode=mode)
                else:
                    dec_output = self.decoder(enc_output[:, :, t].unsqueeze(-1), dec_outputs[-1], mode=mode)
                dec_outputs.append(dec_output)

                if visualize:
                    plot_data = torch.cat(dec_outputs, dim=-1)
                    plot_data = plot_data.to('cpu').detach().numpy().copy()
                    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
                    img = librosa.display.specshow(
                        data=plot_data[0],
                        x_axis='time',
                        y_axis='mel',
                        sr=16000,
                        fmax=7600,
                        fmin=0,
                        n_fft=640,
                        hop_length=160,
                        win_length=640,
                        ax=ax,
                        cmap='viridis'
                    )
                    ax.set(title="generate melspectrogram")
                    ax.label_outer()
                    save_path = Path("~/lip2sp_pytorch/data_check/inference").expanduser()
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(save_path / f"generate_mel_{t}.png")

        # 溜め込んだ出力を時間方向に結合して最終出力にする
        dec_output = torch.cat(dec_outputs, dim=-1)
        # print(f"dec_output = {dec_output}")
        assert dec_output.shape[-1] == max_decoder_time_steps * self.reduction_factor
        return dec_output

    def reset_state(self):
        self.decoder.reset_state()

    # def inference(self, lip, data_len=None, max_len=None, prev=None, gc=None):
    #     # encoder
    #     if self.apply_first_bn:
    #         lip = self.first_batch_norm(lip)
    #     lip_feature = self.ResNet_GAP(lip)
        
    #     if self.which_encoder == "transformer":
    #         enc_output = self.encoder(lip_feature)    # (B, T, C)
    #     elif self.which_encoder == "conformer":
    #         enc_output = self.encoder(lip_feature)
        
    #     # decoder
    #     if self.which_decoder == "transformer":
    #         dec_output = self.decoder.inference(enc_output, prev)
    #         out = self.postnet(dec_output) 
            
    #     elif self.which_decoder == "glu":
    #         dec_output = self.decoder.inference(enc_output, prev)
    #         self.pre = dec_output
    #         out = self.postnet(dec_output) 
    #     return out, dec_output, enc_output



def make_model(cfg, device):
    model = Lip2SP(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            res_layers=cfg.model.res_layers,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_head=cfg.model.n_head,
            dec_n_layers=cfg.model.dec_n_layers,
            dec_d_model=cfg.model.dec_d_model,
            glu_inner_channels=cfg.model.d_model,
            glu_layers=cfg.model.glu_layers,
            glu_kernel_size=cfg.model.glu_kernel_size,
            pre_in_channels=cfg.model.pre_in_channels,
            pre_inner_channels=cfg.model.pre_inner_channels,
            post_inner_channels=cfg.model.post_inner_channels,
            post_n_layers=cfg.model.post_n_layers,
            n_position=cfg.model.length * 5,
            max_len=cfg.model.length // 2,
            which_encoder=cfg.model.which_encoder,
            which_decoder=cfg.model.which_decoder,
            apply_first_bn=cfg.train.apply_first_bn,
            dropout=cfg.train.dropout,
            reduction_factor=cfg.model.reduction_factor,
            use_gc=cfg.train.use_gc,
            input_layer_dropout=cfg.train.input_layer_dropout,
            diag_mask=cfg.model.diag_mask,
        ).to(device)
    return model

import hydra
@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    torch.manual_seed(7)
    batch_size = 1

    # data_len
    data_len = [30, 20, 10]
    if batch_size == 1:
        data_len = [30]
    data_len = torch.tensor(data_len)
    max_len = 30

    # 口唇動画
    lip_channels = 5
    width = 48
    height = 48
    frames = 15
    lip = torch.rand(batch_size, lip_channels, width, height, frames)

    # 音響特徴量
    feature_channels = 4
    feature = torch.rand(batch_size, feature_channels, frames * 2)

    device = 'cpu'
    model = make_model(cfg, device)

    # forward
    model.eval()
    out, dec_output, enc_output = model(
        lip=lip,
        data_len=data_len,
        prev=feature,
        training_method="tf",
        mixing_prob=0.5,
        visualize=False,
        epoch=1,
        iter_cnt=1,
    )
    print(out.shape)

    # inference
    lip = lip[:1]
    print(lip.shape)
    model.eval()
    out, dec_output, enc_output = model(
        lip=lip,
        visualize=False,
    )
    print(out.shape)

if __name__ == "__main__":
    main()