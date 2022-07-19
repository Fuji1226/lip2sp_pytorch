"""
最終的なモデル

最終出力はpostnet出力とdecoder出力の和だったので変更
"""
import os
import sys
import glob

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

try:
    from model.net import ResNet3D
    from model.transformer_taguchi import Postnet, Encoder, Decoder
    from model.conformer.encoder import Conformer_Encoder
    from hparams import create_hparams
    from model.glu_taguchi import GLU
except:
    from net import ResNet3D
    from transformer_taguchi import Postnet, Encoder, Decoder
    from conformer.encoder import Conformer_Encoder
    from hparams import create_hparams
    from model.glu_taguchi import GLU


class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers,
        d_model, n_layers, n_head, dec_n_layers, dec_d_model,
        glu_inner_channels, glu_layers, glu_kernel_size,
        pre_in_channels, pre_inner_channels, post_inner_channels, post_n_layers,
        n_position, max_len, which_encoder, which_decoder, 
        training_method, num_passes, mixing_prob, apply_first_bn,
        dropout=0.1, reduction_factor=2, use_gc=False, input_layer_dropout=False):

        super().__init__()
        assert d_model % n_head == 0

        self.max_len = max_len
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.training_method = training_method
        self.num_passes = num_passes
        self.mixing_prob = mixing_prob
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor

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

    def forward(self, lip, prev=None, data_len=None, max_len=None, gc=None, training_method=None):
        """
        teacher forcingとscheduled samplingが途中で切り替わるように変更
        """
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # encoder
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)

        if self.which_encoder == "transformer":
            enc_output = self.encoder(lip_feature, data_len, self.max_len)    # (B, C, T)
        elif self.which_encoder == "conformer":
            enc_output = self.encoder(lip_feature, data_len, self.max_len)    # (B, T, C) ?
        
        # decoder
        # 学習時
        if prev is not None:
            if self.which_decoder == "transformer":
                dec_output = self.decoder(
                    enc_output, prev, data_len, self.max_len, 
                    training_method=training_method, 
                    num_passes=self.num_passes, 
                    mixing_prob=self.mixing_prob
                )
            elif self.which_decoder == "glu":
                dec_output = self.decoder(
                    enc_output, prev,
                    training_method=training_method, 
                    num_passes=self.num_passes, 
                    mixing_prob=self.mixing_prob
                )
            
        # 推論時
        else:
            dec_outputs = []
            max_decoder_time_steps = enc_output.shape[-1]   # (B, C, T)なので

            if self.which_decoder == "transformer":
                dec_output = self.decoder(enc_output)
                dec_outputs.append(dec_output)
                for _ in range(max_decoder_time_steps - 1):
                    dec_output = self.decoder(enc_output, dec_output)
                    dec_outputs.append(dec_output)
                dec_output = torch.cat(dec_outputs, dim=-1)
                assert dec_output.shape[-1] == max_decoder_time_steps * self.reduction_factor

            elif self.which_decoder == "glu":
                for t in range(max_decoder_time_steps):
                    if t == 0:
                        dec_output = self.decoder(enc_output[:, :, t].unsqueeze(-1))
                    else:
                        dec_output = self.decoder(enc_output[:, :, t].unsqueeze(-1), dec_output)
                    dec_outputs.append(dec_output)
                dec_output = torch.cat(dec_outputs, dim=-1)
                assert dec_output.shape[-1] == max_decoder_time_steps * self.reduction_factor
                
        # postnet
        out = self.postnet(dec_output) 
        return out, dec_output, enc_output

    def inference(self, lip, data_len=None, max_len=None, prev=None, gc=None):
        # encoder
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)
        
        if self.which_encoder == "transformer":
            enc_output = self.encoder(lip_feature)    # (B, T, C)
        elif self.which_encoder == "conformer":
            enc_output = self.encoder(lip_feature)
        
        # decoder
        if self.which_decoder == "transformer":
            dec_output = self.decoder.inference(enc_output, prev)
            out = self.postnet(dec_output) 
            
        elif self.which_decoder == "glu":
            dec_output = self.decoder.inference(enc_output, prev)
            self.pre = dec_output
            out = self.postnet(dec_output) 
        return out, dec_output, enc_output

    def reset_state(self):
        self.decoder.reset_state()


def main():
    batch_size = 8

    # data_len
    data_len = [300, 300, 300, 300, 100, 100, 200, 200]
    data_len = torch.tensor(data_len)

    # 口唇動画
    lip_channels = 5
    width = 48
    height = 48
    frames = 150
    lip = torch.rand(batch_size, lip_channels, width, height, frames)

    # 音響特徴量
    feature_channels = 80
    acoustic_feature = torch.rand(batch_size, feature_channels, frames * 2)

    # parameter
    hparams = create_hparams()

    # build
    net = Lip2SP(
        in_channels=5, 
        out_channels=hparams.out_channels,
        res_layers=hparams.res_layers,
        d_model=hparams.d_model,
        n_layers=hparams.n_layers,
        n_head=hparams.n_head,
        glu_inner_channels=hparams.glu_inner_channels,
        glu_layers=hparams.glu_layers,
        pre_in_channels=hparams.pre_in_channels,
        pre_inner_channels=hparams.pre_inner_channels,
        post_inner_channels=hparams.post_inner_channels,
        n_position=hparams.length * 10,  # 口唇動画に対して長ければいい
        max_len=hparams.length // 2,
        which_encoder=hparams.which_encoder,
        which_decoder=hparams.which_decoder,
        training_method=hparams.training_method,
        num_passes=hparams.num_passes,
        mixing_prob=hparams.mixing_prob,
        dropout=hparams.dropout,
        reduction_factor=hparams.reduction_factor,
        use_gc=hparams.use_gc,
    )

    # training
    outout, dec_output = net(lip=lip, data_len=data_len, prev=acoustic_feature)
    loss_f = nn.MSELoss()
    loss = loss_f(outout, acoustic_feature)
    print(loss)

    # inference
    # 口唇動画
    lip_channels = 5
    width = 48
    height = 48
    frames = 45
    lip = torch.rand(batch_size, lip_channels, width, height, frames)
    inference_out = net.inference(lip=lip)
    print(inference_out.shape)
    




if __name__ == "__main__":
    main()