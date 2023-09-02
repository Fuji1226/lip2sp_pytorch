import hydra
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm
import torch

from data_check import save_data, save_data_pwg
from train_audio_ae import make_model
from train_audio_ae_cycle import make_converter
from parallelwavegan.pwg_train import make_model as make_pwg
from utils import make_test_loader_with_external_data, get_path_test, load_pretrained_model, gen_data_separate, gen_data_concat, select_checkpoint
from calc_accuracy import calc_accuracy, calc_mean

current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(
    cfg,
    lip_encoder,
    audio_encoder,
    audio_decoder,
    lip2audio_converter,
    audio2lip_converter,
    test_loader,
    dataset,
    device,
    save_path,
):
    lip_encoder.eval()
    audio_encoder.eval()
    audio_decoder.eval()
    lip2audio_converter.eval()
    audio2lip_converter.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)

    for batch in tqdm(test_loader, total=(len(test_loader))):
        wav, lip, feature, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        lang_id = lang_id.to(device)

        lip_sep = gen_data_separate(lip, int(cfg.model.input_lip_sec * cfg.model.fps), cfg.model.fps)
        lip_len = lip_len.expand(lip_sep.shape[0])
        spk_emb = spk_emb.expand(lip_sep.shape[0], -1)
        lang_id = lang_id.expand(lip_sep.shape[0])

        if cfg.test.save_pred_lip:
            with torch.no_grad():
                lip_enc_output = lip_encoder(lip, lip_len)
                audio_enc_output_from_lip = lip2audio_converter(lip_enc_output, lip_len)
                feature_pred_lip = audio_decoder(audio_enc_output_from_lip, lip_len, spk_emb, lang_id)
            
            feature_pred_lip = gen_data_concat(
                feature_pred_lip, 
                int(cfg.model.fps * cfg.model.reduction_factor), 
                int((lip_len[0] % cfg.model.fps) * cfg.model.reduction_factor)
            )

            _save_path = save_path / "griffinlim_lip" / speaker[0] / filename[0]
            _save_path.mkdir(parents=True, exist_ok=True)
            save_data(
                cfg=cfg,
                save_path=_save_path,
                wav=wav,
                lip=lip,
                feature=feature,
                output=feature_pred_lip,
                lip_mean=lip_mean,
                lip_std=lip_std,
                feat_mean=feat_mean,
                feat_std=feat_std,
            )


@hydra.main(config_name='config', config_path='conf')
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model_path = select_checkpoint(cfg)

    lip_encoder, audio_encoder, audio_decoder = make_model(cfg, device)
    lip2audio_converter, audio2lip_converter = make_converter(cfg, device)

    lip_encoder = load_pretrained_model(model_path, lip_encoder, "lip_encoder")
    audio_encoder = load_pretrained_model(model_path, audio_encoder, "audio_encoder")
    audio_decoder = load_pretrained_model(model_path, audio_decoder, "audio_decoder")
    lip2audio_converter = load_pretrained_model(model_path, lip2audio_converter, "lip2audio_converter")
    audio2lip_converter = load_pretrained_model(model_path, audio2lip_converter, "audio2lip_converter")
    cfg.train.face_or_lip = model_path.parents[1].name
    cfg.test.face_or_lip = model_path.parents[1].name

    data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)
    
    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader_with_external_data(cfg, data_root, train_data_root)
        generate(
            cfg=cfg,
            lip_encoder=lip_encoder,
            audio_encoder=audio_encoder,
            audio_decoder=audio_decoder,
            lip2audio_converter=lip2audio_converter,
            audio2lip_converter=audio2lip_converter,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
        )

    for data_root, save_path in zip(data_root_list, save_path_list):
        for speaker in cfg.test.speaker:
            if cfg.test.save_pred_audio:
                save_path_spk_gl_audio = save_path / "griffinlim_audio" / speaker
                calc_accuracy(save_path_spk_gl_audio, save_path.parents[0], cfg, "accuracy_griffinlim_audio")
            if cfg.test.save_pred_lip:
                save_path_spk_gl_lip = save_path / "griffinlim_lip" / speaker
                calc_accuracy(save_path_spk_gl_lip, save_path.parents[0], cfg, "accuracy_griffinlim_lip")

        if cfg.test.save_pred_audio:
            calc_mean(save_path.parents[0] / 'accuracy_griffinlim_audio.txt')
        if cfg.test.save_pred_lip:
            calc_mean(save_path.parents[0] / 'accuracy_griffinlim_lip.txt')


if __name__ == '__main__':
    main()