import hydra
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
from data_check import save_data, save_data_pwg
from train_nar_ssl import make_model
from parallelwavegan.pwg_train import make_model as make_pwg
from utils import (
    make_test_loader_with_external_data_raw,
    get_path_test_raw,
    load_pretrained_model,
    gen_data_separate,
    gen_data_concat,
    select_checkpoint,
    fix_random_seed,
    delete_unnecessary_checkpoint,
)
from calc_accuracy import calc_accuracy_new, calc_mean, calc_accuracy_en

current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def generate(
        cfg,
        model,
        pwg,
        test_loader,
        dataset,
        device,
        save_path,
):
    model.eval()
    pwg.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        lip = lip.to(device)
        feature = feature.to(device)
        feature_avhubert = feature_avhubert.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)

        lip_sep = gen_data_separate(lip, int(cfg.model.input_lip_sec * cfg.model.fps), cfg.model.fps)
        feature_avhubert_sep = gen_data_separate(feature_avhubert, int(cfg.model.input_lip_sec * cfg.model.fps), cfg.model.fps)
        lip_len = lip_len.expand(lip_sep.shape[0])
        spk_emb = spk_emb.expand(lip_sep.shape[0], -1)

        with torch.no_grad():
            output = model(
                lip=lip,
                audio=None,
                lip_len=lip_len,
                spk_emb=spk_emb,
            )

        output = gen_data_concat(
            output, 
            int(cfg.model.fps * cfg.model.reduction_factor), 
            int((lip_len[0] % cfg.model.fps) * cfg.model.reduction_factor)
        )

        _save_path = save_path / "griffinlim" / speaker[0] / filename[0]
        _save_path.mkdir(parents=True, exist_ok=True)
        save_data(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            lip=lip,
            feature=feature,
            output=output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        
        with torch.no_grad():
            noise = torch.randn(output.shape[0], 1, output.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
            wav_pred = pwg(noise, output)
            wav_abs = pwg(noise, feature)

        _save_path = save_path / "pwg" / speaker[0] / filename[0]
        _save_path.mkdir(parents=True, exist_ok=True)
        save_data_pwg(
            cfg=cfg,
            save_path=_save_path,
            target=wav,
            output=wav_pred,
            ana_syn=wav_abs,
        )


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg):
    fix_random_seed(cfg.train.random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    pwg, disc = make_pwg(cfg, device)
    model_path_pwg = Path(cfg.test.pwg_path).expanduser()
    pwg = load_pretrained_model(model_path_pwg, pwg, "gen")

    model_path = select_checkpoint(cfg)
    model = make_model(cfg, device)
    model = load_pretrained_model(model_path, model, "model")
    cfg.train.face_or_lip = model_path.parents[2].name
    cfg.test.face_or_lip = model_path.parents[2].name    

    video_dir, audio_dir, save_path = get_path_test_raw(cfg, model_path)
    test_loader, test_dataset = make_test_loader_with_external_data_raw(cfg, video_dir, audio_dir)
    
    generate(
        cfg=cfg,
        model=model,
        pwg=pwg,
        test_loader=test_loader,
        dataset=test_dataset,
        device=device,
        save_path=save_path,
    )
    
    for speaker in cfg.test.speaker:
        save_path_spk = save_path / "griffinlim" / speaker
        save_path_pwg_spk = save_path / "pwg" / speaker
        if cfg.train.tcd_timit.use:
            calc_accuracy_en(save_path_spk, save_path.parents[0], cfg, "accuracy_griffinlim")
            calc_accuracy_en(save_path_pwg_spk, save_path.parents[0], cfg, "accuracy_pwg")
        else:
            calc_accuracy_new(save_path_spk, save_path.parents[0], cfg, "accuracy_griffinlim")
            calc_accuracy_new(save_path_pwg_spk, save_path.parents[0], cfg, "accuracy_pwg")
    calc_mean(save_path.parents[0] / 'accuracy_griffinlim.txt')
    calc_mean(save_path.parents[0] / 'accuracy_pwg.txt')
        
    delete_unnecessary_checkpoint(
        result_dir=save_path.parents[3],
        checkpoint_dir=model_path.parents[1],
    )


if __name__ == "__main__":
    main()