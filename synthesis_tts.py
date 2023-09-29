import os
import time
import torch
import matplotlib.pyplot as plt
from librosa.display import specshow
import seaborn as sns
from synthesis import save_stop_token, check_attention_weight


def generate_for_tts(cfg, model, test_loader, dataset, device, save_path, epoch):
    model.eval()

    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)

    process_times = []

    iter_cnt = 0
    for batch in test_loader:
        wav, feature, text, stop_token, feature_len, text_len, filename, label = batch

        text = text.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        feature_len = feature_len.to(device)
        
        start_time = time.time()
        with torch.no_grad():
            dec_output, output, logit, att_w = model(text, text_len)
            #tf_output, _, _ = model(lip=lip, prev=feature)


      
        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / 'synthesis' / f'epoch_{epoch}_FR'
        os.makedirs(str(_save_path), exist_ok=True)
        _save_path = _save_path / filename[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        #os.makedirs(_save_path, exist_ok=True)
        logit = torch.nn.functional.softmax(logit)
        logit = logit[0].to('cpu').detach().numpy().copy()
        save_stop_token(logit, str(_save_path))
        
        feature_tmp = feature[0].to('cpu').detach().numpy().copy()
        output_tmp = output[0].to('cpu').detach().numpy().copy()
        dec_output_tmp = dec_output[0].to('cpu').detach().numpy().copy()


        plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)
        ax = plt.subplot(3, 1, 1)
        specshow(
        data=feature_tmp, 
        x_axis="time", 
        y_axis="mel", 
        sr=16000,
        cmap="viridis",
        )
        plt.colorbar(format="%+2.f dB")
        plt.xlabel("Time[s]")
        plt.ylabel("Frequency[Hz]")
        plt.title("target")
        ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
        specshow(
        data=output_tmp, 
        x_axis="time", 
        y_axis="mel", 
        sr=16000,
        cmap="viridis",
        )
        plt.colorbar(format="%+2.f dB")
        plt.xlabel("Time[s]")
        plt.ylabel("Frequency[Hz]")
        plt.title("output")

        ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
        specshow(
        data=dec_output_tmp, 
        x_axis="time", 
        y_axis="mel", 
        sr=16000,
        cmap="viridis",
        )
        plt.colorbar(format="%+2.f dB")
        plt.xlabel("Time[s]")
        plt.ylabel("Frequency[Hz]")
        plt.title("dec_output")
        plt.tight_layout()

        plt.savefig(str(_save_path))
        plt.close()

        check_attention_weight(att_w[0], "attention", str(_save_path))

        break
