import os
import time
import torch
import matplotlib.pyplot as plt
from librosa.display import specshow

def generate_for_train_check(cfg, model, test_loader, dataset, device, save_path, epoch):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)

    process_times = []

    iter_cnt = 0
    for batch in test_loader:
        wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        start_time = time.time()
        with torch.no_grad():
            output, dec_output, feat_add_out = model(lip)
            #tf_output, _, _ = model(lip=lip, prev=feature)

    
        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / 'synthesis' / f'epoch_{epoch}'
        os.makedirs(str(_save_path), exist_ok=True)
        _save_path = _save_path / label[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        #os.makedirs(_save_path, exist_ok=True)
       
        feature = feature[0].to('cpu').detach().numpy().copy()
        output = output[0].to('cpu').detach().numpy().copy()
        dec_output = dec_output[0].to('cpu').detach().numpy().copy()

        plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)
        ax = plt.subplot(3, 1, 1)
        specshow(
        data=feature, 
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
        data=output, 
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
        data=dec_output, 
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
        print('synthesis finished')
        break