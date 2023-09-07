import os
import time
import torch
import matplotlib.pyplot as plt
from librosa.display import specshow
import seaborn as sns

def check_attention_weight(att_w, filename, save_path):
    att_w = att_w.to('cpu').detach().numpy().copy()

    plt.figure()
    sns.heatmap(att_w, cmap="viridis", cbar=True)
    plt.title("attention weight")
    plt.xlabel("text")
    plt.ylabel("feature")

    plt.savefig(save_path + f"{filename}.png")
    plt.close()

def generate_for_FR_train_loss(cfg, model, train_loader, dataset, device, save_path, epoch, loss_f):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)

    process_times = []
    iter_cnt = 0
    
    epoch_output_loss = 0
    epoch_dec_output_loss = 0

    iter_cnt = 0
    for batch in train_loader:
        iter_cnt += 1
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        start_time = time.time()
        with torch.no_grad():
            output, dec_output, feat_add_out = model(lip)

        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=output.shape[-1]) 
        epoch_dec_output_loss += dec_output_loss.item()
        
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=output.shape[-1])
        epoch_output_loss += output_loss.item()

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        if iter_cnt >= 3:
            break
    
    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    
    return epoch_output_loss, epoch_dec_output_loss

def generate_for_train_check_taco(cfg, model, test_loader, dataset, device, save_path, epoch, mixing_prob):
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
            print('taco generate')
            output, dec_output, feat_add_out, att_w = model(lip, data_len=data_len)
            #tf_output, _, _ = model(lip=lip, prev=feature)

    
        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / 'synthesis' / f'epoch_{epoch}_FR'
        os.makedirs(str(_save_path), exist_ok=True)
        tmp_label = label[0]+'_FR'
        _save_path = _save_path / label[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        #os.makedirs(_save_path, exist_ok=True)
       
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
        
        
        att_path = _save_path / label[0]
        check_attention_weight(att_w[0], "attention", str(_save_path))

        with torch.no_grad():
            output, dec_output, feat_add_out, att_w = model(lip=lip, prev=feature, data_len=data_len, training_method='tf')


        _save_path = save_path / 'synthesis' / f'epoch_{epoch}_TF'
        os.makedirs(str(_save_path), exist_ok=True)
        _save_path = _save_path / label[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        #os.makedirs(_save_path, exist_ok=True)
       
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
        
        check_attention_weight(att_w[0], "attention", str(_save_path)+"_attention")

        with torch.no_grad():
            output, dec_output, feat_add_out, att_w = model(lip=lip, prev=feature, data_len=data_len, training_method='ss', mixing_prob=mixing_prob)


        _save_path = save_path / 'synthesis' / f'epoch_{epoch}_SS'
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
        
        check_attention_weight(att_w[0], "attention", str(_save_path)+"_attention")

        print('synthesis finished')
        break
    
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
            output, dec_output, feat_add_out, att = model(lip)
            #tf_output, _, _ = model(lip=lip, prev=feature)

    
        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / 'synthesis' / f'epoch_{epoch}_FR'
        os.makedirs(str(_save_path), exist_ok=True)
        tmp_label = label[0]+'_FR'
        _save_path = _save_path / label[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        #os.makedirs(_save_path, exist_ok=True)
       
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
        att_path = _save_path / label[0]
        check_attention_weight(att[0].squeeze(), "attention1", str(_save_path))
        check_attention_weight(att[1].squeeze(), "attention2", str(_save_path))
        
        with torch.no_grad():
            output, dec_output, feat_add_out, att = model(lip=lip, prev=feature, data_len=data_len, training_method='ss', mixing_prob=0.1)


        _save_path = save_path / 'synthesis' / f'epoch_{epoch}_SS'
        os.makedirs(str(_save_path), exist_ok=True)
        _save_path = _save_path / label[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        #os.makedirs(_save_path, exist_ok=True)
       
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
        
        att_path = _save_path / label[0]
        check_attention_weight(att[0].squeeze(), "attention1", str(_save_path))
        check_attention_weight(att[1].squeeze(), "attention2", str(_save_path))
        print('synthesis finished')
        break


def generate_for_train_check_emb(cfg, model, test_loader, dataset, device, save_path, epoch):
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
        wav, lip, feature, feat_add, upsample, data_len, speaker, label, emb = batch
        emb, feature, feat_add, data_len = emb.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        start_time = time.time()
        with torch.no_grad():
            output, dec_output, feat_add_out = model(emb)
            #tf_output, _, _ = model(lip=lip, prev=feature)

    
        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / 'synthesis' / f'epoch_{epoch}_FR'
        os.makedirs(str(_save_path), exist_ok=True)
        tmp_label = label[0]+'_FR'
        _save_path = _save_path / label[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        #os.makedirs(_save_path, exist_ok=True)
       
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

        with torch.no_grad():
            output, dec_output, feat_add_out = model(emb=emb, prev=feature, data_len=data_len, training_method='ss', mixing_prob=0.1)


        _save_path = save_path / 'synthesis' / f'epoch_{epoch}_SS'
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

def generate_for_train_check_nar(cfg, model, test_loader, dataset, device, save_path, epoch):
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
            output, feat_add_out, phoneme = model(lip)
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
        
        plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)
        ax = plt.subplot(2, 1, 1)
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
        ax = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
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

        plt.savefig(str(_save_path))
        plt.close()
        print('synthesis finished')
        break