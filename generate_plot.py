import os
import matplotlib.pyplot as plt
from librosa.display import specshow

def save_target_pred(target, pred_output, pred_dec_output, save_path):

    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)
    ax = plt.subplot(3, 1, 1)
    specshow(
    data=target, 
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
    data=pred_output, 
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
    data=pred_dec_output, 
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

    plt.savefig(str(save_path))
    plt.clf()
    plt.close()
