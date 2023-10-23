from jiwer import wer
from dataset.phoneme_encode import classes2index_tts
import numpy as np
import torch

def get_keys_from_value(dict, val):
    """
    辞書に対してvalueからkeyを得る
    一度数値列に変換した音素列をもう一度音素列に変換するために使用
    """
    for k, v in dict.items():
        if v == val:
            return k

def save_data_lipreading(cfg, save_path, target, output, classes_index):
    """
    target : (T,)
    output : (C, T)
    """
    target = target.to("cpu").detach().numpy()
    output = output.to("cpu").detach().numpy()

    phoneme_answer = [get_keys_from_value(classes_index, i) for i in target]
    phoneme_answer = " ".join(phoneme_answer)

    # 予測結果にはeosが連続する場合があるので、除去する
    phoneme_predict = [get_keys_from_value(classes_index, i) for i in output]
    first_eos_index = 0
    for i in range(len(phoneme_predict)):
        if phoneme_predict[i] == "eos":
            first_eos_index = i + 1
            break
    phoneme_predict = phoneme_predict[:first_eos_index]
    phoneme_predict = " ".join(phoneme_predict)

    phoneme_error_rate = wer(phoneme_answer, phoneme_predict)

    with open(str(save_path / "phoneme.txt"), "a") as f:
        f.write("answer\n")
        f.write(f"{phoneme_answer}\n")
        f.write("\npredict\n")
        f.write(f"{phoneme_predict}\n")
        f.write(f"\nphoneme error rate = {phoneme_error_rate}\n")

    return phoneme_error_rate


def index_to_text(index: torch.tensor):
    output = index.to("cpu").detach().numpy()
    classes_index, _ = classes2index_tts()

    breakpoint()
    phoneme_answer = [get_keys_from_value(classes_index, i) for i in output]
    phoneme_answer = " ".join(phoneme_answer)

    breakpoint()
    return phoneme_answer

def output_to_text(output: torch.tensor):
    index = torch.argmax(output, dim=-1)
    
    phone = index_to_text(index)
    breakpoint()