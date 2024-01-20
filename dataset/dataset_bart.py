import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import omegaconf
import pandas as pd
import random
from data_process.phoneme_encode import classes2index_tts


class DatasetBART(Dataset):
    def __init__(
        self,
        data_path_list: List[Path],
        cfg,
        transform,
    ):
        super().__init__()
        self.data_path_list = data_path_list
        self.cfg = cfg
        self.transform = transform
        self.class_to_id, self.id_to_class = classes2index_tts(cfg)
        
    def __len__(self) -> int:
        return len(self.data_path_list)
        
    def __getitem__(
        self,
        index,
    ):
        data_path = self.data_path_list[index]
        df = pd.read_csv(str(data_path))
        phoneme_target, phoneme_masked, phoneme_len = self.transform(df, data_path, self.class_to_id)
        return (
            phoneme_target,
            phoneme_masked,
            phoneme_len,
        )
        
        
class TransformBART:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test
        self.phoneme_list = [
            "A",
            "E",
            "I",
            "N",
            "O",
            "U",
            "a",
            "b",
            "by",
            "ch",
            "cl",
            "d",
            "dy",
            "e",
            "f",
            "g",
            "gy",
            "h",
            "hy",
            "i",
            "j",
            "k",
            "ky",
            "m",
            "my",
            "n",
            "ny",
            "o",
            "p",
            "py",
            "r",
            "ry",
            "s",
            "sh",
            "t",
            "ts",
            "ty",
            "u",
            "v",
            "w",
            "y",
            "z",
            'kw',
            "pau",
            "sil",
        ]
    
    def __call__(self, df, data_path, class_to_id):
        df['phoneme'] = df['phoneme'].str.replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
        phoneme_target = []
        phoneme_masked = []
        
        phoneme_len_list = []
        for p in df['phoneme'].values:
            p = p.replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
            p = p.split(',')
            phoneme_len_list.append(len(p))

        start_index = 0
        if sum(phoneme_len_list) > (self.cfg.model.bart.phoneme_max_len - 2):
            start_index += random.randint(0, sum(phoneme_len_list) - (self.cfg.model.bart.phoneme_max_len - 2))
            
        cnt = 0
        for i, row in df.iterrows():
            p = row['phoneme']
            p_len = phoneme_len_list[i]
            if cnt < start_index:
                cnt += p_len
                continue
            if cnt + p_len > start_index + (self.cfg.model.bart.phoneme_max_len - 2):
                continue
            cnt += p_len
            
            p = p.replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
            p = p.split(',')
            if len(p) == 1 and p[0] == '':
                continue
            
            if self.train_val_test == 'train':
                mask_flag = random.randint(0, 99) < self.cfg.model.bart.masking_rate
                phoneme_target += p
                if mask_flag:
                    for j in range(len(p)):
                        mask_method_flag = random.randint(0, 99)
                        if mask_method_flag < 80:
                            phoneme_masked += ['mask']
                        elif mask_method_flag < 90:
                            phoneme_masked += random.sample(self.phoneme_list, 1)
                        else:
                            phoneme_masked += [p[j]]
                else:
                    phoneme_masked += p
            else:
                phoneme_target += p
                phoneme_masked += p
                
        phoneme_target.insert(0, '^')
        phoneme_target.append('$')
        phoneme_masked.insert(0, '^')
        phoneme_masked.append('$')
        
        phoneme_target = [class_to_id[p] for p in phoneme_target]
        phoneme_masked = [class_to_id[p] for p in phoneme_masked]
        
        if len(phoneme_target) != len(phoneme_masked):
            raise ValueError(f'phoneme_target_len and phoneme_masked_len should be equal. {data_path}')
        if len(phoneme_target) > self.cfg.model.bart.phoneme_max_len or len(phoneme_masked) > self.cfg.model.bart.phoneme_max_len:
            raise ValueError(f'phoneme_target_len and phoneme_masked_len should be smaller than self.cfg.model.bart.phoneme_max_len. {data_path}')

        phoneme_target = torch.tensor(phoneme_target, dtype=torch.long)
        phoneme_masked = torch.tensor(phoneme_masked, dtype=torch.long)
        phoneme_len = phoneme_target.shape[0]
        return (
            phoneme_target,
            phoneme_masked,
            phoneme_len,
        )
        
        
def collate_time_adjust_bart(batch, cfg):
    phoneme_target_list, phoneme_masked_list, phoneme_len_list = list(zip(*batch))
    phoneme_target_pad_list = []
    phoneme_masked_pad_list = []
    for phoneme_target, phoneme_masked in zip(phoneme_target_list, phoneme_masked_list):
        if len(phoneme_target) > cfg.model.bart.phoneme_max_len:
            # start_idx = random.randint(0, len(phoneme_target) - cfg.model.bart.phoneme_max_len)
            # phoneme_target = phoneme_target[start_idx:start_idx + cfg.model.bart.phoneme_max_len]
            # phoneme_masked = phoneme_masked[start_idx:start_idx + cfg.model.bart.phoneme_max_len]
            pass
        else:
            padding_len = cfg.model.bart.phoneme_max_len - len(phoneme_target)
            phoneme_target = torch.nn.functional.pad(phoneme_target, (0, padding_len), value=0)
            phoneme_masked = torch.nn.functional.pad(phoneme_masked, (0, padding_len), value=0)
        phoneme_target_pad_list.append(phoneme_target)
        phoneme_masked_pad_list.append(phoneme_masked)
    phoneme_target = torch.stack(phoneme_target_pad_list, dim=0)
    phoneme_masked = torch.stack(phoneme_masked_pad_list, dim=0)
    phoneme_len = torch.tensor(phoneme_len_list)
    return (
        phoneme_target,
        phoneme_masked,
        phoneme_len,
    )