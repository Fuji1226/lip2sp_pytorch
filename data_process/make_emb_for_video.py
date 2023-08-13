"""
事前学習モデルを用いた動画embedding
"""

import timm
from torch import nn
import numpy as np
import torch

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
import glob
import os

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#path = '/mnt/diskA/naoaki/dataset/lip/np_files_224/lip_cropped/train/F01_kabulab/BASIC5000_2022_0_mspec80.npz'

path_list = glob.glob('/home/usr1/q70261a/dataset/lip/np_files_224/lip_cropped/train/F01_kabulab/*.npz')


class EfficientNet_V2(nn.Module):
    def __init__(self):
        super(EfficientNet_V2, self).__init__()
        #モデルの定義
        model_name = "tf_efficientnet_b0_ns"
        self.effnet = timm.create_model(model_name, pretrained=True)
        #最終層の再定義
        #self.effnet.classifier = nn.Linear(self.effnet.conv_head.out_channels, n_out)

    def forward(self, x):
        return self.effnet(x)
    
device = 'cuda'

model = EfficientNet_V2().to(device)
model.eval()

# for i in range(lip.shape[0]):
#     tmp = lip[i]
#     grid = make_grid(tmp)
#     show(grid)
#     plt.savefig(f'tmp/test2/test_{i}.png')
#     plt.close()

for path in path_list:
    print(path)
    save_path = path.replace('F01_kabulab', 'emb_eval/F01_kabulab')
    if os.path.isfile(save_path):
        continue
    
    # if 'ATR503_e28_0_mspec80.npz' in path or 'BASIC5000_0296_0_mspec80' in path or 'BASIC5000_0559_0_mspec80' in path or 'BASIC5000_0421_0_mspec80.npz' in path or 'BASIC5000_0583_0_mspec80' in path:
    #     continue
    # if 'BASIC5000_0609_0_mspec80' in path or 'BASIC5000_0652_0_mspec80' in path or 'BASIC5000_0748_0_mspec80' in path or 'BASIC5000_0805_0_mspec80' in path:
    #     continue
    
    # if 'BASIC5000_0876_0_mspec80' in path or 'BASIC5000_0982_0_mspec80' in path or 'BASIC5000_0984_0_mspec80' in path or 'BASIC5000_1109_0_mspec80' in path or 'BASIC5000_1333_0_mspec80' in path or 'BASIC5000_1219_0_mspec80' in path:
    #     continue
    
    # if 'BASIC5000_1354_0_mspec80' in path or 'BASIC5000_1370_0_mspec80' in path or 'BASIC5000_1522_0_mspec80' in path:
    #     continue
    emb_list = []
    npz_key = np.load(str(path))
    lip = torch.from_numpy(npz_key['lip'])
    lip = lip.permute(-1, 0, 1, 2).to(torch.float32)
    for i in range(lip.shape[0]):
        tmp = lip[i].to(device).unsqueeze(0)
        emb = model(tmp)
        
        emb = emb.to('cpu').detach()
        emb_list.append(emb)
    
    test = torch.cat(emb_list, dim=0).numpy().copy()
    print(test.shape)
    
    print(f'save {save_path}')
    #breakpoint()
    np.savez(
        save_path,
        emb = test
    )

print()
print('finnish')