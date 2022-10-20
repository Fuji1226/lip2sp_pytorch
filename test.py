import numpy as np

path = '/home/usr1/q70261a/dataset/lip/np_files/lip_cropped/train/F01_kablab/ATR503_a01_0_mspec80.npz'

x = np.load(path)
breakpoint()
wav, (lip, feature, feat_add, upsample), data_len = x

breakpoint()