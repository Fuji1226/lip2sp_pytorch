import os
import numpy as np
from array import array
import librosa
import soundfile as sf

from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2

config = visqol_config_pb2.VisqolConfig()

reference = '/home/user/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/master/mfft_try_2/33/test_data/audio/pwg/F1/anger_e_006/gt.wav'
degraded = '/home/user/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/master/mfft_try_2/33/test_data/audio/pwg/F1/anger_e_006/generate.wav'

ref, sr_ref = sf.read(reference, dtype='float32')
deg, sr_deg = sf.read(degraded, dtype='float32')

# モノラル化
if ref.ndim > 1:
    ref = ref.mean(axis=1)
if deg.ndim > 1:
    deg = deg.mean(axis=1)

mode = "speech"  # "audio" or "speech"
if mode == "audio":
    config.audio.sample_rate = 48000
    config.options.use_speech_scoring = False
    svr_model_path = "libsvm_nu_svr_model.txt"
elif mode == "speech":
    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
else:
    raise ValueError(f"Unrecognized mode: {mode}")

config.options.svr_model_path = os.path.join(
    os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

target_sr = config.audio.sample_rate
if sr_ref != target_sr:
    ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=target_sr)
if sr_deg != target_sr:
    deg = librosa.resample(deg, orig_sr=sr_deg, target_sr=target_sr)

# numpy -> array('f')（連続メモリで型が float）
ref_arr = array('f', ref.astype(np.float32).tolist())
deg_arr = array('f', deg.astype(np.float32).tolist())

api = visqol_lib_py.VisqolApi()

api.Create(config)

similarity_result = api.Measure(ref_arr, deg_arr)

# 使われているフィールド名一覧
print("present fields:", [fd.name for fd, _ in similarity_result.ListFields()])

# 代表的フィールドの確認（存在すれば）
if hasattr(similarity_result, "moslqo"):
    print("moslqo:", similarity_result.moslqo)

# フレームレベル等の反復フィールドがあれば numpy で形状確認
if hasattr(similarity_result, "frame_level_scores") and len(similarity_result.frame_level_scores) > 0:
    import numpy as np
    arr = np.array(similarity_result.frame_level_scores)
    print("frame_level_scores shape:", arr.shape)

print(similarity_result.moslqo) #!よくわかんないので、一旦一つづつみたい