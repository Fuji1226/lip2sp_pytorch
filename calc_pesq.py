import torchaudio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


data_dir = '/home/usr1/q70261a/lip2sp_pytorch_all/lip2sp_920_re/result/default/generate/face/2022:12:20_16-17-47_transformer_face/mspec80_240/test_data/audio'
wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
stoi = ShortTimeObjectiveIntelligibility(16000, False)

for curdir, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            if "generate" in Path(file).stem:
                iter_cnt += 1
                print(f"\niter_cnt : {iter_cnt}")

                breakpoint()
                wav_gen, fs = torchaudio.load(os.path.join(curdir, file))
                wav_in, fs = torchaudio.load(os.path.join(curdir, "input.wav"))

                wav_gen = wav_gen.squeeze(0)
                wav_in = wav_in.squeeze(0)

                shorter_n_frame = int(min(wav_gen.shape[0], wav_in.shape[0]))
                wav_gen = wav_gen[:shorter_n_frame]
                wav_in = wav_in[:shorter_n_frame]
                assert wav_gen.shape[0] == wav_in.shape[0]

                # pesq, stoi
                p = wb_pesq(wav_gen, wav_in)
                s = stoi(wav_gen, wav_in)
                pesq_list.append(p)
                stoi_list.append(s)
                duration.append(shorter_n_frame / cfg.model.sampling_rate)
                print(f"PESQ = {p}")
                print(f"STOI = {s}")

                wav_gen = wav_gen.to("cpu").numpy()
                wav_in = wav_in.to("cpu").numpy()
                