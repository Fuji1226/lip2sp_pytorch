from curses import window


def create_hparams():
    hparams = dict(
        # video parameter
        fps = 50,
        video_channels = 3,

        # audio parameter
        sampling_rate = 16000,
        n_fft = 1024,
        hop_length = 160,
        win_length = 640,
        f_min = 0,
        f_max = 7600,
        n_mel_channels = 80,

        # frame period
        frame_period=10,

        # acoutic feature frames in training
        length=300,

        # dataloader
        batch_size = 2,
        num_workers = 0,

        # optimizer
        lr = 0.001,
        betas = (0.9, 0.999),

        # training
        max_iter = 5,

        # feature type setting.
        # input grayscale.（グレースケールかRGBか）
        gray=False,

        # input first and second derivative.（動的特徴量を使うかどうか）
        delta=True,

        # "world" or "mspec"（音響特徴量の選択）
        feature_type="mspec",
    )

    class HParams:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    
    hparams = HParams(hparams)
    return hparams


# class HParams:
#     def __init__(self, dictionary):
#         for k, v in dictionary.items():
#             setattr(self, k, v)


# if __name__ == "__main__":
#     hparams = dict(

#         #########################
#         # video Parameters      #
#         #########################
#         fps = 50,

#         #########################
#         # Audio Parameters      #
#         #########################
#         sampling_rate = 16000,
#         n_fft = 1024,
#         hop_length = 256,
#         win_length = 1024,
#         f_min = 0,
#         f_max = 7600,
#         n_mel_channels = 80,
#     )

#     print(hparams)

#     parameter = HParams(hparams)
#     print(parameter.fps)
#     print(parameter.sampling_rate)