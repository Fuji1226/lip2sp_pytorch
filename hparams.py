from curses import window


def create_hparams():
    hparams = dict(
        # video parameter
        fps = 50,

        # audio parameter
        sampling_rate = 16000,
        n_fft = 1024,
        hop_length = 256,
        win_length = 1024,
        f_min = 0,
        f_max = 7600,
        n_mel_channels = 80,
    )

    class HParams:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    
    hparams = HParams(hparams)
    return hparams


class HParams:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


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