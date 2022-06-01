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

        # reduction factor
        reduction_factor = 2,

        # ResNet3D parameter
        res_layers = 5,

        # transformer parameter
        n_layers = 6,
        d_model = 256,
        n_head = 8,
        d_k = 32,   # d_model // n_head
        d_v = 32,   # d_model // n_head
        d_inner = 1024,

        # Prenet & Postnet parameter
        pre_in_channels = 160,  # 音響特徴量のチャンネル数の2倍   
        pre_inner_channels = 32,
        post_inner_channels = 512,

        # out_channels
        out_channels = 80,  # 音響特徴量のチャンネル数（mspec:80, world:29）

        # "world" or "mspec"（音響特徴量の選択）
        feature_type="mspec",

        # glu parameter
        glu_inner_channels = 256,
        glu_layers = 4,

        # dropout
        dropout = 0.1,

        # global condition
        use_gc = False,

        # dataloader
        batch_size = 5,
        num_workers = 0,

        # optimizer
        lr = 0.001,
        betas = (0.9, 0.999),

        # training
        max_iter = 5,
        max_epoch = 5,

        # feature type setting.
        # input grayscale.（グレースケールかRGBか）
        gray=False,

        # input first and second derivative.（動的特徴量を使うかどうか）
        delta=True,
    )

    class HParams:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    
    hparams = HParams(hparams)
    return hparams


