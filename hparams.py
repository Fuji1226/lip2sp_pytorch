from curses import window


def create_hparams():
    hparams = dict(
        #####################
        # data info
        #####################
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

        # input grayscale.（グレースケールかRGBか）
        gray = False,

        # input first and second derivative.（動的特徴量を使うかどうか）
        delta = True,

        #####################
        # training settings     
        #####################
        # frame period
        frame_period = 10,

        # acoutic feature frames in training
        length = 300,     # Unet-discriminatorを使用するときは402か202で！

        # reduction factor
        reduction_factor = 2,

        # ResNet3D parameter
        res_layers = 5,

         # glu parameter
        glu_inner_channels = 256,
        glu_layers = 4,

        # transformer parameter
        n_layers = 6,
        d_model = 256,
        n_head = 8,

        # 使用するencoder（"transformer" or "conformer"）
        which_encoder = "conformer",

        # 使用するdecoder（"transformer" or "glu"）
        # gluはできてないので、transformerで
        which_decoder = "transformer",

        # discriminator（"unet" or "jcu"）
        # 使わない場合はNone
        which_d = "jcu",

        # "world" or "mspec"（音響特徴量の選択）
        feature_type = "mspec",

        # out_channels
        out_channels = None,  # 音響特徴量から計算します

        # Prenet & Postnet parameter
        pre_in_channels = None,  # 音響特徴量から計算します
        pre_inner_channels = 32,
        post_inner_channels = 512,

        # dropout
        dropout = 0.1,

        # global condition
        use_gc = False,

        # 学習方法（"tf" or "ss"）
        # tf : teacher forcing
        # ss : scheduled sampling
        training_method = "ss",

        # parameter for scheduled sampling
        num_passes = 1,     # 繰り返し回数
        mixing_prob = 0.5,  # 混合率

        # dataloader
        batch_size = 5,
        num_workers = 0,

        # optimizer
        lr = 0.001,
        betas = (0.9, 0.999),

        # training
        max_iter = 5,
        max_epoch = 10,
    )

    class HParams:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    
    hparams = HParams(hparams)

    # 音響特徴量からout_channels, pre_in_channelsを設定
    if hparams.feature_type == "mspec":
        hparams.out_channels = 80
        hparams.pre_in_channels = hparams.out_channels * 2

    elif hparams.feature_type == "world":
        hparams.out_channels = 29
        hparams.pre_in_channels = hparams.out_channels * 2

    # Unet-discrimintorを使用する際のフレーム数の確認
    if hparams.which_d == "unet":
        assert hparams.length == 202 or 402, "Unet-discriminatorを使うときは、lengthを202か402にしてください!"
    
    return hparams


if __name__ == "__main__":
    hparams = create_hparams()
    print(hparams.feature_type)
    print(hparams.out_channels)
    print(hparams.pre_in_channels)