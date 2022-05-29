"""
lip2sp/utils.pyと同じです
"""

def prime_factorize(n):
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1:
        a.append(n)
    return tuple(sorted(a))


def get_upsample(fps, fs, frame_period):
    nframes = 1000 // frame_period      # frame_period = 10
    upsample = nframes // fps       # fps = 50
    return int(upsample)    # 2


def get_state_name(feature_type, frame_period, dim):
    return feature_type + "_fp" + str(frame_period) + "_dim" + str(dim)


def get_sp_name(name, feature_type, frame_period, nmels=None):
    ret = name + "_" + feature_type + "_fp" + str(frame_period)

    if feature_type == "mspec":
        ret += "_dim" + str(nmels)

    return ret

if __name__ == "__main__":
    fps = 50
    fs = 16000
    frame_period = 10
    upsample = get_upsample(fps, fs, frame_period)
    print(upsample)