import pandas as pd
import matplotlib.pyplot as plt


def main():
    main_df = pd.read_csv("/home/usr4/r70264c/lip2sp_pytorch/data_process/lrs2/duration_check/main.csv")
    pretrain_df = pd.read_csv("/home/usr4/r70264c/lip2sp_pytorch/data_process/lrs2/duration_check/pretrain.csv")    
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].hist(main_df["duration"].values, bins=20, alpha=0.5, label="main")
    ax[0].hist(pretrain_df["duration"].values, bins=20, alpha=0.5, label="pretrain")
    ax[0].grid()
    ax[0].legend()
    
    ax[1].boxplot((main_df["duration"].values, pretrain_df["duration"].values))
    ax[1].set_xticklabels(["main", "pretrain"])
    ax[1].grid()
    
    fig.tight_layout()
    fig.savefig("duration.png")
    plt.close()
    
    breakpoint()

    
if __name__ == "__main__":
    main()