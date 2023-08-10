import pandas as pd
import matplotlib.pyplot as plt


def main():
    N_DATA_USE = 200
    main_df = pd.read_csv("/home/usr4/r70264c/lip2sp_pytorch/data_process/lrs2/duration_check/main.csv")
    pretrain_df = pd.read_csv("/home/usr4/r70264c/lip2sp_pytorch/data_process/lrs2/duration_check/pretrain.csv")    
    
    train_data_df = main_df
    train_data_id_used = train_data_df["id"].value_counts().nlargest(N_DATA_USE).index.values
    train_data_df = train_data_df.loc[train_data_df["id"].isin(train_data_id_used)]
    breakpoint()
    
    train_data_df = pretrain_df
    train_data_id_used = train_data_df["id"].value_counts().nlargest(N_DATA_USE).index.values
    train_data_df = train_data_df.loc[train_data_df["id"].isin(train_data_id_used)]
    breakpoint()

    
if __name__ == "__main__":
    main()