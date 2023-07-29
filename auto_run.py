import subprocess


def main():
    command_list = [
        ['python', 'train_nar.py', 'train.module_is_fixed=["spk_emb_layer"]'],
        ['python', 'train_nar.py', 'train.module_is_fixed=["spk_emb_layer", "ResNet_GAP"]'],
        ['python', 'train_nar.py', 'train.module_is_fixed=["spk_emb_layer", "encoder"]'],
        ['python', 'train_nar.py', 'train.module_is_fixed=["spk_emb_layer", "decoder"]'],
        ['python', 'train_nar.py', 'train.module_is_fixed=["spk_emb_layer", "ResNet_GAP", "encoder"]'],
        ['python', 'train_nar.py', 'train.module_is_fixed=["spk_emb_layer", "ResNet_GAP", "decoder"]'],
        ['python', 'train_nar.py', 'train.module_is_fixed=["spk_emb_layer", "encoder", "decoder"]'],
    ]
    for command in command_list:
        subprocess.run(command)


if __name__ == '__main__':
    main()