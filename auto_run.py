import subprocess


def main():
    command_list = [
        ['python', '/home/minami/lip2sp_pytorch/result/calc.py'],
        ['python', '/home/minami/lip2sp_pytorch/shells/run.py'],
    ]
    for command in command_list:
        subprocess.run(command)


if __name__ == '__main__':
    main()