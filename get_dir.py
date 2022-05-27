import os
from pathlib import Path


def get_datasetroot():
    ret = Path("~", "dataset")

    ret = ret.expanduser()

    ret.mkdir(exist_ok=True, parents=True)

    return ret

def get_data_directory():
    """return SAVEDIR"""
    return os.path.join(get_datasetroot(), "lip", )