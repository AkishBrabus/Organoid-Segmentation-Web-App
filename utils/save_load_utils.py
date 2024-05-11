import shutil
import os
from utils.file_utils import clear_folder

def save_experiment(fname):
    if fname == None: 
        return None
    save_folder = os.path.join("temp", "save")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        clear_folder(save_folder)
    shutil.make_archive(os.path.join(save_folder, fname), 'zip', os.path.join("temp", "comp"))
    return os.path.join("temp", "save", fname + ".zip")

def load_experiment(fpath):
    if fpath == None:
        return None
    clear_folder("temp")
    shutil.unpack_archive(fpath, os.path.join("temp", "comp"), "zip")
    return os.path.join("temp", "comp")