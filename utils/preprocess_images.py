import os, shutil
from utils.file_utils import clear_folder
from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer

def preprocess_images(impaths, imnames):
    if impaths is None or imnames is None:
        return None

    clear_folder("temp")
    upload = os.path.join("temp", "comp", "uploaded", "test", "images")
    os.makedirs(upload)
    for i in range(len(impaths)):
        impath = impaths[i]
        imname = os.path.splitext(imnames[i])[0]
        bioformats_convert_to_tif(impath, os.path.join(upload, imname + ".tif"))

def bioformats_convert_to_tif(impath, outpath):
    img = AICSImage(impath)
    img_data = img.data[0,0,0,:,:]
    ome_tiff_writer.OmeTiffWriter.save(img_data, outpath)
