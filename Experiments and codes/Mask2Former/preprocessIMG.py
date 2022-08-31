from PIL import Image
from torchvision import transforms
import os

def absoluteFilePaths(directory):
    file_paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            file_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return file_paths

img_list = absoluteFilePaths('./pp2/images')
for img_path in img_list:
    img = Image.open(img_path)
    t1 = transforms.Resize(256)
    img = t1(img)
    t2 = transforms.CenterCrop(224)
    img = t2(img)
    img.save(img_path)
