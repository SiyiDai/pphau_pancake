from os import listdir, makedirs
from os.path import isfile, join, isdir
import re
import shutil

GAP = 300
image_prefix = "left"
image_format = ".jpg"
save_rest_images = False

src_path = "/home/kolz14w/Proj_Master/haucode/proj/anno_yolo/pphau_group_b/sequences/CWbag"
sampled_dir = "/home/kolz14w/Proj_Master/haucode/proj/anno_yolo/pphau_group_b/sequences/CWbag/label_image"
rest_dir = "rest_image"

def downsample_image_files(image_path):
    sampled_names = []
    rest_names = []
    images = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    for image in images:
        regex = re.compile(r'\d+')
        index = regex.findall(image)
        if (int(index[0]) % GAP == 0):
            complete_sample_name = complete_name(index)
            sampled_names.append(complete_sample_name)
        else:
            complete_rest_name = complete_name(index)
            rest_names.append(complete_rest_name)
    
    print(sampled_names)
    
    return sampled_names, rest_names

def complete_name(index):
    return join(image_prefix + index[0] + image_format)

def save_images_to_dir(selected_images, src, dst):
    if not isdir(dst):
        makedirs(dst)
    for filename in selected_images:
        if filename.endswith('.jpg'):
            shutil.copy2(join(src, filename), dst)


if __name__ == "__main__":
    sampled_names, rest_names = downsample_image_files(src_path)
    save_images_to_dir(sampled_names, src_path, sampled_dir)
    if save_rest_images == True:
        save_images_to_dir(rest_names, src_path, rest_dir)