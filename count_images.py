
import os
import glob

base_dir = os.getcwd()
subdirectories = [f.path for f in os.scandir(base_dir) if f.is_dir()]

extensions = ['*.bmp', '*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG']

for d in subdirectories:
    count = 0
    for ext in extensions:
        count += len(glob.glob(os.path.join(d, ext)))
    print(f"Folder: {os.path.basename(d)}, Images: {count}")
