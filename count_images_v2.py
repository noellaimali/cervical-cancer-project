
import os
import glob

base_dir = os.getcwd()
target_classes = ['normal cells', 'abnormal cells']

for class_name in target_classes:
    folder_path = os.path.join(base_dir, class_name)
    if os.path.exists(folder_path):
        cropped_path = os.path.join(folder_path, 'CROPPED')
        search_path = cropped_path if os.path.exists(cropped_path) else folder_path
        
        count = 0
        extensions = ['*.bmp', '*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG']
        for ext in extensions:
            files = glob.glob(os.path.join(search_path, ext))
            for f in files:
                if os.path.getsize(f) < 500 * 1024:
                    count += 1
        print(f"Class: {class_name}, Path: {search_path}, Images: {count}")
