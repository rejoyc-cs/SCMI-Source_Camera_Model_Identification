import os
import random
import numpy as np
import cv2
from pathlib import Path

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

def center_crop(img, dim):
    height, width, _ = img.shape
    crop_height, crop_width = dim
    mid_x, mid_y = width // 2, height // 2
    half_cropH, half_cropW = crop_height // 2, crop_width // 2
    return img[mid_y - half_cropH:mid_y + half_cropH, mid_x - half_cropW:mid_x + half_cropW]

def qaulityScore(img, alpha=0.7, beta=4, gamma=np.log(0.01)):
    h,w,c = img.shape
    img = img/255 #Normalizing the image
    f=0
    
    #For each channel calculating the f_score
    for i in range(c):
        img_channel = img[:,:,i]
        mean_c = np.mean(img_channel)
        mean_c = mean_c / 255
        std_c = np.std(img_channel)
        std_c = std_c/255
        f_c = ((alpha * beta * (mean_c - (mean_c**2))) + ((1- alpha)*(1-np.exp(gamma * std_c)))) #f-score
        f = f+f_c #Adding the f-score
        
    f = f/3
    return f


def process_rana(base_dir, dest_dir, classes, subset, patch_size=64, total_patches=256):
    print("Started")
    with open('patch_extraction_process_rana.txt', 'a') as f:
        error = 0
        class_count = 0
        
        for cl in classes:
            class_count += 1
            img_dir = os.path.join(base_dir, subset, cl)
            patch_dir = os.path.join(dest_dir, subset, cl)
            os.makedirs(patch_dir, exist_ok=True)

            files= list(Path(img_dir).glob("*g")) + list(Path(img_dir).glob("*G")) # Matches .jpg, .jpeg, etc.

            total_files_in_class = len(files)
            files_processed = 0
            print(total_files_in_class)

            for file in files:
                img = cv2.imread(str(file))
                if img is None:
                    f.write(f"Not an image: {file}\n")
                    error += 1
                    continue

                height, width, _ = img.shape
                f.write(f"Original image dim: {img.shape}\n")

                # Crop the image
                crop_h = (height // patch_size) * patch_size
                crop_w = (width // patch_size) * patch_size
                img_crop = center_crop(img, (crop_h, crop_w))
                f.write(f"Crop image dim: {img_crop.shape}\n")

                h, w, _ = img_crop.shape
                patches = []
                    

                for i in range(0, h, patch_size):
                    for j in range(0, w, patch_size):
                        img_patch = img_crop[i:i + patch_size, j:j + patch_size, :]
                        score = qaulityScore(img_patch)
                        patches.append((score, img_patch))
                        
                        
                patches = sorted(patches, key=lambda x: x[0], reverse=True)[:total_patches]

                        
                for i in range(len(patches)):
                    filename = Path(file).stem+'_'+str(i+1)+'.jpg'
                    img_dir = Path(patch_dir) / filename
                    cv2.imwrite(str(img_dir), patches[i][1])

                files_processed += 1
                status_message = f"Class [{class_count}/{len(classes)}] || {subset} : ({files_processed}/{total_files_in_class})"
                print(status_message)
                f.write(status_message + "\n")
                f.write('-' * 80 + '\n')

        f.write(f"Total error: {error}\n")
