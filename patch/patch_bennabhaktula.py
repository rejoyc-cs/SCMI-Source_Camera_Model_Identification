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

def patchQuality(img):
    img = img / 255
    
    std_c_1 = np.std(img[:,:,0])
    std_c_2 = np.std(img[:,:,1])
    std_c_3 = np.std(img[:,:,2])
    mean_std = np.mean([std_c_1, std_c_2, std_c_3])
    
    if((std_c_1 >= 0.005 and std_c_2 >= 0.005 and std_c_3 >= 0.005) and  (std_c_1 <= 0.02 and std_c_2 <= 0.02 and std_c_3 <= 0.02)):
        return (1,mean_std) #Homogeneous
    elif (std_c_1 < 0.005 and std_c_2 < 0.005 and std_c_3 < 0.005 ):
        return (2,mean_std)  #Saturated
    else:
        return (3,mean_std) #Non Homogeneous

def normalize_img(img_lbp):
    img_lbp = img_lbp.astype(np.float64)
    
    # Ensure the image has a valid range
    max_val = np.max(img_lbp)
    if max_val > 0:
        img_lbp = 255.0 * (img_lbp - np.min(img_lbp)) / max_val
    img_lbp = np.clip(img_lbp, 0, 255)
    img_lbp = img_lbp.astype(np.uint8)
    
    return img_lbp

def process_benna(base_dir, dest_dir, classes, subset, patch_size=128, stride_rate=0.25, total_patches=400):
    print("Started")
    with open('patch_extraction_process_status_benna.txt', 'a') as f:
        error = 0
        class_count = 0
        stride=int(patch_size*stride_rate)

        for cl in classes:
            class_count += 1
            img_dir = os.path.join(base_dir, subset, cl)
            patch_dir = os.path.join(dest_dir, subset, cl)
            os.makedirs(patch_dir, exist_ok=True)

            files = list(Path(img_dir).glob("*g")) + list(Path(img_dir).glob("*G"))  # Matches .jpg, .jpeg, etc.

            total_files_in_class = len(files)
            files_processed = 0
            
            homo_patch = []
            nonHomo_patch = []
            satur_patch = []

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
                img_lbp = np.zeros((patch_size,patch_size,3), np.uint8)

                i = 0
                k = 0

                p1 = 0
                p2 = 0
                p3 = 0
                while ((i+patch_size) <= h):
                    j=0
                    while ((j+patch_size)<= w):
                        img_lbp = img_crop[i:i + patch_size, j:j + patch_size, :]
                        #img_lbp = normalize_img(img_lbp)
                        patch_check = patchQuality(img_lbp)
                        
                        #Splitting homogeneous, non-homogeneous, and saturated patches
                        if patch_check[0]==1:
                            p1 = p1+1
                            patch_data = (img_lbp, patch_check[1])
                            homo_patch.append(patch_data)
                        elif patch_check[0]==2:
                            p2 = p2+1
                            patch_data = (img_lbp,patch_check[1])
                            satur_patch.append(patch_data)
                        else:
                            p3 = p3+1
                            patch_data = (img_lbp,patch_check[1])
                            nonHomo_patch.append(patch_data)


                        j=j+stride
                    i=i+stride
                
                ####  Finalizing the files #####
                
                
                print("Total Homo patch extracted: ", p1, " Size of array: ", len(homo_patch), file=f)
                if len(homo_patch) > total_patches:
                    selected_indices = np.random.choice(len(homo_patch), size=total_patches, replace=False)
                    homo_patch = [homo_patch[idx] for idx in selected_indices]  
                k=len(homo_patch)
                print("Total Homo patch extracted: ", p1, " Selected Size of array: ",len(homo_patch), " K: ",k, "Remaining: ",(total_patches-k), file=f)


                print("Total Saturated patch extracted: ", p2, " Size of array: ",len(satur_patch), file=f)
                if k < total_patches:
                    satur_patch = sorted(satur_patch, key=lambda x: x[1])[:total_patches-k]
                else:
                    satur_patch = []
                k = k+len(satur_patch)
                print("Total Saturated patch extracted: ", p2, " Selected Size of array: ",len(satur_patch)," K: ",k,"Remaining: ",(total_patches-k), file=f)


                print("Total NonHomogeneous patch extracted: ", p3, " Size of array: ",len(nonHomo_patch), file=f)
                if k < total_patches:
                    nonHomo_patch = sorted(nonHomo_patch, key=lambda x: x[1])[:total_patches-k]
                else:
                    nonHomo_patch = []
                k = k+len(nonHomo_patch)
                print("Total NonHomogeneros patch extracted: ", p3, " Selected Size of array: ",len(nonHomo_patch)," K: ",k,"Remaining: ",(total_patches-k), file=f)

                
                ####  SAVING FILES ####
                i=1
                for patch in homo_patch:
                    filename = Path(file).stem+'_'+str(i)+'.jpg'
                    img_dir = Path(patch_dir) / filename
                    cv2.imwrite(img_dir, patch[0])
                    i+=1

                for patch in satur_patch:
                    filename = Path(file).stem+'_'+str(i)+'.jpg'
                    img_dir = Path(patch_dir) / filename
                    cv2.imwrite(img_dir, patch[0])
                    i+=1

                for patch in nonHomo_patch:
                    filename = Path(file).stem+'_'+str(i)+'.jpg'
                    img_dir = Path(patch_dir) / filename
                    cv2.imwrite(img_dir, patch[0])
                    i+=1

                files_processed += 1
                status_message = f"Class [{class_count}/{len(classes)}] || {subset} : ({files_processed}/{total_files_in_class})"
                print(status_message)
                f.write(status_message + "\n")
                f.write('-' * 80 + '\n')

        f.write(f"Total error: {error}\n")
