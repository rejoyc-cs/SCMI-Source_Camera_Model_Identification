import os
import random
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

def center_crop(img, dim):
    height, width, _ = img.shape
    crop_height, crop_width = dim
    mid_x, mid_y = width // 2, height // 2
    half_cropH, half_cropW = crop_height // 2, crop_width // 2
    return img[mid_y - half_cropH:mid_y + half_cropH, mid_x - half_cropW:mid_x + half_cropW]

def isSaturated(img, alpha=0.7, beta=4, gamma=np.log(0.01)):
    h, w, c = img.shape
    img = img.astype(np.float32) / 255.0  # Ensure consistent scaling

    f = 0
    for i in range(c):
        img_channel = img[:, :, i]
        mean_c = np.mean(img_channel)
        std_c = np.std(img_channel)
        f_c = ((alpha * beta * (mean_c - mean_c ** 2)) + ((1 - alpha) * (1 - np.exp(gamma * std_c))))
        f += f_c

    return f / 3

def process_liu(base_dir, dest_dir, classes, subset, patch_size=64, T = 64,K = 16, M = 4):
    print("Started")
    with open('patch_extraction_process_status_liu.txt', 'a') as f:
        error = 0
        class_count = 0

        for cl in classes:
            class_count += 1
            img_dir = os.path.join(base_dir, subset, cl)
            patch_dir = os.path.join(dest_dir, subset, cl)
            os.makedirs(patch_dir, exist_ok=True)

            files = list(Path(img_dir).glob("*g")) + list(Path(img_dir).glob("*G"))  # Matches .jpg, .jpeg, etc.

            total_files_in_class = len(files)
            files_processed = 0

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
                quality_score = []
                Z_space = []

                for i in range(0, h, patch_size):
                    for j in range(0, w, patch_size):
                        img_patch = img_crop[i:i + patch_size, j:j + patch_size, :]
                        score = isSaturated(img_patch)
                        quality_score.append((score, img_patch, len(quality_score)))

                        mean_c = np.mean(img_patch)
                        std_c = np.std(img_patch)
                        zi = (mean_c, std_c)
                        Z_space.append((zi, img_patch, len(Z_space)))

                # Edge and feature based selection
                
                quality_score.sort(key=lambda x: x[0], reverse=True)
                E_patches = [(patch[1], patch[2]) for patch in quality_score]

                # Semantic based selection
                
                kmeans = KMeans(n_clusters=K, n_init='auto', max_iter=300, random_state=random_seed)
                kmeans.fit([zi for zi, _, _ in Z_space])

                cluster_centers = kmeans.cluster_centers_
                nearest_points = []

                for center in cluster_centers:
                    distances = cdist([center], [zi for zi, _, _ in Z_space])[0]
                    nearest_indices = np.argsort(distances)[:M]
                    nearest_points.extend([Z_space[i] for i in nearest_indices])

                S_patches = [(patch[1], patch[2]) for patch in nearest_points]

                # Excluding common E_patches
                E_final_patches = []
                seen_patches = set(patch[0].tobytes() for patch in S_patches)
                i = 0

                while len(E_final_patches) < T and i < len(E_patches):
                    patch_bytes = E_patches[i][0].tobytes()
                    if patch_bytes not in seen_patches:
                        E_final_patches.append(E_patches[i])
                    i += 1

                P_patches = S_patches + E_final_patches
                f.write(f"No. of E_patches: {len(E_final_patches)} | No. of S_patches: {len(S_patches)} | Total patches: {len(P_patches)}\n")

                for i in range(len(P_patches)):
                    filename = Path(file).stem+'_'+str(i+1)+'.jpg'
                    img_dir = Path(patch_dir) / filename
                    cv2.imwrite(str(img_dir), P_patches[i][0])

                files_processed += 1
                status_message = f"Class [{class_count}/{len(classes)}] || {subset} : ({files_processed}/{total_files_in_class})"
                print(status_message)
                f.write(status_message + "\n")
                f.write('-' * 80 + '\n')

        f.write(f"Total error: {error}\n")
