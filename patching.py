import os
import argparse

import patch.patch_liu as pl
import patch.patch_huan as ph
import patch.patch_bennabhaktula as pb
import patch.patch_sychandran as ps
import patch.patch_rana as pr
import patch.patch_rafi as prf

def find_classes(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        print(f"Warning: Directory '{path}' does not exist or is empty.")
        return [], {}

    classes = sorted(os.listdir(path))
    class_to_idx = {cl: i for i, cl in enumerate(classes)}
    return classes, class_to_idx

def create_directories(dest_path, classes, subset):
    for cl in classes:
        patch_dir = os.path.join(dest_path, subset, cl)
        os.makedirs(patch_dir, exist_ok=True)

def main(src_path, dest_path, method):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)

    categories = ['test','train', 'val']
    for category in categories:
        classes, _ = find_classes(os.path.join(src_path, category))
        if not classes:
            print(f"Skipping '{category}' as no classes were found.")
            continue

        create_directories(dest_path, classes, category)

        method = method.lower()
        if method == 'liu':
            pl.process_liu(src_path, dest_path, classes, category)
        elif method == 'huan':
            ph.process_huan(src_path, dest_path, classes, category)
        elif method == 'bennabhaktula':
            pb.process_benna(src_path, dest_path, classes, category)
        elif method == 'sychandran':
            ps.process_sychan(src_path, dest_path, classes, category)
        elif method == 'rana':
            pr.process_rana(src_path, dest_path, classes, category)
        elif method == 'rafi':
            prf.process_rafi(src_path, dest_path, classes, category)
        
            
        
        else:
            print(f"Method '{method}' is not implemented yet!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provide source and destination paths")

    parser.add_argument(
        "--src", 
        type=str, 
        default="./data", 
        help="Source directory path (default: ./data)"
    )

    parser.add_argument(
        "--dest", 
        type=str, 
        default="./patches", 
        help="Destination directory path (default: ./)"
    )
    
    parser.add_argument(
        "--method", 
        type=str, 
        default="liu",  # Changed default to lowercase
        help="Select patching method: liu, huan, sychandran, bennabhaktula, rafi, rana"
    )

    args = parser.parse_args()
    main(args.src, args.dest, args.method)
