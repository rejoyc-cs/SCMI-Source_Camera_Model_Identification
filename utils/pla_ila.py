import os
import torch
import os.path
from collections import Counter
import pandas as pd
from collections import defaultdict


random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)  # Only for single GPU


def patch_level_accuracy(test_loader, model, result_folder='./Results',dual=False):
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(result_folder, exist_ok=True)
    
    class_data = defaultdict(list)
    total_correct = 0
    total_samples = 0
    progress_interval = max(len(test_loader) // 10, 1)  # Print progress every 10% of batches
    
    if dual:
        with torch.no_grad():
            for i, (imgs1,imgs2, targets, filenames) in enumerate(test_loader):
                imgs_org,mat_img, targets = imgs1.to(device),imgs2.to(device), targets.to(device)
                output = model(imgs_org,mat_img)
                predicted_class = torch.max(output, 1)[1]
                target_class = torch.max(targets, 1)[1]
                
                for j in range(imgs_org.size(0)):
                    filename = filenames[j]
                    entry = {
                        'Filename': filename,
                        'Target Class': target_class[j].item(),
                        'Predicted Class': predicted_class[j].item()
                    }
                    class_data[target_class[j].item()].append(entry)

                    # Update accuracy tracking
                    if target_class[j].item() == predicted_class[j].item():
                        total_correct += 1
                    total_samples += 1
                
                # Show progress at 10% intervals
                if (i + 1) % progress_interval == 0 or i == len(test_loader) - 1:
                    print(f"Progress Done (PLA): {((i + 1) / len(test_loader)) * 100:.1f}%")
    else:
        with torch.no_grad():
            for i, (images, targets, filenames) in enumerate(test_loader):
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                predicted_class = torch.max(output, 1)[1]
                target_class = torch.max(targets, 1)[1]
                
                for j in range(images.size(0)):
                    filename = filenames[j]
                    entry = {
                        'Filename': filename,
                        'Target Class': target_class[j].item(),
                        'Predicted Class': predicted_class[j].item()
                    }
                    class_data[target_class[j].item()].append(entry)
    
                    # Update accuracy tracking
                    if target_class[j].item() == predicted_class[j].item():
                        total_correct += 1
                    total_samples += 1
                
                # Show progress at 10% intervals
                if (i + 1) % progress_interval == 0 or i == len(test_loader) - 1:
                    print(f"Progress Done (PLA): {((i + 1) / len(test_loader)) * 100:.1f}%")

    # Save predictions to class-specific CSV files
    for target_class, entries in class_data.items():
        df = pd.DataFrame(entries)
        df.to_csv(os.path.join(result_folder, f'class_{target_class}.csv'), index=False)

    # Calculate and print patch level accuracy
    patch_level_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print(f"Patch Level Accuracy: {patch_level_accuracy:.2f}%")
    
    # Flatten predicted data for further processing
    predicted_data = []
    for entries in class_data.values():
        for entry in entries:
            predicted_data.append((entry['Filename'], entry['Target Class'], entry['Predicted Class']))
    
    return patch_level_accuracy,predicted_data


def image_level_predictions(directory,clustered_patch=False):
    csv_files = [f for f in os.listdir(directory) if f.startswith("class_") and f.endswith(".csv")]
    total_images = 0
    correct_images = 0

    predictions = []  # List to store image-level predictions

    for file in csv_files:
        result_dict = {}
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        target_class = None

        for index, row in df.iterrows():
            filename = row.iloc[0]  # First column
            target_class = row.iloc[1]  # Second column (true class)
            predicted_class = row.iloc[2]  # Third column (predicted class)
            
            A_value = None
            if clustered_patch:
                A_value = filename.rsplit("_", 2)[0]
            else:
                A_value = filename.rsplit("_", 1)[0]
                
            if A_value not in result_dict:
                result_dict[A_value] = []
            
            result_dict[A_value].append(predicted_class)

        total_images += len(result_dict)

        for key, values in result_dict.items():
            voted_class = Counter(values).most_common(1)[0][0]
            if voted_class == target_class:
                correct_images += 1

            # Store the predictions
            predictions.append([key, target_class, voted_class])

    # Save predictions to a CSV file
    output_file = os.path.join(directory, "Image Level Prediction.csv")
    df_predictions = pd.DataFrame(predictions, columns=["Filename", "Target", "Predictions"])
    df_predictions.to_csv(output_file, index=False)

    #print("ILA:", correct_images / total_images)
    ila = (correct_images / total_images)*100
    #print(f"Predictions saved to {output_file}")
    return ila