import os
import torch
import torch.nn as nn
import csv

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)  # Only for single GPU

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        output = torch.clamp(output, 1e-32, 1 - 1e-32)
        return -(target * torch.log(output)).sum(dim=1).mean()

def Test(model, test_loader, device, return_result=False,dual_image=False,dual=False):
    model.eval()
    criterion = CrossEntropy()
    test_loss = 0
    total_correct_test = 0
    total_batches = len(test_loader)
    
    if dual:
        with torch.no_grad():
            for batch_idx, (imgs1,imgs2, labels) in enumerate(test_loader):
                img_org, mat_img, target = imgs1.to(device, dtype=torch.float),imgs2.to(device, dtype=torch.float), labels.to(device)
                output = model(img_org,mat_img)
                
                # Compute loss
                loss = criterion(output, target)
                test_loss += loss.item()  # Sum up batch loss
                
                # Calculate accuracy
                _, actual = torch.max(target.data, 1)
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == actual).sum().item()
                total_correct_test += correct
    else:
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(test_loader):
                img_org, target = imgs.to(device, dtype=torch.float), labels.to(device)
                output = model(img_org)
                
                # Compute loss
                loss = criterion(output, target)
                test_loss += loss.item()  # Sum up batch loss
                
                # Calculate accuracy
                _, actual = torch.max(target.data, 1)
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == actual).sum().item()
                total_correct_test += correct

    # Calculate average loss and accuracy
    avg_loss = test_loss / total_batches
    accuracy = 100.0 * total_correct_test / len(test_loader.dataset)
    
    # Print final results
    print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
            
    if return_result:
        return {'accuracy': accuracy, 'loss': avg_loss}
    else:
        return


def Train(model, train_data, optimizer,device,scheduler=None, valid_data=None, epochs=1, return_result=True, 
          overwrite_model=True, save_path='best_model.pth', csv_file=None,dual=False):
    criterion = CrossEntropy()
    train_loss_history = []
    valid_loss_history = []
    train_accuracy_history = []
    valid_accuracy_history = []
    best_accuracy = 0  # Track the best validation accuracy

    # Check if CSV logging is enabled
    if csv_file:
        csv_exists = os.path.isfile(csv_file)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_correct_train = 0
        total_batches = len(train_data)
        print("Total number of batches: ",total_batches)
        interval = max(1, total_batches // 5)  # 20% interval
        
        if dual:
            for batch_idx, (imgs1,imgs2, labels) in enumerate(train_data):
                img_org, mat_img, target = imgs1.to(device, dtype=torch.float),imgs2.to(device, dtype=torch.float), labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = model(img_org,mat_img)
                loss = criterion(output, target)

                total_train_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()

                # Calculate accuracy for the batch
                _, actual = torch.max(target.data, 1)
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == actual).sum().item()
                total_correct_train += correct
                
                if (batch_idx + 1) % interval == 0 or batch_idx == total_batches - 1:
                    print(f"Epoch [{epoch + 1}/{epochs}] - {batch_idx + 1}/{total_batches} batches completed.")
        else:
            for batch_idx, (imgs, labels) in enumerate(train_data):
                img_org, target = imgs.to(device, dtype=torch.float), labels.to(device)
    
                # Forward pass
                optimizer.zero_grad()
                output = model(img_org)
                loss = criterion(output, target)
    
                total_train_loss += loss.item()
    
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
    
                # Calculate accuracy for the batch
                _, actual = torch.max(target.data, 1)
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == actual).sum().item()
                total_correct_train += correct
                
                if (batch_idx + 1) % interval == 0 or batch_idx == total_batches - 1:
                    print(f"Epoch [{epoch + 1}/{epochs}] - {batch_idx + 1}/{total_batches} batches completed.")
        
        

        # Calculate and store average training loss and accuracy for the epoch
        avg_train_loss = total_train_loss / total_batches
        train_accuracy = 100.0 * total_correct_train / len(train_data.dataset)
        train_loss_history.append(avg_train_loss)
        train_accuracy_history.append(train_accuracy)
        print(f"\nEpoch [{epoch + 1}/{epochs}] - Average Training Loss: {avg_train_loss:.4f}, "
              f"Training Accuracy: {train_accuracy:.2f}%")

        # Optional validation step
        avg_valid_loss = None
        valid_accuracy = None
        if valid_data is not None:
            validation_results = Test(model, valid_data, device, return_result=True,dual=dual)
            avg_valid_loss = validation_results['loss']
            valid_accuracy = validation_results['accuracy']
            valid_loss_history.append(avg_valid_loss)
            valid_accuracy_history.append(valid_accuracy)

            # Save model if validation accuracy improves
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                if overwrite_model:
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved with accuracy: {valid_accuracy:.2f}%\n")

        # Write metrics to CSV if csv_file parameter is provided
        if csv_file:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Write header only if the file does not exist
                if not csv_exists:
                    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Valid Loss', 'Valid Accuracy'])
                    csv_exists = True  # Set flag to avoid rewriting header in subsequent epochs

                writer.writerow([
                    epoch + 1,
                    avg_train_loss,
                    train_accuracy,
                    avg_valid_loss if valid_data else None,
                    valid_accuracy if valid_data else None
                ])

    # Return training and validation results if required
    if return_result:
        results = {
            "train_loss": train_loss_history,
            "train_accuracy": train_accuracy_history,
            "valid_loss": valid_loss_history if valid_data else None,
            "valid_accuracy": valid_accuracy_history if valid_data else None
        }
        return results