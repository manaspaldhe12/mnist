import csv
import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from datetime import datetime
import torch.optim as optim

import argparse
from torch.utils.tensorboard import SummaryWriter
import logging

# Configure the logger to save to a local file
logging.basicConfig(
    filename='app.log',          # The name of your local log file
    filemode='a',                # 'a' appends data; 'w' overwrites the file each run
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO           # Minimum severity level to capture
)



# =====================================================================
# COMMAND-LINE INTERFACE CONFIGURATION
# =====================================================================
parser = argparse.ArgumentParser(description="Train and evaluate a simple MNIST NN.")

# Flag that defaults to True. Passing '--no_latest' flips it to False.
parser.add_argument(
    '--no_latest', 
    dest='use_latest_model', 
    action='store_false', 
    help="Do not load the latest trained model; start fresh."
)
parser.set_defaults(use_latest_model=True)

# Flag that defaults to False. Passing '--train' flips it to True.
parser.add_argument(
    '--train', 
    dest='train_model', 
    action='store_true', 
    help="Run the training loop phase."
)
parser.set_defaults(train_model=False)

# Parse the arguments from the terminal execution command
args = parser.parse_args()

# Access values via args.use_latest_model and args.train_model
USE_LATEST_MODEL = args.use_latest_model
TRAIN_MODEL = args.train_model

# Download and load the Training Data
train_dataset = torchvision.datasets.MNIST(
    root='./data',       # Directory where data will be stored
    train=True,          # Load training split (60,000 samples)
    download=True,       # Fetch from internet if not locally present
    transform = transforms.ToTensor()
)

# Download and load the Test Data
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,         # Load test split (10,000 samples)
    download=True,
    transform = transforms.ToTensor()
)

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# Define the network architecture
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.15),

    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.15),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.15),
    
    Flatten(),
    nn.Linear(32 * 3 * 3, 10),
    nn.Sigmoid()
)

model_loaded = False
if USE_LATEST_MODEL:
    # Look for any files matching the pattern
    saved_models = glob.glob("model_*.pth")
    if saved_models:
        # Sort files by modification time to grab the absolute latest one
        latest_model_file = max(saved_models, key=os.path.getmtime)
        logging.info(f"--> Found existing models. Loading weights from: {latest_model_file}")
        model.load_state_dict(torch.load(latest_model_file))
        model_loaded = True
    else:
        logging.info("--> USE_LATEST_MODEL is True, but no 'model_*.pth' files were found. Starting from scratch.")

logging.info(f"Network Architecture:\n {model}")

criterion = nn.MSELoss(reduction='sum')

start_epoch = 0
optimizer = optim.Adam(model.parameters(), lr=0.001) # aha! this was too high.. but I thought loss will be swingy... I guess gradients are swingy, loss stayed high.

if USE_LATEST_MODEL and saved_models:
    checkpoint = torch.load(latest_model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    logging.info(f"--> Resuming training from epoch {start_epoch}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f'runs/{timestamp}')
if TRAIN_MODEL or not model_loaded:

    # Early stopping parameters (vibecoded... because this is not ML)
    patience = 5            # Number of epochs to wait without improvement
    min_delta = 0.001       # Minimum change to qualify as an improvement
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1000): # Loop over the entire dataset 3 times
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            # Flatten images from (batch_size, 1, 28, 28) to (batch_size, 784)
            # default is not a vector, so I was having issues.
            images = images.float() 
            optimizer.zero_grad()
            outputs = model(images)
            # default is not one hot in mnist, so I got a warning.
            one_hot_labels = nn.functional.one_hot(labels, num_classes=10).float()
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        model.eval()  # Disables dropout and batch normalization updates
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.float() 
                outputs = model(images)
                one_hot_labels = nn.functional.one_hot(labels, num_classes=10).float()
                loss = criterion(outputs, one_hot_labels)
                total_loss += loss.item()
                # Calculate accuracy (assuming a classification task)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)


        average_test_loss = total_loss / total_samples
        test_accuracy = correct_predictions / total_samples

        logging.info(f"Test Loss: {average_test_loss:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        writer.add_scalar('Loss/Validation', average_test_loss, epoch)
        writer.add_scalar('Accuracy/Validation', test_accuracy, epoch)

        # --- EARLY STOPPING LOGIC ---
        # Check if the loss improved by at least min_delta
        if avg_loss < (best_loss - min_delta):
            best_loss = avg_loss
            patience_counter = 0  # Reset patience because we improved
            
            # Optional but recommended: Save the best model weights here
            # torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            logging.info(f"--> Loss plateaued. Patience: {patience_counter}/{patience}")
            
        # If we run out of patience, exit the training loop
        if patience_counter >= patience:
            logging.info(f"Stopping early at epoch {epoch+1}! Loss hasn't improved in {patience} epochs.")
            break # This exits the 'for epoch in range(1000)' loop

    filename = f"model_{timestamp}.pth"
    # Save the state dict
    # Instead of just saving model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss
    }
    torch.save(checkpoint, filename)



# Prepare the data row
timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
csv_file = 'results.csv'
file_exists = os.path.isfile(csv_file)

# Append the metrics to the CSV file
with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    
    # Write the header line only if the file is being newly created
    if not file_exists:
        writer.writerow(['Timestamp', 'Model File', 'Avg Train Loss', 'Test Loss', 'Test Accuracy (%)'])
        
    # Append the results row
    writer.writerow([
        timestamp_str,
        filename if (TRAIN_MODEL or not model_loaded) else latest_model_file,
        f"{running_loss/len(train_loader):.4f}" if (TRAIN_MODEL or not model_loaded) else "N/A",
        f"{average_test_loss:.4f}",
        f"{test_accuracy * 100:.2f}%"
    ])

logging.info(f"Successfully appended execution results to {csv_file}")