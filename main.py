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
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # then a sigmoid layer
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):
        # Pass input through the first layer
        x = self.fc1(x)
        # Apply the ReLU activation
        x = self.relu(x)
        # Pass through the fc2 layer
        x = self.fc2(x)
        # sigmoid
        x = self.sigmoid(x)	
        return x


model = SimpleNN(input_size=28*28, hidden_size=8, output_size=10)

model_loaded = False
if USE_LATEST_MODEL:
    # Look for any files matching the pattern
    saved_models = glob.glob("model_*.pth")
    if saved_models:
        # Sort files by modification time to grab the absolute latest one
        latest_model_file = max(saved_models, key=os.path.getmtime)
        print(f"--> Found existing models. Loading weights from: {latest_model_file}")
        model.load_state_dict(torch.load(latest_model_file))
        model_loaded = True
    else:
        print("--> USE_LATEST_MODEL is True, but no 'model_*.pth' files were found. Starting from scratch.")

print("Network Architecture:\n", model)

criterion = nn.MSELoss(reduction='sum')

if TRAIN_MODEL or not model_loaded:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10): # Loop over the entire dataset 3 times
        running_loss = 0.0
        for images, labels in train_loader:
            # Flatten images from (batch_size, 1, 28, 28) to (batch_size, 784)
            # default is not a vector, so I was having issues.
            images = images.reshape(-1, 28*28)
            images = images.float() 
            optimizer.zero_grad()
            outputs = model(images)
            # default is not one hot in mnist, so I got a warning.
            one_hot_labels = nn.functional.one_hot(labels, num_classes=10).float()
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {running_loss/len(train_loader):.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.pth"
    # Save the state dict
    torch.save(model.state_dict(), filename)


model.eval()  # Disables dropout and batch normalization updates
total_loss = 0.0
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
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

print(f"Test Loss: {average_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")