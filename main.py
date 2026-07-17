import os
import csv
import glob
import logging
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# =====================================================================
# 1. CONFIGURATION (Notebook Friendly)
# =====================================================================
@dataclass
class Config:
    """Replaces argparse for seamless Jupyter Notebook usage."""
    use_latest_model: bool = True
    train_model: bool = True       # Set to False if you only want to evaluate
    batch_size: int = 1
    learning_rate: float = 0.001
    epochs: int = 1000
    patience: int = 5              # Early stopping patience
    min_delta: float = 0.001       # Early stopping minimum improvement
    log_file: str = 'app.log'
    csv_file: str = 'results.csv'
    data_dir: str = './data'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================================================================
# 2. UTILITY & SETUP FUNCTIONS
# =====================================================================
def setup_logger(log_file: str):
    """Configures logging to output to both a file and the notebook cell."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate prints in notebooks
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console (Stream) Handler for Notebooks
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)

def get_dataloaders(config: Config):
    """Downloads data and returns training and testing dataloaders."""
    train_dataset = torchvision.datasets.MNIST(
        root=config.data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.MNIST(
        root=config.data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True)
    
    return train_loader, test_loader

def log_results_to_csv(config: Config, filename: str, train_loss: float, test_loss: float, test_acc: float):
    """Appends execution metrics to a local CSV file."""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(config.csv_file)

    with open(config.csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Model File', 'Avg Train Loss', 'Test Loss', 'Test Accuracy (%)'])
            
        writer.writerow([
            timestamp_str,
            filename,
            f"{train_loss:.4f}" if train_loss is not None else "N/A",
            f"{test_loss:.4f}",
            f"{test_acc * 100:.2f}%"
        ])
    logging.info(f"Successfully appended execution results to {config.csv_file}")

# =====================================================================
# 3. MODEL ARCHITECTURE
# =====================================================================
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def create_model():
    """Defines and returns the CNN architecture."""
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.15),

        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.15),

        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.15),
        
        Flatten(),
        nn.Linear(32 * 3 * 3, 10),
        nn.Sigmoid()
    )

def load_latest_checkpoint(model: nn.Module, optimizer: optim.Optimizer, config: Config):
    """Loads the latest model weights and optimizer state if they exist."""
    saved_models = glob.glob("model_*.pth")
    if config.use_latest_model and saved_models:
        latest_model_file = max(saved_models, key=os.path.getmtime)
        logging.info(f"--> Found existing models. Loading weights from: {latest_model_file}")
        
        checkpoint = torch.load(latest_model_file, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return latest_model_file, checkpoint.get('epoch', 0)
    
    logging.info("--> Starting from scratch (no previous models loaded).")
    return None, 0

# =====================================================================
# 4. TRAINING & EVALUATION LOGIC
# =====================================================================
def calculate_loss_and_metrics(outputs, labels, criterion):
    """Centralized loss calculation and accuracy metrics for MNIST."""
    # Convert labels to one-hot for MSE Loss
    one_hot_labels = nn.functional.one_hot(labels, num_classes=10).float()
    loss = criterion(outputs, one_hot_labels)
    
    # Calculate accuracy
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    
    return loss, correct

def train_epoch(model, dataloader, optimizer, criterion, config):
    """Handles a single full pass through the training data."""
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        images, labels = images.float().to(config.device), labels.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss, _ = calculate_loss_and_metrics(outputs, labels, criterion)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, config):
    """Evaluates the model on the testing data."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.float().to(config.device), labels.to(config.device)
            outputs = model(images)
            
            loss, correct = calculate_loss_and_metrics(outputs, labels, criterion)
            total_loss += loss.item()
            correct_predictions += correct
            total_samples += labels.size(0)
            
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# =====================================================================
# 5. MAIN EXECUTION
# =====================================================================
def main():
    # 1. Initialize config and logging
    config = Config()
    setup_logger(config.log_file)
    logging.info(f"Using device: {config.device}")
    
    # 2. Setup Data, Model, Optimizer, and Loss
    train_loader, test_loader = get_dataloaders(config)
    
    model = create_model().to(config.device)
    logging.info(f"Network Architecture Initialized")
    
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 3. Load checkpoint if applicable
    latest_model_file, start_epoch = load_latest_checkpoint(model, optimizer, config)
    
    # 4. Training Loop
    final_filename = latest_model_file
    final_train_loss = None
    
    if config.train_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f'runs/{timestamp}')
        
        best_loss = float('inf')
        patience_counter = 0
        final_epoch = start_epoch
        
        for epoch in range(start_epoch, config.epochs):
            # Train
            avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
            final_train_loss = avg_train_loss
            final_epoch = epoch
            
            # Evaluate
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, config)
            
            # Log metrics
            logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy*100:.2f}%")
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', test_loss, epoch)
            writer.add_scalar('Accuracy/Validation', test_accuracy, epoch)
            
            # Early Stopping
            if avg_train_loss < (best_loss - config.min_delta):
                best_loss = avg_train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                logging.info(f"--> Loss plateaued. Patience: {patience_counter}/{config.patience}")
                
            if patience_counter >= config.patience:
                logging.info(f"Stopping early at epoch {epoch+1}! No improvement in {config.patience} epochs.")
                break
                
        # Save the newly trained model
        final_filename = f"model_{timestamp}.pth"
        checkpoint = {
            'epoch': final_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': final_train_loss
        }
        torch.save(checkpoint, final_filename)
        logging.info(f"Saved new checkpoint: {final_filename}")

    # 5. Final Evaluation & CSV Logging (runs even if we just loaded a model and skipped training)
    logging.info("Running final evaluation...")
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, config)
    log_results_to_csv(config, final_filename, final_train_loss, final_test_loss, final_test_acc)

# Execute if run as a script (or run this cell in a notebook)
if __name__ == "__main__":
    main()