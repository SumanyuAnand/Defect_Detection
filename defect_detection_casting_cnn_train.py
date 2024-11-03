import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import os
import pandas as pd
#set device to GPU if available 
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

class EnhancedCNNModel(nn.Module):
    def __init__(self):
        super(EnhancedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16384, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 16384)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Load the saved model
model = EnhancedCNNModel().to(device)



# Change the loss function to BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

# Define optimizer (using AdamW)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)  # You can adjust the learning rate

# Improved data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the datasets
train_dataset = datasets.ImageFolder(root='D:/ML-AI/Notes-Practice work/Algorithms/Deep Learning/Datasets/Images/casting_data/casting_data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='D:/ML-AI/Notes-Practice work/Algorithms/Deep Learning/Datasets/Images/casting_data/casting_data/test', transform=transform)

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Path to save the trained model
save_path = "D:/ML-AI/Notes-Practice work/Algorithms/Deep Learning/Saved_Model/defect_detection_casting_model.pth"

# Using ReduceLROnPlateau as a learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Lists to keep track of training loss and accuracy
train_losses = []
train_accuracies = []


# Training the model
num_epochs = 25  # Total number of epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Change learning rate after epoch 22
    if epoch == 22:
        new_lr = 1e-4  # Set the new learning rate here
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Learning rate changed to {new_lr}")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()
        print(f"Epoch {epoch+1}/{num_epochs}, Batch images shape: {images.shape}, labels shape: {labels.shape}")
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.view(-1), labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate accuracy
        predicted = (torch.sigmoid(outputs.view(-1)) >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Step the scheduler
    scheduler.step(running_loss)  # Use running_loss for ReduceLROnPlateau

    # Calculate average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')  # Print the current learning rate


# Save the model after training
torch.save(model, save_path)
print(f"Entire model saved at {save_path}")

# Evaluating the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images)
        predicted = (torch.sigmoid(outputs.view(-1)) >= 0.5).float()  # Apply sigmoid to get probabilities
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')


# Create a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = (torch.sigmoid(outputs.view(-1)) >= 0.5).float()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

# Save confusion matrix and classification report
confusion_matrix_df = pd.DataFrame(conf_matrix, index=['Actual ok', 'Actual defects'], columns=['Predicted ok', 'Predicted defects'])
confusion_matrix_df.to_csv('D:/ML-AI/Notes-Practice work/Algorithms/Deep Learning/Saved_Model/defect_confusion_matrix.csv')

with open('D:/ML-AI/Notes-Practice work/Algorithms/Deep Learning/Saved_Model/defect_classification_report.txt', 'w') as f:
    f.write(class_report)

# Save training history to CSV
history_df = pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Loss': train_losses, 'Accuracy': train_accuracies})
history_df.to_csv('D:/ML-AI/Notes-Practice work/Algorithms/Deep Learning/Saved_Model/defect_model_history.csv', index=False)
# Plotting the training loss and accuracy
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.xticks(range(1, num_epochs + 1))
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='green')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.xticks(range(1, num_epochs + 1))
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()  # Display the plot

