# # Missclassification Analysis
# import torch
# import streamlit as st
# from torchvision import transforms
# from PIL import Image
# import torch.nn as nn
# import torch.nn.functional as F

# # Define the device and load the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class EnhancedCNNModel(nn.Module):
#     def __init__(self):
#         super(EnhancedCNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(16384, 512)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 1)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = x.view(-1, 16384)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

# # Load the trained model
# model = torch.load("D:/ML-AI/Notes-Practice work/Algorithms/Deep Learning/Saved_Model/defect_detection_casting_model.pth", map_location=device)
# model.to(device)
# model.eval()

# # Define the transformation
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# # Streamlit UI setup
# st.title("Defect Detection for Casting Products")
# st.write("Upload an image to check if it has defects.")

# # Image upload
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     # Transform the image
#     input_image = transform(image).unsqueeze(0).to(device)
    
#     # Perform inference
#     with torch.no_grad():
#         output = model(input_image)
#         prediction = torch.sigmoid(output).item()  # Get the sigmoid score

#     # Set threshold for classification (e.g., 0.5)
#     threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
#     st.write("Adjust the threshold to fine-tune sensitivity for defect classification.")
    
#     # Interpret the prediction
#     if prediction >= threshold:
#         st.error("Result: Defective")
#     else:
#         st.success("Result: No Defect")
    
#     # Display confidence score
#     st.write(f"Confidence Score: {prediction:.2f}")

# Missclassification Analysis
import torch
import streamlit as st
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the device and load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load the trained model
model = torch.load("D:/ML-AI/Notes-Practice work/Algorithms/Deep Learning/Saved_Model/defect_detection_casting_model.pth", map_location=device)
model.to(device)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Streamlit UI setup
st.title("Defect Detection for Casting Products")
st.write("Upload an image to check if it has defects.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Transform the image
    input_image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        prediction = torch.sigmoid(output).item()  # Get the sigmoid score

    # Set threshold for classification (e.g., 0.5)
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.write("Adjust the threshold to fine-tune sensitivity for defect classification.")
    
    # Reverse the classification logic
    if prediction >= threshold:
        st.success("Result: No Defect")
    else:
        st.error("Result: Defective")
    
    # Display confidence score
    st.write(f"Confidence Score: {prediction:.2f}")
