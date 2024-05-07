import streamlit as st
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import gdown

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Assuming the model expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Google Drive link to the model file
file_id = '1C45H947jWNxeVMa0bZe-qW0aV47wwYNG'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'ai_vs_real_model.pth'
gdown.download(url, output, quiet=False)

# Assuming model is defined or loaded elsewhere in your script
from torchvision import models

# Initialize the model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Set the output size to match your project's classes
# Load the state dict into the model
model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))

model.eval()

def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    class_labels = {
        0: 'Mild Demented',
        1: 'Moderate Demented',
        2: 'Non Demented',
        3: 'Very Mild Demented'
    }
    return class_labels[predicted_class]

# Streamlit interface
st.title('Dementia MRI Scan Classifier')
uploaded_file = st.file_uploader("Upload an MRI scan image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
    st.write("")
    if st.button('Predict'):
        result = predict(image)
        st.write(f"The MRI scan is classified as: *{result}*")
