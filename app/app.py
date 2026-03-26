import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload MRI image to detect tumor</p>", unsafe_allow_html=True)

# Model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 54 * 54, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Paths
MODEL_PATH = "model/brain_tumor_model.pth"
GRAPH_PATH = "model/loss_graph.png"

# Load model safely
model = None

if os.path.exists(MODEL_PATH):
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
else:
    st.warning("⚠️ Model file not found. Prediction disabled.")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Only predict if model exists
    if model:
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            prediction = output.item()

        confidence = prediction if prediction > 0.5 else (1 - prediction)

        st.write(f"📊 Confidence: {confidence * 100:.2f}%")

        if prediction > 0.5:
            st.error("⚠️ Tumor Detected")
        else:
            st.success("✅ No Tumor Detected")
    else:
        st.info("Model not available. Upload model to enable prediction.")

# Show graph
st.subheader("📈 Training Graph")

if os.path.exists(GRAPH_PATH):
    st.image(GRAPH_PATH)
else:
    st.write("Train model to generate graph")

st.markdown("---")
st.markdown("⚠️ This is for educational purposes only")import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload MRI image to detect tumor</p>", unsafe_allow_html=True)

# Model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 54 * 54, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Paths
MODEL_PATH = "model/brain_tumor_model.pth"
GRAPH_PATH = "model/loss_graph.png"

# Load model safely
model = None

if os.path.exists(MODEL_PATH):
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
else:
    st.warning("⚠️ Model file not found. Prediction disabled.")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Only predict if model exists
    if model:
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            prediction = output.item()

        confidence = prediction if prediction > 0.5 else (1 - prediction)

        st.write(f"📊 Confidence: {confidence * 100:.2f}%")

        if prediction > 0.5:
            st.error("⚠️ Tumor Detected")
        else:
            st.success("✅ No Tumor Detected")
    else:
        st.info("Model not available. Upload model to enable prediction.")

# Show graph
st.subheader("📈 Training Graph")

if os.path.exists(GRAPH_PATH):
    st.image(GRAPH_PATH)
else:
    st.write("Train model to generate graph")

st.markdown("---")
st.markdown("⚠️ This is for educational purposes only")