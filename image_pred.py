import os
import gdown
import streamlit as st
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models

# target class names (e.g., FashionMNIST classes)
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# loading and defining the model for predictions
@st.cache_resource # cache the model loading function to avoid reloading on every interaction
def load_weights_from_gdrive():
    file_id = "1ePzG42SL7qy_tJ7p7cL-mxaTmssClbT9" # google drive file ID for model weights
    output_path = "FashionNIST.pth"  # local path to save the downloaded weights

    # Construct the internal download URL
    url = f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link"

    # Only download if the file does not already exist locally
    if not os.path.exists(output_path):
        with st.spinner(
            "Downloading FashionMNIST weights from Google Drive... Please wait."
        ):
            try:
                gdown.download(url, output_path, quiet=True)
                st.success("Weights downloaded successfully!")
            except Exception as e:
                st.error(f"Download failed: {e}")

    return output_path

load_weights_from_gdrive() #call the function to download weights from google drive

@st.cache_resource
def load_model(): # defining the model loading function
    model = models.vgg16(pretrained=True) # Load pre-trained VGG16 model
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10)
    )
    # load state dict and set to evaluation mode
    model.load_state_dict(torch.load("FashionNIST.pth", map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    return model

model = load_model() # call model function 

# define image transformations for preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), #transforms grayscale object to 3-RGB channels
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ]
)

st.title("Image Classifier")

st.write("This app takes image and classifies objects into the ideal class")

# upload image to the app for prediction
uploaded_image = st.file_uploader("Choose an image", type = ['png','jpg','jpeg'])

# validating image upload
if uploaded_image is not None:
    image = Image.open(uploaded_image) # open the image file using PIL

st.image(
    image,
    caption="Uploaded Image",
    use_container_width=True
) 

# print image details
st.write(f"Dimensions: {image.size[0]}x{image.size[1]} pixels")

#transform image to a 224 x 224 dimension
transformed_image = transform(image).unsqueeze(0) # add batch dimension for model input

# print transfomed image details to validate transformation
st.write(f"Dimensions: {transformed_image.shape[1]}x{transformed_image.shape[2]} pixels") #confirm transformation


#using model to predict labels
if st.button("Predict"):
    with torch.no_grad():
        # get raw model outputs (logits)
        outputs = model(transformed_image)
        
        # use softmax to convert logits to probability scores (0.0 to 1.0)
        # softmax was applied since this is a multi-class classification problem
        probabilities = F.softmax(outputs, dim=1)[0]
        
        # get top prediction class index and value
        confidence, prediction = torch.max(probabilities, dim=0)

        # display top prediction & confidence
        predicted_label = classes[prediction.item()]
        confidence_pct = confidence.item() * 100

    st.success(f"**Predicted Label:** {predicted_label}")
    st.metric(label="Confidence Level", value=f"{confidence_pct:.2f}%")


    # format probabilities into a DataFrame for Streamlit bar chart
    prob_df = pd.DataFrame({
        "Class": classes,
        "Probability": probabilities.numpy()
    }).set_index("Class")

    st.bar_chart(prob_df) #build bar chart to visualize the probabilities for each class



