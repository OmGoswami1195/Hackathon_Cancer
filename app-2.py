import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Create a sidebar for page selection
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Select a Page", ["Cancer Predictor", "Types of Brain Cancer"])

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1631563019676-dade0dbdb8fc?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Load the trained model
model_path = "brain_cancer.h5"  # Update with the correct path
loaded_model = tf.keras.models.load_model(model_path)

# Define class details
class_details = ["brain_glioma", "brain_menin", "brain_tumor"]

# Streamlit App
st.title("Brain Cancer Classification App")

# Function to display cancer details
def display_cancer_details(cancer_type):
    if cancer_type == "brain_glioma":
        st.header("Brain Glioma")
        st.write("Brain gliomas are tumors that arise from glial cells, which are supportive cells of the brain.")
        st.write("They can occur in various parts of the brain and may be classified as low-grade or high-grade.")
        st.write("Symptoms may include headaches, seizures, and neurological deficits.")

    elif cancer_type == "brain_menin":
        st.header("Brain Meningioma")
        st.write("Brain meningiomas are tumors that develop from the meninges, the layers of tissue covering the brain.")
        st.write("These tumors are typically slow-growing and may be benign in many cases.")
        st.write("Symptoms may include headaches, changes in vision, and seizures.")

    elif cancer_type == "brain_tumor":
        st.header("Brain Tumor (General)")
        st.write("Brain tumors can be of various types, including gliomas, meningiomas, and metastatic tumors.")
        st.write("They can be benign or malignant and may cause a range of symptoms depending on their location.")
        st.write("Treatment options include surgery, radiation therapy, and chemotherapy.")


if page == "Cancer Predictor":
  # File Upload
  uploaded_file = st.file_uploader("Choose an image...", type="jpg")

  if uploaded_file is not None:
      # Display the uploaded image
      st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

      # Preprocess the image
      image = Image.open(uploaded_file).resize((256, 256))
      image_array = np.array(image) / 255.0
      image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

      # Make prediction
      prediction = loaded_model.predict(image_array)
      predicted_class = np.argmax(prediction)
      predicted_label = class_details[predicted_class]
      confidence = prediction[0][predicted_class]

      # Display prediction
      st.write(f"Prediction: {predicted_label}")
      st.write(f"Confidence: {confidence:.2%}")
elif page == "Types of Brain Cancer":
    st.header("Types of Brain Cancer")
    st.write("There are several types of brain cancer, including:")
    st.markdown("- **Brain Glioma:** A tumor arising from glial cells.")
    st.markdown("- **Brain Meningioma:** A tumor developing from the meninges.")
    st.markdown("- **Brain Tumor (General):** Various types, including gliomas, meningiomas, and metastatic tumors.")

    # Display details for each type of brain cancer
    selected_cancer_type = st.selectbox("Select a specific type of brain cancer:", class_details)
    display_cancer_details(selected_cancer_type)


