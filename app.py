import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img

model = tf.keras.models.load_model('rice_image_classification_.hdf5')

def classify_image(image_path, model):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(175, 175))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    # Make prediction using the loaded model
    prediction = model.predict(img_array)

    # Return the predicted class label and confidence score
    class_index = np.argmax(prediction)
    class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    class_label = class_labels[class_index]
    confidence_score = prediction[0][class_index]

    return class_label, confidence_score

def home():
    st.title("Welcome to Rice Image Classification!")
    st.write("This website uses a machine learning model to classify different types of rice.")
    st.write("Upload an image of a rice grain and the model will predict its type.")
    st.write("Click on the 'Classify' button to initiate the classification process.")

def about():
    st.title("About Rice Image Classification")
    st.write("This website was created to help people identify different types of rice grains using machine learning.")
    st.write("The model used in this website was trained on a dataset of rice images from around the world.")
    st.write("We hope that this website can help people learn more about rice and appreciate the diversity of this important crop.")

def contact():
    st.title("Contact Us")
    st.write("If you have any questions or comments about this website, please feel free to contact us.")
    st.write("Email: contact@riceimageclassification.com")



def vendors():
    st.title("Find Rice Vendors")
    st.write("Allow us to locate you and find rice wholesalers near you:")
    
    # Use the Streamlit `st.session_state` object to store the user's location
    if "location" not in st.session_state:
        st.session_state.location = None
    
    # Add a button to initiate the search for wholesalers
    if st.button("Search"):
        # Use the Mapbox API to search for wholesalers near the user's location
        url = "https://api.mapbox.com/geocoding/v5/mapbox.places/rice%20wholesalers.json"
        params = {
            "access_token": "pk.eyJ1IjoidGV2aW50ZW11IiwiYSI6ImNsZmV4Z2RwcDFkZW0zeG4xb21yemRqemwifQ.GahZE16VVDxTl9RDQHz9Lg",
            "proximity": st.session_state.location,
            "types": "poi",
            "limit": 10,
        }
        response = requests.get(url, params=params).json()
        print(response)
        
        # Display the search results
        if "features" in response:
            st.write("Results:")
            data = []
            for feature in response["features"]:
                name = feature["text"]
                address = ", ".join(feature["place_name"].split(", ")[1:])
                longitude, latitude = feature["center"]
                
                data.append({"name": name, "address": address, "latitude": latitude, "longitude": longitude})
                
            # Create a Pandas DataFrame with the search results
            df = pd.DataFrame(data)
            
            # Add a marker for each rice wholesaler
            st.map(df, use_container_width=True)
            st.write(df[["name", "address"]])
        else:
            st.write("No results found.")
    
    # Use the Mapbox API to get the user's location and store it in `st.session_state`
    if st.session_state.location is None:
        url = "https://api.mapbox.com/geolocation/v1/ip"
        params = {"access_token": "pk.eyJ1IjoidGV2aW50ZW11IiwiYSI6ImNsZmV4Z2RwcDFkZW0zeG4xb21yemRqemwifQ.GahZE16VVDxTl9RDQHz9Lg"}
        response = requests.get(url, params=params).json()
        print(response)
        try:
            longitude = response["location"]["longitude"]
            latitude = response["location"]["latitude"]
            st.session_state.location = f"{longitude},{latitude}"
        except KeyError:
            st.write("Could not get location. Please allow location access in your browser settings.")
    
    # Render a map image centered on the user's location
    url = "https://api.mapbox.com/styles/v1/mapbox/streets-v11/static"
    params = {
        "access_token": "pk.eyJ1IjoidGV2aW50ZW11IiwiYSI6ImNsZmV4Z2RwcDFkZW0zeG4xb21yemRqemwifQ.GahZE16VVDxTl9RDQHz9Lg",
        "center": st.session_state.location,
        "zoom": 10,
        "size": "600x400",
        "retina": True,
    }
    response = requests.get(url, params=params)

    # Display the map image using the `pyplot` module from `matplotlib`
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("Could not load map.")




def main():
    st.set_page_config(page_title="Rice Image Classification", page_icon="🍚", layout="wide")
    st.sidebar.title("Navigation")
    pages = {"Home": home, "About": about, "Contact": contact, "Vendors": vendors}
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Display the selected page
    page = pages[selection]
    page()

    # Add a file uploader widget on the home page
    if selection == "Home":
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Add a button to initiate the classification process
            if st.button("Classify"):
                # Call the classify_image function
                class_label, confidence_score = classify_image(uploaded_file, model)

                # Display the predicted class label and confidence score
                st.write(f"Predicted class: {class_label}")
                st.write(f"Confidence score: {confidence_score:.2f}")

if __name__ == '__main__':
    main()
