import streamlit as st
from PIL import Image
import numpy as np 
from fastai.tabular.learner import load_learner
from fastai import *
from fastai.vision import *
import torch

st.title("Histopathological Image Classification with PyTorch")
st.header("MSI-MSS prediction Example")
st.text("Upload a histopathological images for classifying as MSI or no-MSS")

def predict():
    # load the pre trained model
    learn_inf = load_learner('export.pkl')
    learn_inf.dls.vocab
    pred_class,pred_idx,probs = learn_inf.predict(img)
    st.write('predicted Class: {} '.format(pred_class) )
    st.write('Predicted Probability:{:.3} '.format(float(probs[pred_idx])))
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

# Uploading the File to the Page
selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('About MSI-MSS','Upload Image','Predict')
    )
if selected_box == 'About MSI-MSS':
    st.write("A change that occurs in certain cells such as cancer cells in which the number of repeated DNA bases in a microsatellite a short, repeated sequence of DNA is different from what it was when the microsatellite was inherited. ") 
if selected_box == 'Upload Image':
    uploadFile = st.file_uploader(label="Upload image")
        
if selected_box == 'Predict':
    predict()
    
# Checking the Format of the page
if uploadFile is not None:
    global img
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    st.image(img)
    st.write("Image Uploaded Successfully")
else:
    st.write("Make sure you image is in JPG/PNG Format.")

# if st.button('predict'):
#     predict()
