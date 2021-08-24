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
    st.write("Microsatellite instability (MSI) is the condition of genetic hypermutability 
             (predisposition to mutation) that results from impaired DNA mismatch repair (MMR). 
             The presence of MSI represents phenotypic evidence that MMR is not functioning normally.
             MMR corrects errors that spontaneously occur during DNA replication, such as single base mismatches or short insertions and deletions. 
             The proteins involved in MMR correct polymerase errors by forming a complex that binds to the mismatched section of DNA, excises the error, 
             and inserts the correct sequence in its place.[1] Cells with abnormally functioning MMR are unable to correct errors that occur during 
             DNA replication and consequently accumulate errors. This causes the creation of novel microsatellite fragments. 
             Polymerase chain reaction-based assays can reveal these novel microsatellites and provide evidence for the presence of MSI.
             Microsatellites are repeated sequences of DNA. These sequences can be made of repeating units of one to six base pairs in length. 
             Although the length of these microsatellites is highly variable from person to person and contributes to the individual 
             DNA "fingerprint", each individual has microsatellites of a set length. 
             The most common microsatellite in humans is a dinucleotide repeat of the nucleotides C and A, 
             which occurs tens of thousands of times across the genome. Microsatellites are also known as simple sequence repeats (SSRs).") 
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
