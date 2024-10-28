import streamlit as st
from model import NextWordModel
import torch
import torch.nn as nn
from helper import inference

st.title('Generate text !')
st.write('Provide a prompt, and the AI will generate the next lines based on your settings.')

with st.sidebar:
    st.header('Settings')

    embedding_len = st.selectbox(
        'Select Embedding Length',
        options=[64,256],
        help='Choose the length of word embeddings.'
    )

    context_len = st.selectbox(
        'Select Context Length',
        options=[4,8],
        help='Choose the number of words model considers to generate the text'
    )

    activation_fn = st.selectbox(
        'Select the activation function',
        options=['relu','tanh'],
        help='Choose the activation function for the model'
    )

    num_lines = st.slider(
        'Select the number of lines to generate',
        min_value = 1,
        max_value = 10,
        value = 3,
        help = 'Choose the number of lines that you want to generate'
    )

st.subheader('Enter your prompt')
prompt = st.text_area('Text Prompt',placeholder='Enter your text prompt here....')

output_placeholder = st.empty()
if st.button('Generate Text'):
    output_placeholder.write('Generating Text....')

    output_placeholder.write(inference(embedding_len,context_len,num_lines,prompt,activation_fn))