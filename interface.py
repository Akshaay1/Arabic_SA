import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Model choices
models = {
    'AraBERT': 'aubmindlab/bert-base-arabert',
    'GigaBERT': 'lanwuwei/GigaBERT-v3-Arabic-and-English',
    'llama':'Orkhan/llama-2-7b-absa'

    # Add additional models as needed
    
}

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Caching the model loading
@st.cache(allow_output_mutation=True)
def get_model(model_name):
    return load_model(model_name)

st.title('Sentiment Analysis App')

# Model selection
choice = st.selectbox('Choose your model:', list(models.keys()))

# Load selected model
model_pipeline = get_model(models[choice])

# Text input
user_input = st.text_area("Enter the text to analyze sentiment", "")

# Analyze sentiment
if st.button('Analyze'):
    if user_input:
        # Predicting the sentiment
        prediction = model_pipeline(user_input)
        # Displaying the sentiment
        st.write('Predicted Sentiment:', prediction[0]['label'])
    else:
        st.write('Please enter text to analyze.')
