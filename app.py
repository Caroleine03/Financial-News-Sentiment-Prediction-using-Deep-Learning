import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import torch.nn.functional as F

def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub('"', "", text)
    text = re.sub("[.,:;!?]", "", text)
    text = re.sub(" - ", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text

def encode_tokens(tokens):
    tweet = []
    for i in tokens:
        tweet.append(word2idx.get(i, 1))
    return tweet

def pad_sequence(tokens, max_len=20):
    tokens = tokens.copy()
    if len(tokens) < max_len:
        for i in range(max_len - len(tokens)):
            tokens.append(0)
        return tokens
    elif len(tokens) > max_len:
        return tokens[:max_len]
    else:
        return tokens
    
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)
        linear = self.fc(hidden)
        return linear
    
vocab_size = 19503
embedding_dim = 50
hidden_size = 64
model = SentimentGRU(vocab_size, embedding_dim, hidden_size)

with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

model.load_state_dict(torch.load('model.pth'))

model.eval()

st.title("Financial Tweet Sentiment Predictor")

user_input = st.text_area("Enter a financial tweet or headline:")

if st.button("Predict Sentiment"):
    cleaned = clean_text(user_input)
    tokens = cleaned.split()
    encoded = encode_tokens(tokens)
    padded = pad_sequence(encoded)
    input_tensor = torch.tensor([padded])
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
    prediction = output.argmax(dim=1)
    predicted_class = prediction.item()
    label_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    sentiment = label_map[predicted_class]
    st.write(f"Predicted Sentiment: {sentiment}")
    prob_values = probs[0].tolist()
    prob_dict = {"Bearish": prob_values[0], "Bullish": prob_values[1], "Neutral": prob_values[2]}
    st.bar_chart(prob_dict)




    



 



    
