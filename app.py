import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
from transformers import RobertaTokenizer, TFRobertaModel
import plotly.graph_objs as go
from textblob import TextBlob
import re
from sklearn.preprocessing import StandardScaler
import gdown

# Download the model from Google Drive
file_id = '1L69so1We0E67X0bRHSOkEsvPYeSIEnr6'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
output = 'trained_model.h5'
gdown.download(url, output, quiet=False)


# Define the company tickers and names
companies_to_focus = {
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'AAPL': 'Apple'
}

# Initialize tokenizer and BERT model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = TFRobertaModel.from_pretrained('roberta-base')

# Define lookback window
look_back = 5

# Register the custom layer for deserialization
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Load the trained model with the custom layer
custom_objects = {'TransformerBlock': TransformerBlock}
model = tf.keras.models.load_model('trained_model.h5', custom_objects=custom_objects)

# Function to preprocess text for BERT embeddings
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower().strip()
    tokens = text.split()
    return ' '.join(tokens)

# Function to get BERT embeddings
def get_bert_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token's embedding

# Function to predict future prices
def predict_prices(news_headlines, look_back_window, bert_dim, combined_dim, scaler, target_scalers):
    processed_articles = [preprocess_text(article) for article in news_headlines]
    bert_embeddings = [get_bert_embeddings([article], tokenizer, bert_model)[0] for article in processed_articles]

    # Ensure the embeddings have the correct shape
    bert_embeddings = bert_embeddings[-look_back_window:]
    if len(bert_embeddings) < look_back_window:
        # Pad the embeddings if there are not enough look-back days
        padding = [np.zeros((bert_dim,)) for _ in range(look_back_window - len(bert_embeddings))]
        bert_embeddings = padding + bert_embeddings

    if combined_dim > bert_dim:
        # Combine with dummy data to match the expected combined dimension
        dummy_data = np.zeros((look_back_window, combined_dim - bert_dim))
        combined_features = np.concatenate([bert_embeddings, dummy_data], axis=-1)
    else:
        combined_features = np.array(bert_embeddings)

    # Reshape for model input
    combined_features = np.array(combined_features).reshape(1, look_back_window, -1)

    # Scale the combined features
    combined_features_scaled = scaler.transform(combined_features.reshape(-1, combined_features.shape[-1]))
    combined_features_scaled = combined_features_scaled.reshape(combined_features.shape)

    # Predict using the loaded model
    predictions_scaled = model.predict(combined_features_scaled)

    # Inverse transform the predictions to get the original scale
    predictions = {ticker: target_scalers[ticker].inverse_transform(predictions_scaled[ticker]) for ticker in companies_to_focus.keys()}
    return predictions

# Function to perform sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to fetch fundamental data for a company
def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    fundamentals = stock.info
    return {
        "PE_Ratio": fundamentals.get("trailingPE", np.nan),
        "EPS": fundamentals.get("trailingEps", np.nan),
        "Revenue": fundamentals.get("totalRevenue", np.nan),
        "Market_Cap": fundamentals.get("marketCap", np.nan)
    }

# Load the dataset
news_data = pd.read_csv('final_dataset_without_last_column.csv')
news_data['Date'] = pd.to_datetime(news_data['Date'])
news_data['Processed_Article'] = news_data['News_Article'].apply(preprocess_text)
news_data['Sentiment'] = news_data['Processed_Article'].apply(get_sentiment)

# Define dimensions
bert_dim = bert_model.config.hidden_size  # typically 768 for BERT models
combined_dim = 1543  # Update this to the correct combined dimension

# Initialize scalers
scaler = StandardScaler()
target_scalers = {ticker: StandardScaler() for ticker in companies_to_focus.keys()}

# Simulate fitting scalers with initial data
def fit_scalers():
    combined_features_list = []
    targets_list = []

    for ticker in companies_to_focus.keys():
        # Simulate fetching stock data
        stock_data = yf.download(ticker, start='2021-01-01', end='2021-12-31')
        stock_data.reset_index(inplace=True)

        # Fetch moving averages
        ma50 = stock_data['Close'].rolling(window=50).mean()
        ma200 = stock_data['Close'].rolling(window=200).mean()

        stock_data['MA50'] = ma50
        stock_data['MA200'] = ma200

        # Generate dummy combined features matching the expected combined dimension
        num_samples = len(stock_data)
        dummy_bert_features = np.zeros((num_samples, 768))  # Example BERT feature size
        dummy_other_features = np.zeros((num_samples, combined_dim - 768))
        combined_features = np.hstack([dummy_bert_features, dummy_other_features])

        combined_features_list.append(combined_features)
        targets_list.append(stock_data['Close'].values)

    combined_features_array = np.concatenate(combined_features_list, axis=0)
    targets_array = np.concatenate(targets_list, axis=0).reshape(-1, len(companies_to_focus))

    scaler.fit(combined_features_array)
    for i, ticker in enumerate(companies_to_focus.keys()):
        target_scalers[ticker].fit(targets_array[:, i].reshape(-1, 1))

fit_scalers()

# Streamlit App Layout
st.title("Stock Price Prediction App")

# Sidebar Description
st.sidebar.title("About the App")
st.sidebar.markdown("""
This application predicts the stock prices of major companies using news headlines and sentiment analysis.
We utilize BERT embeddings, technical indicators, and fundamental data for robust predictions.
""")

st.sidebar.title("Model Description")
st.sidebar.markdown("""
Our model leverages a transformer-based architecture with BERT embeddings to capture the semantic meaning of news articles.
We incorporate technical indicators, such as moving averages, and fundamental data to improve the prediction accuracy.
""")

# Fetch data
today = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = today

# Get today's news headlines
todays_news = news_data[news_data['Date'] == today].head(6)  # Display at most 6 headlines

# Get stock data and predictions
stock_data_dict = {}
fundamental_data_dict = {}
for ticker in companies_to_focus:
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the Date column is present
    stock_data.reset_index(inplace=True)

    # Fetch moving averages
    ma50 = stock_data['Close'].rolling(window=50).mean()
    ma200 = stock_data['Close'].rolling(window=200).mean()

    stock_data['MA50'] = ma50
    stock_data['MA200'] = ma200

    stock_data_dict[ticker] = stock_data
    fundamental_data_dict[ticker] = fetch_fundamental_data(ticker)

# Call predict_prices once
news_headlines = todays_news['Processed_Article'].tolist()
predictions = predict_prices(news_headlines, look_back, bert_dim, combined_dim, scaler, target_scalers)
predictions_dict = {ticker: predictions[ticker] for ticker in companies_to_focus}

# Display predicted prices for tomorrow
st.subheader("Predicted Prices for Tomorrow")
for ticker, company in companies_to_focus.items():
    today_price = stock_data_dict[ticker]['Close'].values[-1]
    predicted_price = predictions_dict[ticker][0][0]
    arrow = "⬆️" if predicted_price > today_price else "⬇️"
    color = "green" if predicted_price > today_price else "red"
    st.markdown(f"**{company} ({ticker}):** {predicted_price:.2f} {arrow}", unsafe_allow_html=True)

# Display news headlines with sentiment in a table
st.subheader("Latest News")
news_table = todays_news[['News_Article', 'Sentiment']].copy()
news_table.columns = ['News Article', 'Sentiment']
news_table['Sentiment'] = news_table['Sentiment'].apply(lambda x: f"<span style='color:{'green' if x > 0 else 'red'}'>{x:.2f}</span>")
st.write(news_table.to_html(escape=False, index=False), unsafe_allow_html=True)

# Manual prediction input
st.subheader("Manual Prediction Input")
manual_news_headlines = st.text_area("Enter News Headlines", "").split('\n')

if st.button("Predict Manually"):
    if manual_news_headlines:
        manual_predictions = predict_prices(manual_news_headlines, look_back, bert_dim, combined_dim, scaler, target_scalers)
        for ticker, company in companies_to_focus.items():
            manual_prediction = manual_predictions[ticker][0][0]
            today_price = stock_data_dict[ticker]['Close'].values[-1]
            arrow = "⬆️" if manual_prediction > today_price else "⬇️"
            st.write(f"Predicted price for {company} ({ticker}): {manual_prediction:.2f} {arrow}")

# Display stock price charts with actual, predicted prices, and technical indicators
for ticker, company in companies_to_focus.items():
    stock_data = stock_data_dict[ticker]
    fig = go.Figure()

    # Add actual stock price trace
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Actual Close'))

    # Add predicted price trace
    predicted_price = predictions_dict[ticker][0][0]
    predicted_date = stock_data['Date'].iloc[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(x=[predicted_date], y=[predicted_price], mode='markers', name='Predicted Close', marker=dict(color='red', size=10)))

    # Add moving average traces
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MA50'], mode='lines', name='MA50'))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MA200'], mode='lines', name='MA200'))

    # Customize the layout
    fig.update_layout(
        title=f'{company} ({ticker}) Stock Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True
    )

    # Display the chart
    st.plotly_chart(fig)

    # Display fundamental data
    st.subheader(f"{company} ({ticker}) Fundamentals")
    fundamentals = fundamental_data_dict[ticker]
    st.markdown(f"""
    - **PE Ratio**: {fundamentals['PE_Ratio']}
    - **EPS**: {fundamentals['EPS']}
    - **Revenue**: {fundamentals['Revenue']}
    - **Market Cap**: {fundamentals['Market_Cap']}
    """)

# "See More" Section
st.subheader("See More")
st.markdown("""
We also trained a model that uses Topic Modelling, TF-IDF, and Named Entity Recognition (NER) as features.
For more details, check out our [GitHub Repository](https://github.com/KlimLe/ML4B-Stock-Prediction/tree/main).
""")
