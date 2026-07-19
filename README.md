# Financial News Sentiment Prediction using Deep Learning & BERT

A sentiment classification system that labels finance-related tweets as **Bearish**, **Bullish**, or **Neutral** using deep learning (RNN, LSTM, GRU), with a Streamlit dashboard for live predictions.

## Project Structure

```
G_financial_sentiment_analysis/
├── sent_train.csv          # Training data (9,543 tweets)
├── sent_valid.csv          # Validation data (2,388 tweets)
├── jupyter_notebook.ipynb  # Full pipeline: preprocessing, model training, evaluation
├── app.py                  # Streamlit dashboard
├── model.pth                # Saved trained GRU weights (best-performing model)
├── word2idx.pkl              # Saved vocabulary (word → number mapping)
├── requirements.txt
└── README.md
```

## Dataset

- **Source:** Twitter Financial News Sentiment dataset
- **Labels:** 0 = Bearish, 1 = Bullish, 2 = Neutral
- **Size:** 9,543 training tweets, 2,388 validation tweets
- **Class distribution:** Neutral (~65%), Bullish (~20%), Bearish (~15%) — imbalanced, handled using class weights during training

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Caroleine03/Financial-News-Sentiment-Prediction-using-Deep-Learning
cd G_financial_sentiment_analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Data
`sent_train.csv` and `sent_valid.csv` are included in this repo, in the main project folder.

## How to Run

### A. Training pipeline (Jupyter Notebook)

Open and run `jupyter_notebook.ipynb` from top to bottom. This notebook covers:

1. **Data loading & exploration** — null/duplicate checks, class distribution analysis
2. **Text cleaning** — removes URLs, mentions, quotes, punctuation, extra spacing; preserves stock tickers (e.g. `$AAPL`) and negative numbers (e.g. `-5%`)
3. **Tokenization & vocabulary building** — vocabulary built **only from training data** (19,503 tokens including `<PAD>` and `<UNK>`) to avoid data leakage
4. **Encoding & padding** — sequences encoded to integers and padded/truncated to a fixed length of 20 tokens (95th percentile of training tweet lengths)
5. **Model training** — three architectures built and compared: Simple RNN, LSTM, GRU, each trained with **early stopping** (stops automatically once validation accuracy fails to improve for 3 consecutive epochs, restoring the best-seen weights)
6. **Evaluation** — accuracy, macro F1-score, confusion matrix, per-class precision/recall
7. **Model export** — best model (GRU) and vocabulary saved to `model.pth` and `word2idx.pkl`

### B. Streamlit dashboard

From the project folder, run:
```bash
python -m streamlit run app.py
```

This opens a browser window where you can:
- Enter a financial tweet or headline
- Click **Predict Sentiment** to see the predicted class
- View a bar chart of the model's confidence across all three classes

## Model Comparison

Three architectures were built and evaluated on the validation set, each using an Embedding layer (vocab size 19,503, embedding dimension 50) feeding into a recurrent layer (hidden size 64) and a final linear layer producing 3 class scores. Class-weighted cross-entropy loss was used to address label imbalance, and all three models were trained with **early stopping** (patience = 3 epochs) for a fair, consistent comparison rather than manually-picked epoch counts.

| Model      | Accuracy | Macro F1 | Bearish F1 | Bullish F1 | Neutral F1 |
|------------|----------|----------|------------|------------|------------|
| Simple RNN | 63.5%    | 0.283    | 0.05       | 0.02       | 0.78       |
| LSTM       | 68.8%    | 0.617    | 0.48       | 0.59       | 0.79       |
| **GRU**    | **74.5%**| **0.659**| 0.52       | 0.62       | 0.84       |

**GRU was selected as the deployed model**, achieving the best accuracy and macro F1, and the most balanced performance across all three classes under early stopping. Simple RNN's collapse under early stopping (heavily favoring the majority Neutral class) is consistent with its known difficulty retaining context across longer sequences (vanishing gradients), and highlights why early stopping — not just picking a "lucky" epoch count — is essential for a fair architecture comparison.

**Known limitations:**
- Predictions are sensitive to truncation for tweets longer than 20 tokens (the fixed padding length) — key sentiment-bearing words at the end of longer texts can be cut off, occasionally flipping the predicted class.
- Bearish is consistently the weakest-performing class across all three architectures, likely due to having the fewest training examples (1,442, vs. 6,178 for Neutral).
- Some words (e.g. "crash") appear in both Bearish- and Neutral-labeled training tweets in similar proportions, so confident Neutral predictions on short, ambiguous phrases containing such words reflect genuine label ambiguity in the training data rather than a model error.

## Requirements

See `requirements.txt`. Core dependencies:
- Python 3.10+
- torch
- pandas
- numpy
- scikit-learn
- streamlit

## Notes

- The vocabulary (`word2idx`) is built exclusively from the training set to prevent data leakage; unseen words at inference time (including in the validation set and Streamlit app) map to `<UNK>` (measured at 11.3% of validation tokens).
- `model.pth` and `word2idx.pkl` must be present in the same folder as `app.py` for the Streamlit dashboard to run.
- The optional BERT/FinBERT fine-tuning step was attempted but descoped due to CPU training time constraints; see the project report for details.