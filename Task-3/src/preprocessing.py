import pandas as pd
import re
import string

# Function to clean text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = text.strip()

    return text


def load_and_preprocess(path):
    # Load dataset
    df = pd.read_csv(path, encoding='latin-1')

    # Keep only required columns
    df = df[['v1', 'v2']]

    # Rename columns
    df.columns = ['label', 'message']

    # Convert labels to numeric
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Apply cleaning
    df['message'] = df['message'].apply(clean_text)

    return df

df = load_and_preprocess("data/spam.csv")