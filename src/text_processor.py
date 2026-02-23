import os
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def process_text(text):
    """
    1. load raw data
    2. Lowercase
    3. Tokenize
    4. remove stopwords
    5. Apply stemming
    6. Save processed docs
    """

    # 1. lowercase
    text = text.lower()

    try:
        nltk.data.find("corpora/stopwords")
    except nltk.downloader.DownloadError:
        nltk.download("stopwords")

    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.DownloadError:
        nltk.download("punkt")

    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. process tokens
    processed_tokens = []

    for token in tokens:
        # 4. Keep only alphabetic words : removing punctuation
        if token.isalpha():
            # 5. Remove Stopwords
            if token not in stop_words:
                # 6 Stemming
                stemmed = stemmer.stem(token)

                processed_tokens.append(stemmed)

    print(processed_tokens)
    return processed_tokens


def process_dataset(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    processed_docs = []

    for doc in documents:
        tokens = process_text(doc["text"])

        processed_doc = {
            "doc_id": doc["doc_id"],
            "tokens": tokens,
            "length": len(tokens),
        }

        processed_docs.append(processed_doc)

    os.makedirs("../data/processed", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False)


if __name__ == "__main__":
    process_dataset("data/raw/ag_news.json", "data/processed/processed_ag_news.json")


# process_text("A paragraph is a distinct, self-contained unit of writing comprising one or more sentences that develop a single, central idea")
