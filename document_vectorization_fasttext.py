import spacy
import fasttext
import transformers

import torch

import os
import re
import json

nlp = spacy.load("en_core_web_sm")
ft = fasttext.load_model("cc.en.300.bin")

tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert = transformers.AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

def preprocess_text(text: str):
    doc = nlp(text)
    
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

    return " ".join(tokens)

def get_articles():
    articles = []

    directory = os.path.join(os.path.dirname(__file__), "Krapivin2009", "docsutf8")

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        key_filename = filename.replace("txt", "key")
        key_path = os.path.join(os.path.dirname(__file__), "Krapivin2009", "keys", key_filename)

        document = ""
        keys = []

        with open(file_path, "r") as file:
            document = file.read()
            
        title = re.search(r"--T\n(.*?)\n--A", document, re.DOTALL).group(1)
        pure_abstract = re.search(r"--A\n(.*?)\n--B", document, re.DOTALL).group(1)
        
        abstract = preprocess_text(pure_abstract).replace("\n", " ")

        fasttext_vector = ft.get_sentence_vector(abstract).tolist()

        tokens = tokenizer(abstract, return_tensors="pt")

        with torch.no_grad():
            outputs = scibert(**tokens)

        scibert_vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy().tolist()

        with open(key_path, "r") as file:
            keys = [line.strip() for line in file.readlines()]

        articles.append({
            "id": filename.split(".")[0],
            "title": preprocess_text(title),
            "abstract": preprocess_text(abstract),
            "fasttext_vector": fasttext_vector,
            "scibert_vector": scibert_vector,
            "keys": keys
        })

    return articles

def save_as_json(articles):
    with open("articles.json", "w") as file:
        json.dump(articles, file, indent=4)

if __name__ == "__main__":
    articles = get_articles()
    save_as_json(articles)