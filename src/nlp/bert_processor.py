import torch
from transformers import AutoTokenizer, AutoModel

class BertProcessor:
    def __init__(self, path="./models/nlp/distilcamembert"):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()