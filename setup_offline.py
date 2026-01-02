from transformers import AutoTokenizer, AutoModel
import os

def setup():
    model_name = "cmarkea/distilcamembert-base"
    save_path = "./models/nlp/distilcamembert"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Downloading French DistilCamemBERT weights...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"Offline weights ready in {save_path}")

if __name__ == "__main__":
    setup()