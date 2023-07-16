import torch
from transformers import BertTokenizer, BertForSequenceClassification

class BertArticle:
    def __init__(self):
        self.model_name = 'bert-large-uncased'
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['not_growth', 'growth']
        self.model.to(self.device)

    def classify_theme(self, article):
        inputs = self.tokenizer.encode_plus(article, return_tensors='pt', truncation=True, padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label_id = torch.argmax(logits, dim=1).item()
            predicted_label = self.labels[predicted_label_id]
            growth_probability = probabilities[0][1].item()
            not_growth_probability = probabilities[0][0].item()

        print("Growth Probability:", growth_probability)
        print("Not Growth Probability:", not_growth_probability)

        return predicted_label