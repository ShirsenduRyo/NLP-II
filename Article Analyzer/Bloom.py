import transformers
import torch

class BloomArticleClassifier:
    def __init__(self, model_name='bloom'):
        self.model_name = model_name
        self.model = transformers.AutoModel.from_pretrained(self.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['growth', 'not_growth']
        self.model.to(self.device)

    def classify_theme(self, article):
        encoded_article = self.tokenizer(article, return_tensors='pt', truncation=True, padding=True)
        encoded_article = encoded_article.to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_article)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label_id = torch.argmax(logits, dim=1).item()
            predicted_label = self.labels[predicted_label_id]
            growth_probability = probabilities[0][1].item()
            not_growth_probability = probabilities[0][0].item()

        print("Growth Probability:", growth_probability)
        print("Not Growth Probability:", not_growth_probability)

        return predicted_label

    def summarize(self, article):
        encoded_article = self.tokenizer(article, return_tensors='pt', truncation=True, padding=True)
        encoded_article = encoded_article.to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_article)
            summary = outputs.summary

        return summary

    def extract_key_subjects(self, article):
        encoded_article = self.tokenizer(article, return_tensors='pt', truncation=True, padding=True)
        encoded_article = encoded_article.to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_article)
            key_subjects = outputs.key_subjects

        return key_subjects