import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import ElectraTokenizer, ElectraForSequenceClassification



class GPT2ArticleClassifier:
    def __init__(self):
        self.model_name = 'gpt2'
        self.model = GPT2ForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['adverse', 'growth']
        self.model.to(self.device)

    def summarize_article(self, article, max_length=100):
        model_name = 'gpt2'
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        inputs = tokenizer.encode(article, return_tensors='pt')
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_length=100, num_return_sequences=1)# Need To sort this
            summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summarized_text

    def classify_growth(self, article):
        inputs = self.tokenizer(article, return_tensors='pt')
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label_id = torch.argmax(logits, dim=1).item()
            predicted_label = self.labels[predicted_label_id]
            predicted_probability = probabilities[0][predicted_label_id].item()

        return predicted_label, predicted_probability
    

class RobertaArticleClassifier:
    def __init__(self):
        self.model_name = 'roberta-base'
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['adverse', 'growth']
        self.model.to(self.device)

    def classify_growth(self, article):
        inputs = self.tokenizer.encode_plus(article, return_tensors='pt', truncation=True, padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label_id = torch.argmax(logits, dim=1).item()
            predicted_label = self.labels[predicted_label_id]
            predicted_probability = probabilities[0][predicted_label_id].item()

        return predicted_label, predicted_probability

class BertArticleClassifier:
    def __init__(self, bert = 'bert-large-uncased'):
        self.model_name = bert #'bert-large-uncased'/'bert-large-cased'
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['adverse', 'growth']
        self.model.to(self.device)

    def classify_growth(self, article):
        inputs = self.tokenizer.encode_plus(article, return_tensors='pt', truncation=True, padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label_id = torch.argmax(logits, dim=1).item()
            predicted_label = self.labels[predicted_label_id]
            predicted_probability = probabilities[0][predicted_label_id].item()
            
        return predicted_label, predicted_probability


class DistilBertArticleClassifier:
    def __init__(self, distilbert = 'distilbert-base-uncased'):
        self.model_name = distilbert #'distilbert-base-uncased'/'distilbert-base-cased'
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['adverse', 'growth']
        self.model.to(self.device)

    def classify_growth(self, article):
        inputs = self.tokenizer.encode_plus(article, return_tensors='pt', truncation=True, padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label_id = torch.argmax(logits, dim=1).item()
            predicted_label = self.labels[predicted_label_id]
            predicted_probability = probabilities[0][predicted_label_id].item()

        return predicted_label, predicted_probability
    
class BartArticleGenerator:
    def __init__(self):
        self.model_name = 'facebook/bart-large-cnn'
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
    
    def summarize_article(self, article):
        inputs = self.tokenizer([article], max_length=1024, truncation=True, return_tensors='pt')
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        summarize_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summarize_text
    

class ElectraArticleClassifier:
    def __init__(self):
        self.model_name = 'google/electra-base-discriminator'
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        self.model = ElectraForSequenceClassification.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['adverse', 'growth']
        self.model.to(self.device)

    def classify_growth(self, article):
        inputs = self.tokenizer.encode_plus(article, return_tensors='pt', truncation=True, padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label_id = torch.argmax(logits, dim=1).item()
            predicted_label = self.labels[predicted_label_id]
            predicted_probability = probabilities[0][predicted_label_id].item()
#             growth_probability = probabilities[0][1].item()
#             adverse_probability = probabilities[0][0].item()

#         print("Growth Probability:", growth_probability)
#         print("adverse Probability:", adverse_probability)

        return predicted_label, predicted_probability