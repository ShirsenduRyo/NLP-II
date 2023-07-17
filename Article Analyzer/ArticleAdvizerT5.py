import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration, T5Config

class T5ArticleAdvizer:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.config = T5Config.from_pretrained(model_name)
        self.model = TFT5ForConditionalGeneration.from_pretrained(model_name, \
                                                                  config=self.config)
        
    def answer_question(self, news_article, question):
        inputs = self.tokenizer.encode_plus(f"question: {question} \
                    context: {news_article}", return_tensors="tf", padding="max_length", \
                    truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def summarize_article(self, news_article):
        inputs = self.tokenizer.encode_plus(f"summarize: {news_article}", return_tensors="tf", \
                                            padding="max_length", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, \
                                      max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary