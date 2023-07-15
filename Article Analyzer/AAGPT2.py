import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

class ArticleAnalyzer:
    def __init__(self, model_name="gpt2-medium"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = TFGPT2LMHeadModel.from_pretrained(model_name)

    def analyze_article(self, article):
        input_text = f"Is the article actionable? {article}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="tf")

        outputs = self.model.generate(input_ids, max_length=64, num_return_sequences=1)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        is_actionable = "Yes" if "Yes" in generated_text else "No"

        return is_actionable

    def summarize_article(self, article):
        input_text = f"summarize: {article}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="tf")

        outputs = self.model.generate(input_ids, max_length=150, num_return_sequences=1)

        generated_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_summary
