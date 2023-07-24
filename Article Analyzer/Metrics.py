import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, BertForMaskedLM
import torch

class SummaryMetrics:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _longest_common_subsequence(self, reference_tokens, candidate_tokens):
        reference_set = set(reference_tokens)
        common_tokens = [token for token in candidate_tokens if token in reference_set]
        return len(common_tokens)

    def rouge_n(self, reference, candidate, n):
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        ngram_reference = set(nltk.ngrams(reference_tokens, n))
        ngram_candidate = set(nltk.ngrams(candidate_tokens, n))
        intersection = ngram_reference.intersection(ngram_candidate)
        return len(intersection) / len(ngram_reference)

    def rouge_l(self, reference, candidate):
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())

        # Calculate the LCS length
        lcs_length = self._longest_common_subsequence(reference_tokens, candidate_tokens)

        precision = lcs_length / max(len(reference_tokens), 1)
        recall = lcs_length / max(len(candidate_tokens), 1)

        # Handling division by zero for precision or recall
        if precision == 0 or recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score

    def rouge_s(self, reference, candidate):
        reference_tokens = set(nltk.word_tokenize(reference.lower()))
        candidate_tokens = set(nltk.word_tokenize(candidate.lower()))
        intersection = len(reference_tokens.intersection(candidate_tokens))
        return intersection / max(len(reference_tokens), len(candidate_tokens), 1)

    def bert_score(self, reference, candidate):
        inputs = self.tokenizer(candidate, return_tensors='pt', padding=True, truncation=True)
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            masked_lm_logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(masked_lm_logits.view(-1, self.tokenizer.vocab_size), inputs['input_ids'].view(-1))
        return torch.exp(-masked_lm_loss)

    def quip_score(self, reference, candidate):
        smoothing_function = SmoothingFunction().method1
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)


