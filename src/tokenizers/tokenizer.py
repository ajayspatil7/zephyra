import tiktoken
from .specialTokens import ZephyraTokens

import tiktoken
import torch

class ZephyraTokens:
    PAD = "<pad>"
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"
    MASK = "<mask>"
    USER = "<user>"
    ASSISTANT = "<assistant>"
    SYSTEM = "<system>"
    CONTEXT = "<context>"
    QUESTION = "<question>"
    ANSWER = "<answer>"
    RATIONALE_START = "<rationale_start>"
    RATIONALE_END = "<rationale_end>"

class ZephyraTokenizer:
    def __init__(self, model_name="cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.special_tokens = ZephyraTokens()
        self.add_special_tokens()
        self.vocab_size = self.tokenizer.n_vocab

    def add_special_tokens(self):
        for token in vars(self.special_tokens).values():
            if token not in self.tokenizer.encode(token):
                self.tokenizer.add_special_tokens([token])

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            text = f"{self.special_tokens.BOS} {text} {self.special_tokens.EOS}"
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def getVocabSize(self):
        return self.vocab_size

    def get_pad_token_id(self):
        return self.tokenizer.encode(self.special_tokens.PAD)[0]

    def encode_plus(self, question, context, max_length=None, padding='max_length', truncation=True, return_tensors=None):
        encoded_question = self.encode(f"{self.special_tokens.QUESTION} {question}")
        encoded_context = self.encode(f"{self.special_tokens.CONTEXT} {context}")
        
        encoded = encoded_question + encoded_context

        if max_length and truncation:
            encoded = encoded[:max_length]

        attention_mask = [1] * len(encoded)

        if padding == 'max_length' and max_length:
            padding_length = max_length - len(encoded)
            encoded = encoded + [self.get_pad_token_id()] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        if return_tensors == 'pt':
            encoded = torch.tensor(encoded)
            attention_mask = torch.tensor(attention_mask)

        return {
            'input_ids': encoded,
            'attention_mask': attention_mask
        }

    def batch_encode_plus(self, batch_questions_and_contexts, max_length=None, padding=True, truncation=True, return_tensors=None):
        batch_outputs = [
            self.encode_plus(question, context, max_length, padding, truncation, return_tensors)
            for question, context in batch_questions_and_contexts
        ]

        if return_tensors == 'pt':
            return {
                'input_ids': torch.stack([output['input_ids'] for output in batch_outputs]),
                'attention_mask': torch.stack([output['attention_mask'] for output in batch_outputs])
            }
        else:
            return {
                'input_ids': [output['input_ids'] for output in batch_outputs],
                'attention_mask': [output['attention_mask'] for output in batch_outputs]
            }

    def char_to_token(self, text, char_index):
        encoded = self.tokenizer.encode(text[:char_index])
        return len(encoded)