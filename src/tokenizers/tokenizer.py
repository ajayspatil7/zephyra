import tiktoken
from .specialTokens import ZephyraTokens

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
    
    def get_sep_token_id(self):
        return self.tokenizer.encode(self.special_tokens.EOS)[0]

    def get_bos_token_id(self):
        return self.tokenizer.encode(self.special_tokens.BOS)[0]

    def get_eos_token_id(self):
        return self.tokenizer.encode(self.special_tokens.EOS)[0]

    def get_question_token_id(self):
        return self.tokenizer.encode(self.special_tokens.QUESTION)[0]

    def get_context_token_id(self):
        return self.tokenizer.encode(self.special_tokens.CONTEXT)[0]

    def encode_plus(self, text, text_pair=None, max_length=None, padding='max_length', truncation=True, return_tensors=None):
        encoded = self.encode(text)
        if text_pair:
            encoded += self.encode(text_pair)
        if max_length:
            if truncation:
                encoded = encoded[:max_length]
            if padding == 'max_length':
                encoded = encoded + [self.get_pad_token_id()] * (max_length - len(encoded))
        attention_mask = [1] * len(encoded)
        if return_tensors == 'pt':
            import torch
            encoded = torch.tensor(encoded)
            attention_mask = torch.tensor(attention_mask)
        return {
            'input_ids': encoded,
            'attention_mask': attention_mask
        }

    def batch_encode_plus(self, batch_text_or_text_pairs, max_length=None, padding=True, truncation=True, return_tensors=None):
        batch_outputs = [self.encode_plus(text, text_pair, max_length, padding, truncation, return_tensors)
                         for text, text_pair in batch_text_or_text_pairs]
        
        if return_tensors == 'pt':
            import torch
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
        # This is a basic implementation and might need to be adjusted based on tiktoken's behavior
        encoded = self.tokenizer.encode(text[:char_index])
        return len(encoded)