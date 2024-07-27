import tiktoken
from specialTokens import ZephyraTokens

# class ZephyraTokenizer:
#     def __init__(self, model_name="cl100k_base"):
#         self.tokenizer = tiktoken.get_encoding(model_name)
#         self.special_tokens = ZephyraTokens()
#         self.addSpecialZephyraTokens()
#         self.vocab_size = self.tokenizer.n_vocab


#     def addSpecialZephyraTokens(self):
#         for token in vars(self.special_tokens).values():
#             if token not in self.tokenizer.encode(token):
#                 self.tokenizer.add_special_tokens([token])

#     def encode(self, text, add_special_tokens=True):
#         if add_special_tokens:
#             text = f"{self.special_tokens.BOS} {text} {self.special_tokens.EOS}"
#         return self.tokenizer.encode(text)

#     def decode(self, tokens):
#         return self.tokenizer.decode(tokens)

#     def getVocabSize(self):
#         return self.vocab_size

#     def getPaddingTokenId(self):
#         return self.tokenizer.encode(self.special_tokens.PAD)[0]

#     def encodeCOQA_sample(self, context, questions, answers, rationales):
#         """
#         Encode a full CoQA example including context, questions, answers, and rationales.
#         """
#         encoded = self.encode(f"{self.special_tokens.CONTEXT} {context}")
        
#         for q, a, r in zip(questions, answers, rationales):
#             encoded += self.encode(f"{self.special_tokens.QUESTION} {q}")
#             encoded += self.encode(f"{self.special_tokens.ANSWER} {a}")
#             encoded += self.encode(f"{self.special_tokens.RATIONALE_START} {r} {self.special_tokens.RATIONALE_END}")
        
#         return encoded

#     def find_rationale_span(self, context_tokens, rationale_tokens):
#         """
#         Find the start and end indices of the rationale span in the context.
#         """
#         for i in range(len(context_tokens) - len(rationale_tokens) + 1):
#             if context_tokens[i:i+len(rationale_tokens)] == rationale_tokens:
#                 return i, i + len(rationale_tokens)
#         return -1, -1  # If not found

#     def encode_with_rationale_positions(self, context, question, answer, rationale):
#         """
#         Encode a CoQA example and return token IDs along with rationale start and end positions.
#         """
#         context_tokens = self.encode(context)
#         question_tokens = self.encode(question)
#         answer_tokens = self.encode(answer)
#         rationale_tokens = self.encode(rationale)

#         rationale_start, rationale_end = self.find_rationale_span(context_tokens, rationale_tokens)

#         full_encoding = (
#             self.encode(self.special_tokens.CONTEXT) +
#             context_tokens +
#             self.encode(self.special_tokens.QUESTION) +
#             question_tokens +
#             self.encode(self.special_tokens.ANSWER) +
#             answer_tokens
#         )

#         return full_encoding, rationale_start, rationale_end
    

class ZephyraTokeniser:
    def __init__(self, vocab=None):
        # Define special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<mask>": 4,
            "<|user|>": 5,  # USER
            "<|assistant|>": 6,  # ASSISTANT
            "<|system|>": 7,  # SYSTEM
            "<|context|>": 8,  # CONTEXT
            "<|question|>": 9,  # QUESTION
            "<|answer|>": 10, # ANSWER
            "<|rationale_start|>": 11, # RATIONALE_START
            "<|rationale_end|>": 12, # RATIONALE_END
            
        }

        # Initialize vocabulary
        self.vocab = vocab if vocab else self.special_tokens
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def add_tokens(self, new_tokens):
        for token in new_tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token

    def tokenize(self, text):
        tokens = text.split()
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        return token_ids

    def detokenize(self, token_ids):
        tokens = [self.inv_vocab.get(token_id, "<unk>") for token_id in token_ids]
        return " ".join(tokens)

    def encode(self, text, max_length=None, add_special_tokens=True):
        token_ids = self.tokenize(text)
        if add_special_tokens:
            token_ids = [self.vocab["<s>"]] + token_ids + [self.vocab["</s>"]]
        if max_length:
            token_ids = token_ids[:max_length]
            token_ids += [self.vocab["<pad>"]] * (max_length - len(token_ids))
        return token_ids

    def decode(self, token_ids):
        if token_ids[0] == self.vocab["<s>"]:
            token_ids = token_ids[1:]
        if token_ids[-1] == self.vocab["</s>"]:
            token_ids = token_ids[:-1]
        return self.detokenize(token_ids)

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def convert_ids_to_tokens(self, token_ids):
        return [self.inv_vocab.get(token_id, "<unk>") for token_id in token_ids]

    def pad_sequence(self, sequences, max_length=None):
        if not max_length:
            max_length = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            seq = seq[:max_length]
            seq += [self.vocab["<pad>"]] * (max_length - len(seq))
            padded_sequences.append(seq)
        return padded_sequences

# Example usage
tokenizer = ZephyraTokeniser()
text = "Hello, how are you?"
encoded = tokenizer.encode(text, max_length=10)
decoded = tokenizer.decode(encoded)

print("Original Text:", text)
print("Encoded Text:", encoded)
print("Decoded Text:", decoded)
