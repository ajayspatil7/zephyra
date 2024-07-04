import torch
import torch.nn as nn
import math
from experiments.cpp import lxa

class SelfAttention(nn.Module):


    def __init__(self, embed_dim):
        
        super().__init__()
        self.embed_dim = embed_dim
        
        # Usig nn.Linear for a basic Linear Layer with embed_dim as the input size and embed_dim as the output size
        # Representing qkv as a linear layer
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Scale factor for scaling the attentions down to manage the magnitude of the attention scores
        self.scale = math.sqrt(embed_dim)


    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        """
        (Step 1) : Calculating attention scores
            
            q, k, v are the Query Key Value Vectors

            :q -> x * W^q
            :k -> x * W^k
            :v -> x * W^v
            :Dk -> dimension(k)

        (Step 2) : Calculating attention score : 
            
            The dot product of ( Q * transpose(K) )

        (Step 3) : Calculating the final attention using SDPA ( Scalar Dot Product Attention )
            
            Attention(Q, K, V) = softmax( attention / sqrt(Dk) ) * v
        
        Example : Let's consider a input vector with 3 embedings [x1, x2, x3]
        
        1) Embedding tokens
            X = [x1, x2, x3, x4]
        
        2) Transforming X into QKV
            :q -> x * W^q -> Assume q = [q1, q2, q3]
            :k -> x * W^k -> Assume k = [k1, k2, k3]
            :v -> x * W^v -> Assume v = [v1, v2, v3]

        3) Calculating Attention Score
            score(i, j) = q[i] * k[j]
            After applying scaling and softmaxing it we get Attention scores matrix

        4) Attention Score matrix
            Z (i, j) represents that, attention score of token i on token j

            [ Z(1,1), Z(1,2), Z(1,3)  ]
            [ Z(2,1), Z(2,2), Z(2,3)  ]
            [ Z(3,1), Z(3,2), Z(3,3)  ]

        5) Final score
            Multiplying the attention scores by value vectors to get the final output
            So,
                Output(i) = Sum(Z(i, j) * v(j))
        """

        # TODO [LOW] : Find a way to detach the below matmul to run on a GPU
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        print("Self Attention strassen is running...")
        return lxa.strassen(attn_probs, v)
        # return torch.matmul(attn_probs, v)


class MultiHeadAttention(nn.Module):

    """
    Multi Head Attention is multiple Attention blocks; which means multiple set of Q, K, V vectors.
    This will help to visit different words in a sentence or sequence in parallel,
    """


    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.proj(out)
    
"""<-----------End of attention.py----------->"""

