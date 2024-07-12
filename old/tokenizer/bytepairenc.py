import re
from collections import defaultdict

"""
Scratch implementation of Byte Pair Encoding Algorithm for effective tokenisation
"""

def bytePairEncoding(data: str, itter: int) -> dict:
    """
    :data -> The large text we want to tokenise
    :itter -> Basically 'Epochs'

    ::returns dict with the word and their respective token
    """

    vocab = getVocab(data)
    for i in range(itter):
        pairs = getPair(vocab)
        best = max(pairs, key=pairs.get)
        vocab = mergeVocab(best, vocab)
    
    return vocab


def getVocab(data: str):
    """
    :data -> Just a large string
    ::returns words in the large string and their occurance count
    """

    vocab = defaultdict(int)
    for l in data:
        for w in l.split():
            vocab[' '.join(list(w)) + '</w>'] += 1
    
    return vocab


def getPair(vocab: dict) -> dict:
    
    """
    :vocab -> Takes vocabulary (dict) 
    ::returns the possible pair based on frequent occurance
    """
    
    pairs = defaultdict(int)
    for w, o in vocab.items():
        symbols = w.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += o

    return pairs


def mergeVocab(pair: str, vocab: dict) -> dict:

    """
    :pair -> pair of characters
    :vocab -> a possible word and occurance count; a dictonary
    ::returns new vocab with word and where the repeated occurance occured
    """
    
    vocab_out = {}
    tokens = re.escape(' '.join(pair))
    expression = re.compile(r'(?<!\S)' + tokens + r'(?!\S)')

    for word in vocab:
        return_word = expression.sub(''.join(pair), word)
        vocab_out[return_word] = vocab[word]

    return vocab_out


# Example of a vocab as dic
data = """
|system|>In recent times, social media has become a popular medium for many election campaigns. It not only allows candidates to reach out to a large section of the electorate, it is also a potent medium for people to express their opinion on the proposed policies and promises of candidates. Analyzing social media data is challenging as the text can be noisy, sparse and even multilingual. In addition, the information may not be completely trustworthy, particularly in the presence of propaganda, promotions and rumors. In this paper we describe our work for analyzing election campaigns using social media data. Using data from the 2012 US presidential elections and the 2013 Philippines General elections, we provide detailed experiments on our methods that use granger causality to identify topics that were most “causal” for public opinion and which in turn, give an interpretable insight into “elections topics” that were most important. Our system was deployed by the largest media organization in the Philippines during the 2013 General elections and using our work, the media house able to identify and report news stories much faster than competitors and reported higher TRP ratings during the election.</s>
<|user|>social media has become a popular medium for what ?</s>
<|assistant|>many election campaigns</s>

<|system|>In recent times, social media has become a popular medium for many election campaigns. It not only allows candidates to reach out to a large section of the electorate, it is also a potent medium for people to express their opinion on the proposed policies and promises of candidates. Analyzing social media data is challenging as the text can be noisy, sparse and even multilingual. In addition, the information may not be completely trustworthy, particularly in the presence of propaganda, promotions and rumors. In this paper we describe our work for analyzing election campaigns using social media data. Using data from the 2012 US presidential elections and the 2013 Philippines General elections, we provide detailed experiments on our methods that use granger causality to identify topics that were most “causal” for public opinion and which in turn, give an interpretable insight into “elections topics” that were most important. Our system was deployed by the largest media organization in the Philippines during the 2013 General elections and using our work, the media house able to identify and report news stories much faster than competitors and reported higher TRP ratings during the election.</s>
<|user|>social media has become a popular medium for what ?</s>
<|assistant|>many election campaigns</s>

<|system|>In recent times, social media has become a popular medium for many election campaigns. It not only allows candidates to reach out to a large section of the electorate, it is also a potent medium for people to express their opinion on the proposed policies and promises of candidates. Analyzing social media data is challenging as the text can be noisy, sparse and even multilingual. In addition, the information may not be completely trustworthy, particularly in the presence of propaganda, promotions and rumors. In this paper we describe our work for analyzing election campaigns using social media data. Using data from the 2012 US presidential elections and the 2013 Philippines General elections, we provide detailed experiments on our methods that use granger causality to identify topics that were most “causal” for public opinion and which in turn, give an interpretable insight into “elections topics” that were most important. Our system was deployed by the largest media organization in the Philippines during the 2013 General elections and using our work, the media house able to identify and report news stories much faster than competitors and reported higher TRP ratings during the election.</s>
<|user|>social media has become a popular medium for what ?</s>
<|assistant|>many election campaigns</s>

<|system|>In recent times, social media has become a popular medium for many election campaigns. It not only allows candidates to reach out to a large section of the electorate, it is also a potent medium for people to express their opinion on the proposed policies and promises of candidates. Analyzing social media data is challenging as the text can be noisy, sparse and even multilingual. In addition, the information may not be completely trustworthy, particularly in the presence of propaganda, promotions and rumors. In this paper we describe our work for analyzing election campaigns using social media data. Using data from the 2012 US presidential elections and the 2013 Philippines General elections, we provide detailed experiments on our methods that use granger causality to identify topics that were most “causal” for public opinion and which in turn, give an interpretable insight into “elections topics” that were most important. Our system was deployed by the largest media organization in the Philippines during the 2013 General elections and using our work, the media house able to identify and report news stories much faster than competitors and reported higher TRP ratings during the election.</s>
<|user|>social media has become a popular medium for what ?</s>
<|assistant|>many election campaigns</s>

<|system|>A novel sparsity optimization method is proposed to select features for multi-class classification problems by directly optimizing a l2,p -norm ( 0 < p ≤ 1 ) based sparsity function subject to data-fitting inequality constraints to obtain large between-class margins. The direct sparse optimization method circumvents the empirical tuning of regularization parameters in existing feature selection methods that adopt the sparsity model as a regularization term. To solve the direct sparsity optimization problem that is non-smooth and non-convex when 0 < p < 1, we propose an efficient iterative algorithm with proved convergence by converting it to a convex and smooth optimization problem at every iteration step. The proposed algorithm has been evaluated based on publicly available datasets. The experiments have demonstrated that our algorithm could achieve feature selection performance competitive to state-of-the-art algorithms.</s>
<|user|>what type of method is proposed to select classification problems ?</s>
<|assistant|>data fitting inequality constraints</s>

<|system|>A fundamental problem for artificial intelligence is identifying perceptual primitives from raw sensory signals that are useful for higher-level reasoning. We equate these primitives with initially unknown recurring patterns called motifs. Autonomously learning the motifs is difficult because their number, location, length, and shape are all unknown. Furthermore, nonlinear temporal warping may be required to ensure the similarity of motif occurrences. In this paper, we extend a leading motif discovery algorithm by allowing it to operate on multidimensional sensor data, incorporating automatic parameter estimation, and providing for motif-specific similarity adaptation. We evaluate our algorithm on several data sets and show how our approach leads to faster real world discovery and more accurate motifs compared to other leading methods.</s>
<|user|>what part of motif occurrences , nonlinear temporal warping be required to ensure what ?</s>
<|assistant|>the similarity</s>

<|system|>While multitask learning has been extensively studied, most existing methods rely on linear models (e.g. linear regression, logistic regression), which may fail in dealing with more general (nonlinear) problems. In this paper, we present a new approach that combines dictionary learning with gradient boosting to achieve multitask learning with general (nonlinear) basis functions. Specifically, for each task we learn a sparse representation in a nonlinear dictionary that is shared across the set of tasks. Each atom of the dictionary is a nonlinear feature mapping of the original input space, learned in function space by gradient boosting. The resulting model is a hierarchical ensemble where the top layer of the hierarchy is the task-specific sparse coefficients and the bottom layer is the boosted models common to all tasks. The proposed method takes the advantages of both dictionary learning and boosting for multitask learning: knowledge across tasks can be shared via the dictionary, and flexibility and generalization performance are guaranteed by boosting. More important, this general framework can be used to adapt any learning algorithm to (nonlinear) multitask learning. Experimental results on both synthetic and benchmark real-world datasets confirm the effectiveness of the proposed approach for multitask learning.</s>
<|user|>what does each atom of the dictionary include ?</s>
<|assistant|>the original input space</s>
""".split('.')



bpe = bytePairEncoding(data, 1010)

for k, v in bpe.items():
    print(f"{k} ---- {v}")

