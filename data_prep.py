import numpy as np
import nltk
import re
import emoji

def tokenize(corpus):
    data = re.sub(r"[,!?;-]+", ".", corpus)
    data = nltk.word_tokenize(data)
    data = [ ch.lower() for ch in data
             if ch.isalpha() or
             ch == "." or
             emoji.get_emoji_regexp().search(ch)]

    return data

def get_windows(corpus, C=2):
    i = C
    while i < len(corpus) - C:
        context = corpus[i-C:i] + corpus[i+1:i+C+1]
        center_word = corpus[i]
        yield context, center_word
        i += 1

def generate_dict(corpus):
    words = sorted(list(set(corpus)))
    n = len(words)
    idx = 0
    word2ind = {}
    ind2word = {}
    for k in words:
        word2ind[k] = idx
        ind2word[idx] = k
        idx += 1
    return word2ind, ind2word

def one_hot(word, word2ind, V):
    assert len(word2ind) == V
    one_hot = np.zeros(V)
    n = word2ind[word]
    one_hot[n] = 1

    return one_hot

def context_word_to_vec(context, word2ind, V):
    word_vectors = [one_hot(word, word2ind, V) for word in context]
    context_word_vectors = np.mean(word_vectors, axis=0)
    return context_word_vectors


def get_training_example(corpus, C, word2ind, V):
    for context, center in get_windows(corpus, C):
        yield context_word_to_vec(context, word2ind, V), one_hot(center, word2ind, V)

def get_batches(corpus, C, word2ind, V, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_training_example(corpus, C, word2ind, V):
        if len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []

##if __name__ == "__main__":
##    corpus = "Hello there, this is a test sentence to demo data prep"
##    corpus = tokenize(corpus)
##    word2ind, ind2word = generate_dict(corpus)
##    V = len(word2ind)
##    C = 2
##    for context, center in get_training_example(corpus, C, word2ind, V):
##        print(f"Context {context} --> Center {center}")
