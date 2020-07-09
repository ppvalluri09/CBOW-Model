from sklearn.manifold import TSNE
import pickle
from matplotlib import pyplot as plt
from data_prep import *
import sys

text_corpus = open("./dataset/shakespeare.txt").read()
text_corpus = text_corpus.replace("\n", " ")
corpus = tokenize(text_corpus)
word2ind, ind2word = generate_dict(corpus)
V = len(word2ind)
print("Vocab Size", V)
start = int(sys.argv[1])
end = int(sys.argv[2])
embedding_matrix = pickle.load(open("./embeddings/embedding_matrix.pickle", "rb")).T[start:end, :]
print("Embedding Matrix Dimensions", embedding_matrix.shape)
tsne = TSNE(n_components=2)
embedding_matrixD2 = tsne.fit_transform(embedding_matrix)
print("Embedding Matrix after t-SNE", embedding_matrixD2.shape)

fig, ax = plt.subplots()
ax.scatter(embedding_matrixD2[:, 0], embedding_matrixD2[:, 1], c="b", label="Embeddings")
for k, v in word2ind.items():
    if v >= end - start:
        break
    ax.annotate(k, (embedding_matrixD2[v, 0], embedding_matrixD2[v, 1]))
plt.show()
