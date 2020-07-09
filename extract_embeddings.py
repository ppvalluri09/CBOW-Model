import pickle
import numpy as np

parameters = pickle.load(open("./models/paramsC2E3.pickle", "rb"))

embedding_matrix = (parameters["W1"] + parameters["W2"].T) / 2

print("Embedding Matrix shape", embedding_matrix.shape)
print("Embedding Matrix \n")
print(embedding_matrix)
pickle.dump(embedding_matrix, open("./embeddings/embedding_matrix.pickle", "wb"))
