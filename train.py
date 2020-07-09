from data_prep import *
import pickle
from matplotlib import pyplot as plt
import sys

np.random.seed(42)

def relu(x):
    result = x.copy()
    result = np.maximum(0, result)
    return result

def softmax(x):
    result = x.copy()
    result = np.exp(result)
    total = np.sum(result, axis=0)
    result = result / total
    return result


def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[1]
    log_pred = np.log(y_pred)
    log_pred = y_true * log_pred
    log_pred = np.sum(log_pred, axis=0, keepdims=True)
    J = (-1./m) * np.sum(log_pred)
    return J

def init_params(dims):
    assert type(dims) == tuple
    input_dim = dims[0]
    hidden_dim = dims[1]
    output_dim = dims[2]

    W1 = np.random.rand(hidden_dim, input_dim)
    b1 = np.random.rand(hidden_dim, 1)
    W2 = np.random.rand(output_dim, hidden_dim)
    b2 = np.random.rand(output_dim, 1)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters

def forward_pass(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    z1 = np.dot(W1, X) + b1
    h = relu(z1)
    z2 = np.dot(W2, h) + b2
    y_pred = softmax(z2)

    intermediate_results = {"h": h, "z1": z1, "z2": z2, "y_pred": y_pred}

    return intermediate_results

def calculate_gradients(x, y_true, intermediate_results, parameters):
    gradients = {}
    y_pred = intermediate_results["y_pred"]
    m = y_pred.shape[1]
    dW2 = (1./m) * np.dot(y_pred - y_true, intermediate_results["h"].T)
    db2 = (1./m) * (y_pred - y_true)

    dW1 = (1./m) * np.dot(relu(np.dot(parameters["W2"].T, y_pred - y_true)), x.T)
    db1 = (1./m)* relu(np.dot(parameters["W2"].T, y_pred - y_true))

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients

def train(corpus, C, word2ind, V, batch_size, alpha=0.03, epochs=2):
    m = batch_size
    H = 100
    loss_history = []
    parameters = init_params((V, H, V))
    for epoch in range(1, epochs+1):
        print(f"Epoch [{epoch}/{epochs}]")
        running_loss = 0.0
        for i, (x,y) in enumerate(get_batches(corpus, C, word2ind, V, m)):
            results = forward_pass(x, parameters)
            loss = cross_entropy_loss(results["y_pred"], y)
            running_loss += loss
            gradients = calculate_gradients(x, y, results, parameters)
            parameters["W1"] = parameters["W1"] - (alpha*gradients["dW1"])
            parameters["b1"] = parameters["b1"] - (alpha*gradients["db1"])
            parameters["W2"] = parameters["W2"] - (alpha*gradients["dW2"])
            parameters["b2"] = parameters["b2"] - (alpha*gradients["db2"])

            if i % 200 == 0:
                loss_history.append(loss)
                print(f"Epoch {epoch}, Batch {i}, Loss {loss}")

        print(f"Epoch [{epoch}/{epochs}], Loss {running_loss/(len(corpus) // batch_size)}")

    return parameters, loss_history
  

# note that the context half size and the number of epochs are to passed in as command line args...

text_corpus = open("./dataset/shakespeare.txt", "r").read()
text_corpus = text_corpus.replace("\n", " ")
corpus = tokenize(text_corpus)
print("Corpus Length", len(corpus))
word2ind, ind2word = generate_dict(corpus)
V = len(word2ind)
print("Size of Vocab", V)
if sys.argv[1]:
    C = int(sys.argv[1])
else:
    C = 2
print("Context Half Size", C)

if sys.argv[2]:
    epochs = int(sys.argv[2])
else:
    epochs = 3

parameters, loss_history = train(corpus, C, word2ind, V, batch_size=64, alpha=0.008, epochs=epochs)

pickle.dump(parameters, open(f"./models/paramsC{C}E{epochs}.pickle", "wb"))

plt.plot(loss_history, "r", label="Training Loss")
plt.legend()
plt.show()
