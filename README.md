# Preprocessing

### Steps

<ul>
    <li>Substitute all punctuations with "." (punctuations as in the foll:- [, ! ? ; -])</li>
    <li>Ignore all numbers except for special ones such as 3.141 or PINCODES, etc...</li>
    <li>Ignore all special characters</li>
    <li>Tokenize the sentence using the word_tokenize function provided by the NLTK library</li>
    <li>Hnadle emojis using the emoji api in python and convert all words to lower case</li>
</ul>

# Sliding Window

Now we use the sliding window technique to get the center word and context word with some context half-size C.
Example:- <i>"Hello there, this is an example sentence"</i>

### Preprocessing
Output:- <i>"hello there. this is an example sentence"</i>

### Sliding Window to get context and center words
Say we define our context half-size to be C = 2
Output:- [(["hello", there", "this", "is"], ["."]),
	  (["there", ".", "is", "an"], ["this"]),
	  ([".", "this", "an", "example"], ["is"]),
	  (["this", "is", "example", "sentence"], ["an"])]

# Data Prep

Convert the context vectors to one-hot vectors and take the average, this would be the input to our shallow nertwork. The output would be the one-hot vector of the center word.

# Embedding Matrix

After training the model we extract the embedding matrix by taking the average of <b>W1</b> and <b>W2<sup>T</sup></b>, this would be our embedding matrix.

# Visualizing the Embedding Matrix

Since our vocabulary consists of around 5K words, we limit our embedding matrix size to around 50 - 100 words and apply the dimensionality reduction technique <b>t-SNE</b> and annotate with our vocabulary...
