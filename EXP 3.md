<H3>ENTER YOUR NAME : SHARMITHA V</H3>
<H3>ENTER YOUR REGISTER NO : 212223110048</H3>
<H3>EX. NO.1</H3>
<H1 ALIGN =CENTER> Sentence Similarity Detection Using Siamese LSTM Network</H1>

## AIM:
To detect semantic similarity between pairs of sentences using a Siamese LSTM network

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook


## ALGORITHM:
1.Tokenize and pad both sentence inputs.
2.Embed words using a shared Embedding layer.
3.Encode sentences with a shared BiLSTM.
4.Compute L1 distance between encoded vectors.
5.Classify similarity with a sigmoid Dense layer.

##  PROGRAM:
~~~
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

# Sentence pairs (labels: 1 for similar, 0 for dissimilar)
sentence_pairs = [
    ("How are you?", "How do you do?", 1),
    ("How are you?", "What is your name?", 0),
    ("What time is it?", "Can you tell me the time?", 1),
    ("What is your name?", "Tell me the time?", 0),
    ("Hello there!", "Hi!", 1),
]

# Separate into two sets of sentences and their labels
sentences1 = [pair[0] for pair in sentence_pairs]
sentences2 = [pair[1] for pair in sentence_pairs]
labels = np.array([pair[2] for pair in sentence_pairs])

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences1 + sentences2)
vocab_size = len(tokenizer.word_index) + 1

# Convert sentences to sequences
max_len = 100  # Max sequence length
X1 = pad_sequences(tokenizer.texts_to_sequences(sentences1), maxlen=max_len)
X2 = pad_sequences(tokenizer.texts_to_sequences(sentences2), maxlen=max_len)

# Input layers for two sentences
input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(max_len,))

# Embedding layer
embedding_dim = 1000
embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)

# Shared LSTM layer
shared_lstm = Bidirectional(LSTM(512))

# Process the two inputs using the shared LSTM
encoded_1 = shared_lstm(embedding(input_1))
encoded_2 = shared_lstm(embedding(input_2))

# Calculate the L1 distance between the two encoded sentences
def l1_distance(vectors):
    x, y = vectors
    return K.abs(x - y)

l1_layer = Lambda(l1_distance)
l1_distance_output = l1_layer([encoded_1, encoded_2])

# Add a dense layer for classification (similar/dissimilar)
output = Dense(1, activation='sigmoid')(l1_distance_output)

# Create the Siamese network model
siamese_network = Model([input_1, input_2], output)
siamese_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
siamese_network.summary()

# Train the model
siamese_network.fit([X1, X2], labels, epochs=12, batch_size=2)

# Test with a new sentence pair
test_sentences1 = ["How are you?"]
test_sentences2 = ["How do you do?"]

test_X1 = pad_sequences(tokenizer.texts_to_sequences(test_sentences1), maxlen=max_len)
test_X2 = pad_sequences(tokenizer.texts_to_sequences(test_sentences2), maxlen=max_len)

# Predict similarity
similarity = siamese_network.predict([test_X1, test_X2])
print(f"Similarity Score: {similarity[0][0]}")

~~~
## OUTPUT:

![image](https://github.com/user-attachments/assets/45057612-e6ac-40cf-9b38-f38be3c86dc6)

![image](https://github.com/user-attachments/assets/544996e7-10f1-4b17-8a95-ec42bd3cc0c9)


## RESULT:
Thus, the implementation to detect sentence similarity using an LSTM-based Siamese network has been successfully executed.
