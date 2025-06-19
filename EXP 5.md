<H3>ENTER YOUR NAME : SHARMITHA V</H3>
<H3>ENTER YOUR REGISTER NO : 212223110048</H3>
<H3>EX. NO.5</H3>
<H1 ALIGN =CENTER> Word Embedding using Word2Vec for Semantic Similarity in Text Corpus</H1>

## AIM:
To preprocess a text corpus and train a Word2Vec model to extract word embeddings and identify similar words.

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook


## ALGORITHM:

1.Tokenize and lowercase the corpus sentences.
2.Remove punctuation and stopwords using NLTK.
3.Train the Word2Vec model using the preprocessed tokens.
4.Extract the vector embedding for a specific word (e.g., "vector").
5.Find the top 2 most similar words and their similarity scores.


##  PROGRAM:
~~~
# Import necessary libraries
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample corpus
corpus = [
    "Natural language processing is a field of artificial intelligence.",
    "It enables computers to understand human language.",
    "Word embedding is a representation of words in a dense vector space.",
    "Gensim is a library for training word embeddings in Python.",
    "Machine learning and deep learning techniques are widely used in NLP."
]

# Preprocess the text: Tokenize, remove punctuation and stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return tokens

# Apply preprocessing to the corpus
processed_corpus = [preprocess_text(sentence) for sentence in corpus]

# Train a Word2Vec model
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=2, min_count=1, sg=1)  # sg=1 uses Skip-gram

# Save the model for future use
model.save("word2vec_model.model")

# Test the model by finding the embedding of a word
word = "vector"
if word in model.wv:
    print(f"Embedding for '{word}':\n{model.wv[word]}")
else:
    print(f"'{word}' not found in vocabulary.")

# Find similar words
similar_words = model.wv.most_similar(word, topn=2)
print(f"Words similar to '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity:.4f}")

~~~
## OUTPUT:

![image](https://github.com/user-attachments/assets/f3565664-2aa1-4603-b5c0-1b7d9316ca6f)


## RESULT:

Thus ,the Word2Vec model was successfully trained on the preprocessed corpus.
