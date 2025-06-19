<H3>ENTER YOUR NAME : SHARMITHA V</H3>
<H3>ENTER YOUR REGISTER NO : 212223110048</H3>
<H3>EX. NO.2</H3>
<H1 ALIGN =CENTER> Machine Translation using KNN</H1>

## AIM:
To create a basic machine translation system using the K-Nearest Neighbors (KNN) algorithm with cosine similarity on TF-IDF vectors.

## EQUIPMENTS REQUIRED :
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## ALGORITHM:

1.Import necessary libraries like sklearn and numpy.
2.Create a bilingual dictionary with English and French sentence pairs.
3.Convert English sentences into TF-IDF vectors.
4.For a new sentence, compute cosine similarity with all dictionary entries.
5.Return the French translation of the most similar English sentence.

##  PROGRAM:
~~~
!pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Sample bilingual dictionary
english_sentences = [
    "hello", "how are you", "good morning", "good night", "thank you",
    "see you later", "what is your name", "my name is John", "where is the library",
    "I like to read books"
]

french_sentences = [
    "bonjour", "comment ça va", "bonjour", "bonne nuit", "merci",
    "à plus tard", "quel est ton nom", "mon nom est John", "où est la bibliothèque",
    "j'aime lire des livres"
]

vectorizer = TfidfVectorizer()
english_vectors = vectorizer.fit_transform(english_sentences)
def knn_translate(input_sentence, k=1):
    input_vector = vectorizer.transform([input_sentence])

    # Compute cosine similarity between the input sentence and all sentences in the dictionary
    similarities = cosine_similarity(input_vector, english_vectors).flatten()

    # Get indices of the top-k similar sentences
    top_k_indices = similarities.argsort()[-k:][::-1]

    # Retrieve and display the French translations for the most similar sentences
    translations = [french_sentences[i] for i in top_k_indices]
    return translations

# Test sentences
test_sentences = ["good evening", "where is the library", "thank you very much"]

# Translate each test sentence
for sentence in test_sentences:
    translations = knn_translate(sentence, k=1)  # Use k=1 for the closest translation
    print(f"English: {sentence} -> French: {translations[0]}")

~~~
## OUTPUT:
![image](https://github.com/user-attachments/assets/bf13eceb-e88a-479f-8012-64ea05abb5b7)

## RESULT:

The KNN-based translation system successfully translated English sentences into French by identifying the most similar sentence in the dictionary using cosine similarity.
