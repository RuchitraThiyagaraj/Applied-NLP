<H3>ENTER YOUR NAME :T.RUCHITRA</H3>
<H3>ENTER YOUR REGISTER NO : 212223110043 </H3>
<H3>EX. NO.1</H3>
<H1 ALIGN =CENTER> Sentiment Analysis using Naive Bayes</H1>

## AIM:
Apply the Naive Bayes Algorithm to perform sentiment analysis by using a customized data set.

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## ALGORITHM:
1.Import necessary libraries like nltk, sklearn, and numpy.
2.Prepare labeled text data with sentiments.
3.Split the data and vectorize using CountVectorizer with stopword removal.
4.Train a MultinomialNB classifier on the training data.
5.Predict and evaluate model accuracy and performance.

##  PROGRAM:
~~~
# Import necessary libraries
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Download NLTK data (stopwords)
nltk.download('stopwords')

# Sample dataset of sentences with their sentiments (1 = positive, 0 = negative)
data = [
    ("I love this product, it works great!", 1),
    ("This is the best purchase I have ever made.", 1),
    ("Absolutely fantastic service and amazing quality!", 1),
    ("I am very happy with my order, will buy again.", 1),
    ("This is a horrible experience.", 0),
    ("I hate this so much, it broke on the first day.", 0),
    ("Worst product I have ever used, total waste of money.", 0),
    ("I am disappointed with this product, it didn't work as expected.", 0)
]

# Separate sentences and labels
sentences = [pair[0] for pair in data]
labels = np.array([pair[1] for pair in data])

# Split dataset into training and testing sets
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=42)

# Text Preprocessing
# Tokenization, removing stopwords and converting text into numerical data using CountVectorizer

# Instead of using a set, use a list for stop_words
stop_words = stopwords.words('english') # Changed this line to create a list

# Initialize CountVectorizer (this will convert text into a bag-of-words representation)
vectorizer = CountVectorizer(stop_words=stop_words)

# Fit the vectorizer on the training data and transform both training and test sets
X_train = vectorizer.fit_transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

# Initialize the Naive Bayes Classifier
nb_classifier = MultinomialNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict sentiments for the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Test the model with new sentences
test_sentences = ["I am happy to comment!", "This is a terrible product."]
test_X = vectorizer.transform(test_sentences)

# Predict sentiments for new sentences
predictions = nb_classifier.predict(test_X)

# Output predictions
for sentence, sentiment in zip(test_sentences, predictions):
    print(f"Sentence: '{sentence}' => Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
~~~
## OUTPUT:

![image](https://github.com/user-attachments/assets/18213d50-0719-44bb-a443-b22fa0a527c3)


## RESULT:
Thus, the Implementation of Naive Bayes was successful and gave accurate sentiment predictions.
