<H3>ENTER YOUR NAME : T.RUCHITRA</H3>
<H3>ENTER YOUR REGISTER NO : 212223110043</H3>
<H3>EX. NO.6</H3>
<H1 ALIGN =CENTER> Auto-Correction of Words Using Edit Distance and Word Probability</H1>

## AIM:
To implement automatic word correction by computing edit distance and selecting the most probable correction from a known vocabulary.

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## ALGORITHM:

1.Load vocabulary from NLTK’s English word list.
2.Define edit distance function using dynamic programming.
3.Create a word frequency model to estimate word probabilities.
4.Generate candidates within edit distance ≤ 2 of the misspelled word.
5.Select correction with the highest probability among candidates.



##  PROGRAM:
~~~
# Install NLTK if needed
!pip install nltk
import nltk
nltk.download('words')
from nltk.corpus import words
import re
from collections import Counter
# Use the NLTK words corpus as our vocabulary
word_list = words.words()
word_freq = Counter(word_list)  # Count frequencies, though here it's a simple corpus with each word appearing once

# Define a set of all known words
WORD_SET = set(word_list)
# Define a function to calculate minimum edit distance
def edit_distance(word1, word2):
    dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
    for i in range(len(word1) + 1):
        for j in range(len(word2) + 1):
            if i == 0:
                dp[i][j] = j  # Cost of insertions
            elif j == 0:
                dp[i][j] = i  # Cost of deletions
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No change cost
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                   dp[i][j - 1],      # Insertion
                                   dp[i - 1][j - 1])  # Substitution
    return dp[-1][-1]

# Define a function to calculate word probability
def word_probability(word, N=sum(word_freq.values())):
    return word_freq[word] / N if word in word_freq else 0
# Suggest corrections based on edit distance and probability
def autocorrect(word):
    # If the word is correct, return it as is
    if word in WORD_SET:
        return word

    # Find candidate words within an edit distance of 1 or 2
    candidates = [w for w in WORD_SET if edit_distance(word, w) <= 2]

    # Choose the candidate with the highest probability
    corrected_word = max(candidates, key=word_probability, default=word)

    return corrected_word
# Test the function with common misspellings
test_words = ["speling", "korrect", "exampl", "wrld"]

for word in test_words:
    print(f"Original: {word} -> Suggested: {autocorrect(word)}")


~~~
## OUTPUT:

![image](https://github.com/user-attachments/assets/c4f42cc0-197c-4e85-a559-c6c4491876c0)



## RESULT:

Thus,The auto-correction system successfully suggests accurate corrections for misspelled words based on edit distance and dictionary-based probabilities.
