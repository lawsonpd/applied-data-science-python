# Project 2 - Introduction to NLTK

In part 1 of this project we use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 we create a spelling recommender function that uses nltk to find words similar to the misspelling. 

## Part 1 - Analyzing Moby Dick


```python
import nltk
import pandas as pd
import numpy as np

# To work with the raw text use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# To work with the novel in nltk.Text format use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)
```

### Example 1

How many tokens (words and punctuation symbols) are in text1?

*Returns an integer.*


```python
def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)
```

### Example 2

How many unique tokens (unique words and punctuation) does text1 have?

*Returns an integer.*


```python
def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))
```

### Example 3

After lemmatizing the verbs, how many unique tokens does text1 have?

*Returns an integer.*


```python
from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))
```

### Part 1

What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)

*Returns a float.*


```python
def lexical_diversity():
    
    all_tokens = nltk.word_tokenize(moby_raw)
    unique_tokens = set(nltk.word_tokenize(moby_raw))
    return len(unique_tokens) / len(all_tokens)
```

### Part 2

What percentage of tokens is 'whale'or 'Whale'?

*Returns a float.*


```python
def percent_whale_Whale():
    
    whale_tokens = [w for w in moby_tokens if w == 'whale' or w == 'Whale']
    return len(whale_tokens)/len(moby_tokens)*100
```


### Part 3

What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?

*Returns a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*


```python
def most_frequent_tokens(count=20):
    
    freqs = nltk.FreqDist(moby_tokens)
    
    return freqs.most_common(count)
```


### Part 4

What tokens have a length of greater than 5 and frequency of more than 150?

*Returns a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*


```python
def token_with_len_gt_5():
    freqs = nltk.FreqDist(moby_tokens)
    
    return sorted([token for token in freqs if len(token) > 5 and freqs.get(token) > 150])
```


### Part 5

Find the longest word in text1 and that word's length.

*Returns a tuple `(longest_word, length)`.*


```python
def find_longest_word():
    longest_word = sorted(moby_tokens, key=len, reverse=True)[0]
    
    return longest_word, len(longest_word)
```


### Part 6

What unique words have a frequency of more than 2000? What is their frequency?

*Returns a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*


```python
def get_word_freqs(f=2000):
    freq_words = nltk.FreqDist(moby_tokens)
    words_sorted = sorted([token for token in freq_words if token.isalpha() and freq_words.get(token) > f], key=freq_words.get, reverse=True)
    word_freqs = [freq_words[word] for word in words_sorted]
    
    return list(zip(word_freqs, words_sorted))
```


### Part 7

What is the average number of tokens per sentence?

*Returns a float.*


```python
def avg_tokens_per_sentence():
    
    sents = nltk.sent_tokenize(moby_raw)
    words = nltk.word_tokenize(moby_raw)
    return len(words) / len(sents)
```

### Part 8

What are the 5 most frequent parts of speech in this text? What is their frequency?

*Returns a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*


```python
def five_most_freq_pos():
    moby_text = nltk.word_tokenize(moby_raw)
    pos = nltk.pos_tag(moby_text)
    
    tags = [tag for word, tag in pos]
    tag_freqs = nltk.FreqDist(tags)
    
    return tag_freqs.most_common(5)
```

## Part 2 - Spelling Recommender

For this part of the project create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.

For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.

*Each of the three different recommenders will use a different distance measure (outlined below).

Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.


```python
from nltk.corpus import words
from nltk.metrics import jaccard_distance as jaccard
from nltk.metrics import edit_distance as levenshtein

correct_spellings = words.words()
```

### Part 9

For this recommender, provide recommendations for the three default words provided above using the following distance metric:

**[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**

*Returns a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
def word_recs_1(entries=['cormulent', 'incendenece', 'validrate']):
    first_letters = [word[0] for word in entries]
    
    same_first_letter = [[word for word in correct_spellings if word[0] == letter] for letter in first_letters]
    
    top = ['', '', '']
    for i in range(len(entries)):
        best_score = 1.0
        for word in same_first_letter[i]:
            current_score = jaccard(set(nltk.ngrams(entries[i], 3)), set(nltk.ngrams(word, 3)))
            if current_score < best_score:
                best_score = current_score
                top[i] = word
    
    return top
```

### Part 10

For this recommender, provide recommendations for the three default words provided above using the following distance metric:

**[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**

*Returns a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
def word_recs_2(entries=['cormulent', 'incendenece', 'validrate']):
    first_letters = [word[0] for word in entries]
    
    same_first_letter = [[word for word in correct_spellings if word[0] == letter] for letter in first_letters]
    
    # use zip(entries, same_first_letter)?
    
    top = ['', '', '']
    for i in range(len(entries)):
        best_score = 1.0
        for word in same_first_letter[i]:
            current_score = jaccard(set(nltk.ngrams(entries[i], 4)), set(nltk.ngrams(word, 4)))
            if current_score < best_score:
                best_score = current_score
                top[i] = word
    
    return top
```

### Part 11

For this recommender, provide recommendations for the three default words provided above using the following distance metric:

**[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**

*Returns a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
def word_recs_3(entries=['cormulent', 'incendenece', 'validrate']):
    first_letters = [word[0] for word in entries]
    
    same_first_letter = [[word for word in correct_spellings if word[0] == letter] for letter in first_letters]
    
    # use zip(entries, same_first_letter)?
    
    top = ['', '', '']
    for i in range(len(entries)):
        best_score = 1000000
        for word in same_first_letter[i]:
            current_score = levenshtein(entries[i], word, transpositions=True)
            if current_score < best_score:
                best_score = current_score
                top[i] = word
    
    return top
```
