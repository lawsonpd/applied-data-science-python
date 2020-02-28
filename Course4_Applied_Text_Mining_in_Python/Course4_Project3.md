# Project 3

In this project we explore text message data and create models to predict if a message is spam or not. 


```python
import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)
```

<div>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>text</th>
        <th>target</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>Go until jurong point, crazy.. Available only ...</td>
        <td>0</td>
      </tr>
      <tr>
        <th>1</th>
        <td>Ok lar... Joking wif u oni...</td>
        <td>0</td>
      </tr>
      <tr>
        <th>2</th>
        <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
        <td>1</td>
      </tr>
      <tr>
        <th>3</th>
        <td>U dun say so early hor... U c already then say...</td>
        <td>0</td>
      </tr>
      <tr>
        <th>4</th>
        <td>Nah I don't think he goes to usf, he lives aro...</td>
        <td>0</td>
      </tr>
      <tr>
        <th>5</th>
        <td>FreeMsg Hey there darling it's been 3 week's n...</td>
        <td>1</td>
      </tr>
      <tr>
        <th>6</th>
        <td>Even my brother is not like to speak with me. ...</td>
        <td>0</td>
      </tr>
      <tr>
        <th>7</th>
        <td>As per your request 'Melle Melle (Oru Minnamin...</td>
        <td>0</td>
      </tr>
      <tr>
        <th>8</th>
        <td>WINNER!! As a valued network customer you have...</td>
        <td>1</td>
      </tr>
      <tr>
        <th>9</th>
        <td>Had your mobile 11 months or more? U R entitle...</td>
        <td>1</td>
      </tr>
    </tbody>
  </table>
</div>


```python
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)
```

### Part 1
What percentage of the documents in `spam_data` are spam?

*Returns a float, the percent value (i.e. $ratio * 100$).*


```python
def spam_percentage():
    spam_count = spam_data['target'].sum()
    
    total = spam_data['target'].count()
    
    spam_percent = spam_count / total * 100
    
    return spam_percent
```


### Part 2

Fit the training data `X_train` using a Count Vectorizer with default parameters.

What is the longest token in the vocabulary?

*Returns a string.*


```python
from sklearn.feature_extraction.text import CountVectorizer

def longest_token_in_vocab():
    vect = CountVectorizer().fit(X_train)
    
    longest_token = max(vect.vocabulary_, key=len)
    
    return longest_token
```


### Part 3

Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.

Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.

*Returns the AUC score as a float.*


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def count_vect_auc_score():
    vect = CountVectorizer().fit(X_train)
    
    train_vect = vect.transform(X_train)
    
    multi_nbc = MultinomialNB(alpha=0.1).fit(train_vect, y_train)
    
    test_vect = vect.transform(X_test)
    
    preds = multi_nbc.predict(test_vect)
    
    auc_score = roc_auc_score(y_test, preds)
    
    return auc_score
```


### Part 4

Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.

What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?

Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.

The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 

*Returns a tuple of two series
`(smallest tf-idfs series, largest tf-idfs series)`.*


```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf_features():    
    vectorizer = TfidfVectorizer().fit(X_train)
    
    train_vect = vectorizer.transform(X_train)
    
    feature_names = np.array(vectorizer.get_feature_names())
    
    sorted_tfidf_index = train_vect.max(0).toarray()[0].argsort()
    
    feature_series = pd.Series(vectorizer.idf_, index=vectorizer.get_feature_names())
    
    feature_series.sort_values(inplace=True)
    
    top = feature_series[-20:].sort_index()
    bottom = feature_series[:20].sort_index()
    
    return bottom, top

```


### Part 5

Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.

Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.

*Returns the AUC score as a float.*


```python
def tfidf_fect_auc_score():
    vect = TfidfVectorizer(min_df=3)
    
    train_vect = vect.fit_transform(X_train)
   
    nbc = MultinomialNB(alpha=0.1).fit(train_vect, y_train)
    
    test_vect = vect.transform(X_test)
    
    preds = nbc.predict(test_vect)
    
    auc_score = roc_auc_score(y_test, preds)
    
    return auc_score

```


### Part 6

What is the average length of documents (number of characters) for not spam and spam documents?

*Returns a tuple (average length not spam, average length spam).*


```python
def avg_doc_length_spam_nonspam():
    is_spam = spam_data.where(spam_data['target'] == 1).dropna()
    not_spam = spam_data.where(spam_data['target'] == 0).dropna()
    
    return not_spam['text'].apply(len).mean(), is_spam['text'].apply(len).mean()
```


<br>
<br>
The following function has been provided to help student combine new features into the training data:


```python
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
```

### Part 7

Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.

Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.

*Returns the AUC score as a float.*


```python
from sklearn.svm import SVC

def tfidf_vect_auc_score_2():
    vectorizer = TfidfVectorizer(min_df=5).fit(X_train, y_train)
    
    train_vect = vectorizer.transform(X_train)
    
    train_doc_length = X_train.apply(len)
    
    train_vect_with_length = add_feature(train_vect, train_doc_length)
    
    svc = SVC(C=10000)
    
    svc.fit(train_vect_with_length, y_train)
    
    test_vect = vectorizer.transform(X_test)
    
    test_doc_length = X_test.apply(len)
    
    test_vect_with_length = add_feature(test_vect, test_doc_length)
    
    predictions = svc.predict(test_vect_with_length)
    
    score = roc_auc_score(y_test, predictions)
    
    return score
```


### Part 8

What is the average number of digits per document for not spam and spam documents?

*Returns a tuple (average # digits not spam, average # digits spam).*


```python
import re

def avg_digits_spam_nonspam():
    is_spam = spam_data.where(spam_data['target'] == 1).dropna()
    not_spam = spam_data.where(spam_data['target'] == 0).dropna()
    
    spam_avg_digits = is_spam['text'].apply(lambda doc: len(re.findall(r'\d', doc))).mean()
    not_spam_avg_digits = not_spam['text'].apply(lambda doc: len(re.findall(r'\d', doc))).mean()
    
    return not_spam_avg_digits, spam_avg_digits
```


### Part 9

Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).

Using this document-term matrix and the following additional features:
* the length of document (number of characters)
* **number of digits per document**

fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.

*Returns the AUC score as a float.*


```python
from sklearn.linear_model import LogisticRegression

def tfidf_vect_auc_score_3():
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    
    train_vect = vectorizer.transform(X_train)
    train_doc_length = X_train.apply(len)
    train_vect_with_length = add_feature(train_vect, train_doc_length)
    train_digit_count = X_train.apply(lambda x: len(re.findall(r'\d', x)))
    train_vect_with_length_and_digit_count = add_feature(train_vect_with_length, train_digit_count)
    
    reg = LogisticRegression(C=100).fit(train_vect_with_length_and_digit_count, y_train)
    
    test_vect = vectorizer.transform(X_test)
    test_doc_length = X_test.apply(len)
    test_vect_with_length = add_feature(test_vect, test_doc_length)
    test_digit_count = X_test.apply(lambda x: len(re.findall(r'\d', x)))
    test_vect_with_length_and_digit_count = add_feature(test_vect_with_length, test_digit_count)\
    
    reg_predictions = reg.predict(test_vect_with_length_and_digit_count)
    
    score = roc_auc_score(y_test, reg_predictions)
    
    return score
```


### Part 10

What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?

*Hint: Use `\w` and `\W` character classes*

*Returns a tuple (average # non-word characters not spam, average # non-word characters spam).*


```python
def avg_nonword_chars_spam_nonspam():
    is_spam = spam_data.where(spam_data['target'] == 1).dropna()
    not_spam = spam_data.where(spam_data['target'] == 0).dropna()
    
    is_spam['avg non word chars'] = is_spam['text'].apply(lambda doc: len(re.findall(r'\W', doc)))
    not_spam['avg non word chars'] = not_spam['text'].apply(lambda doc: len(re.findall(r'\W', doc)))

    return not_spam['avg non word chars'].mean(), is_spam['avg non word chars'].mean()

```


### Part 11

Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**

To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.

Using this document-term matrix and the following additional features:
* the length of document (number of characters)
* number of digits per document
* **number of non-word characters (anything other than a letter, digit or underscore.)**

fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.

Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.

The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.

The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
['length_of_doc', 'digit_count', 'non_word_char_count']

*Returns a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*


```python
def count_vect_auc_score_2():
    vectorizer = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb')
    
    train_vect = vectorizer.fit_transform(X_train, y_train)
    
    test_vect = vectorizer.transform(X_test)
    
    # add doc length feature to train and test
    train_doc_length = X_train.apply(len)
    train_vect_with_length = add_feature(train_vect, train_doc_length)
    test_doc_length = X_test.apply(len)
    test_vect_with_length = add_feature(test_vect, test_doc_length)
    
    # add number of digits feature to train and test
    num_digits_train = X_train.apply(lambda doc: len(re.findall(r'\d', doc)))
    train_vect_doc_length_num_digits = add_feature(train_vect_with_length, num_digits_train)
    num_digits_test = X_test.apply(lambda doc: len(re.findall(r'\d', doc)))
    test_vect_doc_length_num_digits = add_feature(test_vect_with_length, num_digits_test)
    
    # add number of non-word chars feature to train and test
    non_word_chars_train = X_train.apply(lambda doc: len(re.findall(r'\W', doc)))
    final_train_vect = add_feature(train_vect_doc_length_num_digits, non_word_chars_train)
    non_word_chars_test = X_test.apply(lambda doc: len(re.findall(r'\W', doc)))
    final_test_vect = add_feature(test_vect_doc_length_num_digits, non_word_chars_test)
    
    reg = LogisticRegression(C=100).fit(final_train_vect, y_train)
    
    test_preds = reg.predict(final_test_vect)
    
    auc_score = roc_auc_score(y_test, test_preds)
    
    coefs = reg.coef_[0]
    
    largest_coefs = sorted(coefs)[-10:]
    smallest_coefs = sorted(coefs)[:10]
    
    return auc_score, smallest_coefs, largest_coefs[::-1]

```
