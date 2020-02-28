# Assignment 3 - Evaluation

In this assignment we trained several models and evaluate how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
 
Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. 
 
The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.


```python
import numpy as np
import pandas as pd
```

### Part 1
Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?

*Returns a float between 0 and 1.* 


```python
def percent_fraud():
    
    fraud_df = pd.read_csv('fraud_data.csv')
    
    is_fraud = fraud_df[fraud_df['Class'] == 1]
    not_fraud = fraud_df[fraud_df['Class'] == 0]
    
    return len(is_fraud) / len(fraud_df)
```


```python
# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

### Part 2

Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?

*This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*


```python
def dummy_clf_scores():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    
    clf_dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    
    dummy_acc_score = clf_dummy.score(X_test, y_test)
    
    preds = clf_dummy.predict(X_test)
    
    dummy_recall_score = recall_score(y_test, preds)
    
    return dummy_acc_score, dummy_recall_score
```

### Part 3

Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?

*Returns a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*


```python
def svc_scores():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    clf = SVC().fit(X_train, y_train)
    
    test_preds = clf.predict(X_test)
    
    acc = clf.score(X_test, y_test)
    rec = recall_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    
    return acc, rec, prec

```

### Part 4

Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.

*Returns a confusion matrix, a 2x2 numpy array with 4 integers.*


```python
def svc_confusion_matrix():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    clf = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)

    y_dec_fn = clf.decision_function(X_test)
    
    df = pd.DataFrame(list(zip(y_test, y_dec_fn)), columns=['y_test', 'y_dec_fn'])
    
    df['y_dec_fn'] = df['y_dec_fn'].apply(lambda x: 1 if x >= -220 else 0)
    
    conf_mtx = confusion_matrix(df['y_test'], df['y_dec_fn'])
    
    return conf_mtx

```

### Part 5

Train a logisitic regression classifier with default parameters using X_train and y_train.

For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).

Looking at the precision recall curve, what is the recall when the precision is `0.75`?

Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?

*Returns a tuple with two floats, i.e. `(recall, true positive rate)`.*


```python
def lr_scores():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
        
    lr = LogisticRegression().fit(X_train, y_train)
    
    lr_dec_fn = lr.decision_function(X_test)
    test_preds = lr.predict(X_test)
    
    prec, rec, _ = precision_recall_curve(y_test, lr_dec_fn)
    fpr, tpr, _ = roc_curve(y_test, lr_dec_fn)
    
    # make dataframes of curve values
    pr_df = pd.DataFrame(list(zip(prec, rec)), columns=['precision', 'recall'])
    fpr_tpr_df = pd.DataFrame(list(zip(fpr, tpr)), columns=['false positive rate', 'true positive rate'])
    
    # recall value when precision is 0.75
    r_spec = float(pr_df[pr_df['precision'] == 0.75].recall)
    
    # rows near 0.16 FPR value
    tpr_spec_sel = fpr_tpr_df[(fpr_tpr_df['false positive rate'] >= 0.159) & (fpr_tpr_df['false positive rate'] <= 0.161)]
    # TPR value - avg of rows in selection
    tpr_spec = tpr_spec_sel['true positive rate'].mean()
    
    return r_spec, tpr_spec

```

### Part 6

Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.

`'penalty': ['l1', 'l2']`

`'C':[0.01, 0.1, 1, 10, 100]`

From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.

|      	| `l1` 	| `l2` 	|
|:----:	|----	|----	|
| **`0.01`** 	|    ?	|   ? 	|
| **`0.1`**  	|    ?	|   ? 	|
| **`1`**    	|    ?	|   ? 	|
| **`10`**   	|    ?	|   ? 	|
| **`100`**   	|    ?	|   ? 	|

<br>

*Returns a 5 by 2 numpy array with 10 floats.* 


```python
def lr_grid_search_results():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    
    gs = GridSearchCV(lr, {'penalty': ['l1', 'l2'], 'C':[0.01, 0.1, 1, 10, 100]}, scoring='recall')
    
    gs.fit(X_train, y_train)
    
    g = gs.cv_results_['mean_test_score']
    
    return g.reshape(5,2)

```
