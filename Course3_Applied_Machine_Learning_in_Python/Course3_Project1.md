# Project 1 - Introduction to Machine Learning

For this project, we used the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients.


```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# print(cancer.DESCR) # Print the data set description
```

### Part 1

Convert the sklearn.dataset `cancer` to a DataFrame. 

*Returns a `(569, 31)` DataFrame with* 

*columns =*

    ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension',
    'target']

*and index =*

    RangeIndex(start=0, stop=569, step=1)


```python
def convert_to_dataframe():

    df = pd.DataFrame(cancer.data, index=range(569), columns=cancer.feature_names)
    df['target'] = cancer.target

    return df
```

### Part 2
Determine class distribution. (I.e. how many instances of `malignant` (encoded 0) and how many `benign` (encoded 1))

*Returns a Series named `target` of length 2 with integer values and index =* `['malignant', 'benign']`


```python
def find_class_dist():
    cancerdf = convert_to_dataframe()
    
    mal = cancerdf.loc[lambda cancerdf: cancerdf.target == 0]
    ben = cancerdf.loc[lambda cancerdf: cancerdf.target == 1]
    num_mal = len(mal)
    num_ben = len(ben)
    target = pd.Series([num_mal, num_ben], index=['malignant', 'benign'])
    
    return target
```

### Part 3
Split the DataFrame into `X` (the data) and `y` (the labels).

*Returns a tuple of length 2:* `(X, y)`*, where* 
* `X`*, a pandas DataFrame, has shape* `(569, 30)`
* `y`*, a pandas Series, has shape* `(569,)`.


```python
def split_data_and_labels():
    cancerdf = convert_to_dataframe()
    
    cols = len(cancerdf.columns)
    
    X = cancerdf.iloc[:, :cols-1]
    y = cancerdf['target']
    
    return X, y
```

### Part 4
Using `train_test_split`, split `X` and `y` into training and test sets `(X_train, X_test, y_train, and y_test)`.

*Returns a tuple of length 4:* `(X_train, X_test, y_train, y_test)`*, where* 
* `X_train` *has shape* `(426, 30)`
* `X_test` *has shape* `(143, 30)`
* `y_train` *has shape* `(426,)`
* `y_test` *has shape* `(143,)`


```python
from sklearn.model_selection import train_test_split

def split_train_and_test_sets():
    X, y = split_data_and_labels()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    return X_train, X_test, y_train, y_test

```

### Part 5
Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with `X_train`, `y_train` and using one nearest neighbor (`n_neighbors = 1`).

*Returns a * `sklearn.neighbors.classification.KNeighborsClassifier`.


```python
from sklearn.neighbors import KNeighborsClassifier

def fit_knn_classifier():
    X_train, X_test, y_train, y_test = split_train_and_test_sets()
    
    knn = KNeighborsClassifier(n_neighbors=1)
    
    return knn.fit(X_train, y_train)
```

### Part 6
Using knn classifier, predict the class label using the mean value for each feature.

*Returns a numpy array either `array([ 0.])` or `array([ 1.])`*


```python
def predict_label_from_means():
    cancerdf = convert_to_dataframe()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    model = answer_five()
    return model.predict(means)
```

### Part 7
Using knn classifier, predict the class labels for the test set `X_test`.

*Returns a numpy array with shape `(143,)` and values either `0.0` or `1.0`.*


```python
def predict_test_set_labels():
    X_train, X_test, y_train, y_test = split_train_and_test_sets()
    knn = answer_five()
    
    return knn.predict(X_test)
```

### Part 8
Find the score (mean accuracy) of the knn classifier using `X_test` and `y_test`.

*Returns a float between 0 and 1*


```python
def clf_score():
    X_train, X_test, y_train, y_test = split_train_and_test_sets()
    knn = answer_five()
    
    return knn.score(X_test, y_test)
```
