# Project 2 - Intro to Machine Learning

In this Project we explored the relationship between model complexity and generalization performance, by adjusting key parameters of various supervised learning models. Part 1 of this Project will look at regression and Part 2 will look at classification.

## Part 1 - Regression

Set up the variables needed for later sections.


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
```

### Part 1.1

Write a function that fits a polynomial LinearRegression model on the *training data* `X_train` for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model) For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. `np.linspace(0,10,100)`) and store this in a numpy array. The first row of this array should correspond to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.

<br>
*Returns a numpy array with shape `(4, 100)`*


```python
def poly_LR_predictions():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    X_train_resh = X_train.reshape(11,1)
    X_test_resh = X_test.reshape(4,1)
    
    X_predict = np.linspace(0,10,100)
    X_predict_resh = X_predict.reshape(-1,1)
    
    predictions = np.zeros([4,100])

    # stuff for poly deg 1
    poly_1 = PolynomialFeatures(1)
    poly_feat_train_1 = poly_1.fit_transform(X_train_resh)
    reg_1 = LinearRegression().fit(poly_feat_train_1, y_train)
    poly_feat_predict_1 = poly_1.fit_transform(X_predict_resh)
    predictions[0] = reg_1.predict(poly_feat_predict_1)
    
    # stuff for poly deg 3
    poly_3 = PolynomialFeatures(3)
    poly_feat_3 = poly_3.fit_transform(X_train_resh)
    reg_3 = LinearRegression().fit(poly_feat_3, y_train)
    poly_feat_predict_3 = poly_3.fit_transform(X_predict_resh)
    predictions[1] = reg_3.predict(poly_feat_predict_3)
    
    # stuff for poly deg 6
    poly_6 = PolynomialFeatures(6)
    poly_feat_6 = poly_6.fit_transform(X_train_resh)
    reg_6 = LinearRegression().fit(poly_feat_6, y_train)
    poly_feat_predict_6 = poly_6.fit_transform(X_predict_resh)
    predictions[2] = reg_6.predict(poly_feat_predict_6)

    # stuff for poly deg 9
    poly_9 = PolynomialFeatures(9)
    poly_feat_9 = poly_9.fit_transform(X_train_resh)
    reg_9 = LinearRegression().fit(poly_feat_9, y_train)
    poly_feat_predict_9 = poly_9.fit_transform(X_predict_resh)
    predictions[3] = reg_9.predict(poly_feat_predict_9)
    
    return predictions

```

### Part 1.2

Write a function that fits a polynomial LinearRegression model on the training data `X_train` for degrees 0 through 9. For each model compute the $R^2$ (coefficient of determination) regression score on the training data as well as the the test data, and return both of these arrays in a tuple.

*Returns one tuple of numpy arrays `(r2_train, r2_test)`. Both arrays should have shape `(10,)`*


```python
def poly_LR_reg_scores():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    reg_scores_train = np.zeros(10)
    reg_scores_test = np.zeros(10)
    
    for n in range(10):
        poly = PolynomialFeatures(n)
        poly_feat_train = poly.fit_transform(X_train.reshape(-1,1))
        
        reg = LinearRegression().fit(poly_feat_train, y_train)
        
        poly_feat_test = poly.fit_transform(X_test.reshape(-1,1))
        
        predicts_train = reg.predict(poly_feat_train)
        predicts_test = reg.predict(poly_feat_test)
        
        reg_scores_train[n] = r2_score(y_train, predicts_train)
        reg_scores_test[n] = r2_score(y_test, predicts_test)

    return reg_scores_train, reg_scores_test

```

### Part 1.3

Based on the $R^2$ scores from part 2 (degree levels 0 through 9), what degree level 
corresponds to a model that is underfitting? What degree level corresponds to a model that is overfitting? What choice of degree level would provide a model with good generalization performance on this dataset? 

*Returns one tuple with the degree values in this order: `(Underfitting, Overfitting, Good_Generalization)`.* 


```python
def model_fit_assessments():
    
    training_scores, test_scores = answer_two()
    
    underfit = training_scores.min()
    overfit = training_scores.max()
    good_gen = test_scores.max()
    
    return training_scores.tolist().index(underfit), training_scores.tolist().index(overfit), test_scores.tolist().index(good_gen)

```

### Part 1.4

Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.

For this part, train two models: a non-regularized LinearRegression model (default parameters) and a regularized Lasso Regression model (with parameters `alpha=0.01`, `max_iter=10000`) both on polynomial features of degree 12. Return the $R^2$ score for both the LinearRegression and Lasso model's test sets.

*Returns one tuple `(LinearRegression_R2_test_score, Lasso_R2_test_score)`*


```python
def linear_and_lasso_reg_scores():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score
    
    X_train_reshaped = X_train.reshape(-1,1)
    X_test_reshaped = X_test.reshape(-1,1)

    poly_feats_train = PolynomialFeatures(12).fit_transform(X_train_reshaped)
    
    linreg = LinearRegression().fit(poly_feats_train, y_train)
    lassoreg = Lasso(alpha=0.01, max_iter=10000).fit(poly_feats_train, y_train)
    
    poly_feats_test = PolynomialFeatures(12).fit_transform(X_test_reshaped)
    
    lin_pred = linreg.predict(poly_feats_test)
    lasso_pred = lassoreg.predict(poly_feats_test)
    
    lin_r2score = r2_score(y_test, lin_pred)
    lasso_r2score = r2_score(y_test, lasso_pred)

    return lin_r2score, lasso_r2score

```

## Part 2 - Classification

Here's an application of machine learning that could save your life! For this section of the Project we will be working with the [UCI Mushroom Data Set](http://archive.ics.uci.edu/ml/datasets/Mushroom?ref=datanews.io) stored in `mushrooms.csv`. The data will be used to train a model to predict whether or not a mushroom is poisonous. The following attributes are provided:

*Attribute Information:*

1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s 
2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s 
3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y 
4. bruises?: bruises=t, no=f 
5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s 
6. gill-attachment: attached=a, descending=d, free=f, notched=n 
7. gill-spacing: close=c, crowded=w, distant=d 
8. gill-size: broad=b, narrow=n 
9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y 
10. stalk-shape: enlarging=e, tapering=t 
11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=? 
12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s 
13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s 
14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
16. veil-type: partial=p, universal=u 
17. veil-color: brown=n, orange=o, white=w, yellow=y 
18. ring-number: none=n, one=o, two=t 
19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z 
20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y 
21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y 
22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d

<br>

The data in the mushrooms dataset is currently encoded with strings. These values will need to be encoded to numeric to work with sklearn. We'll use pd.get_dummies to convert the categorical variables into indicator variables. 


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Part 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2
```

## Part 2.1

Using `X_train2` and `y_train2` from the preceeding cell, train a DecisionTreeClassifier with default parameters and random_state=0. What are the 5 most important features found by the decision tree?

As a reminder, the feature names are available in the `X_train2.columns` property, and the order of the features in `X_train2.columns` matches the order of the feature importance values in the classifier's `feature_importances_` property. 

*Returns a list of length 5 containing the feature names in descending order of importance.*


```python
def decision_tree_features():
    from sklearn.tree import DecisionTreeClassifier

    # instantiate classifier
    clf = DecisionTreeClassifier(random_state=0)
    
    # fit classifier to training data
    clf.fit(X_train2, y_train2)
    
    # get all feature importance
    feat_imp = clf.feature_importances_

    # build series out of importances with training columns as index
    fi_series = pd.Series(feat_imp, index=X_train2.columns)
    
    # sort feature importances
    fi_sorted = fi_series.sort_values(ascending=False)
    
    # get 5 max
    max_fi = fi_sorted.iloc[:5]
    
    return list(max_fi.index)
```

### Part 2.2

For this part, use the `validation_curve` function in `sklearn.model_selection` to determine training and test scores for a Support Vector Classifier (`SVC`) with varying parameter values.  Recall that the validation_curve function, in addition to taking an initialized unfitted classifier object, takes a dataset as input and does its own internal train-test splits to compute results.

**Because creating a validation curve requires fitting multiple models, for performance reasons this part will use just a subset of the original mushroom dataset: please use the variables X_subset and y_subset as input to the validation curve function (instead of X_mush and y_mush) to reduce computation time.**

The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel.  So the first step is to create an `SVC` object with default parameters (i.e. `kernel='rbf', C=1`) and `random_state=0`. Recall that the kernel width of the RBF kernel is controlled using the `gamma` parameter.  

With this classifier, and the dataset in X_subset, y_subset, explore the effect of `gamma` on classifier accuracy by using the `validation_curve` function to find the training and test scores for 6 values of `gamma` from `0.0001` to `10` (i.e. `np.logspace(-4,1,6)`). Recall that you can specify what scoring metric you want validation_curve to use by setting the "scoring" parameter.  In this case, we want to use "accuracy" as the scoring metric.

For each level of `gamma`, `validation_curve` will fit 3 models on different subsets of the data, returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.

Find the mean score across the three models for each level of `gamma` for both arrays, creating two arrays of length 6, and return a tuple with the two arrays.

e.g.

if one of your array of scores is

    array([[ 0.5,  0.4,  0.6],
           [ 0.7,  0.8,  0.7],
           [ 0.9,  0.8,  0.8],
           [ 0.8,  0.7,  0.8],
           [ 0.7,  0.6,  0.6],
           [ 0.4,  0.6,  0.5]])
       
it should then become

    array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])

*Returns one tuple of numpy arrays `(training_scores, test_scores)` where each array in the tuple has shape `(6,)`.*


```python
def svc_scores():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    
    v = np.zeros(6)
    
    all_training_scores, all_test_scores = validation_curve(SVC(random_state=0), X_subset, y_subset, "gamma", np.logspace(-4,1,6), scoring='accuracy')
    
    training_scores = np.mean(all_training_scores, axis=1)
    test_scores = np.mean(all_test_scores, axis=1)

    return training_scores, test_scores
```

### Part 2.3

Based on the scores from part 6, what gamma value corresponds to a model that is underfitting (and has the worst test set accuracy)? What gamma value corresponds to a model that is overfitting (and has the worst test set accuracy)? What choice of gamma would be the best choice for a model with good generalization performance on this dataset (high accuracy on both training and test set)? 

*Returns one tuple with the degree values in this order: `(Underfitting, Overfitting, Good_Generalization)` Please note there is only one correct solution.*


```python
def svc_gamma_assessment():
    
    training_scores, test_scores = answer_six()
    
    return # Return your answer
```
