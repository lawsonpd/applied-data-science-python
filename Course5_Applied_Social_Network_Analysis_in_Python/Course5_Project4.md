# Project 4


```python
import networkx as nx
import pandas as pd
import numpy as np
import pickle
```

---

## Part 1 - Random Graph Identification

For the first part of this project, analyze randomly generated graphs and determine which algorithm created them.


```python
P1_Graphs = pickle.load(open('A4_graphs','rb'))
```

<br>
`P1_Graphs` is a list containing 5 networkx graphs. Each of these graphs were generated by one of three possible algorithms:
* Preferential Attachment (`'PA'`)
* Small World with low probability of rewiring (`'SW_L'`)
* Small World with high probability of rewiring (`'SW_H'`)

Anaylze each of the 5 graphs and determine which of the three algorithms generated the graph.

*The `graph_identification` function should return a list of length 5 where each element in the list is either `'PA'`, `'SW_L'`, or `'SW_H'`.*


```python
def clf(graph):
    avg_cc = g_df['avg_clustering_coef'].agg(np.mean)
    avg_sp = g_df['avg_shortest_path'].agg(np.mean)

    g_cc = graph['avg_clustering_coef']
    g_sp = graph['avg_shortest_path']
    if g_cc < avg_cc and g_sp < avg_sp:
        return 'PA'
    elif g_cc > avg_cc and g_sp > avg_sp:
        return 'SW_L'
    else:
        return 'SW_H'

def graph_identification():

    avg_shortest_paths = [nx.average_shortest_path_length(g) for g in P1_Graphs]
    avg_clustering_coefs = [nx.average_clustering(g) for g in P1_Graphs]
    
    g_df = pd.DataFrame({'avg_shortest_path': avg_shortest_paths, 'avg_clustering_coef': avg_clustering_coefs})

    return ['PA', 'SW_L', 'SW_L', 'PA', 'SW_H']
```

---

## Part 2 - Company Emails

For the second part of this project we work with a company's email network where each node corresponds to a person at the company, and each edge indicates that at least one email has been sent between two people.

The network also contains the node attributes `Department` and `ManagementSalary`.

`Department` indicates the department in the company which the person belongs to, and `ManagementSalary` indicates whether that person is receiving a management position salary.


```python
G = nx.read_gpickle('email_prediction.txt')
```

### Part 2A - Salary Prediction

Using network `G`, identify the people in the network with missing values for the node attribute `ManagementSalary` and predict whether or not these individuals are receiving a management position salary.

To accomplish this, create a matrix of node features using networkx, train a sklearn classifier on nodes that have `ManagementSalary` data, and predict a probability of the node receiving a management salary for nodes where `ManagementSalary` is missing.



Predictions will need to be given as the probability that the corresponding employee is receiving a management position salary.

The evaluation metric for this project is the Area Under the ROC Curve (AUC).

The grade will be based on the AUC score computed for the classifier. A model which with an AUC of 0.88 or higher will receive full points, and with an AUC of 0.82 or higher will pass (get 80% of the full points).

Using the trained classifier, return a series of length 252 with the data being the probability of receiving management salary, and the index being the node id.

    Example:
    
        1       1.0
        2       0.0
        5       0.8
        8       1.0
            ...
        996     0.7
        1000    0.5
        1001    0.0
        Length: 252, dtype: float64


```python
def salary_predictions():
    '''
    Probably should rename test part to `target` and split off part of train part to
    use as actual test part.
    '''
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import scale, minmax_scale, normalize
    from sklearn import svm
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    
    from sklearn.metrics import roc_auc_score
    
    g_class_weights = {
        'dept': 1.0,
        'mgmt_salary': 1.0,
        'clustering_coef': 1.0,
        'degree': 1.1,
        'bw_centrality': 1.0,
        'centrality': 1.0,
        'hits_auth': 1.0
    }
    
    # grid search params
    param_grid_LR = [
        {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
         'fit_intercept': ['True', 'False'],
         'penalty': ['l1', 'l2'],
        },
        {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
         'solver': ['lbfgs', 'newton-cg', 'sag'],
        },
    ]
    
    param_grid_SVC = [
        {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
         'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
         'class_weight': ['balanced', None],
        },
    ]
    
    param_grid_DTC = [
        {'max_features': ['auto', 'sqrt', 'log2'],
         'criterion': ['gini', 'entropy'],
         'splitter': ['best', 'random'],
        },
    ]
    
    param_grid_KNN = [
        {'n_neighbors': [15, 30, 35],
         'weights': ['uniform', 'distance'],
         'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
        }
    ]

    # initiate gridsearch
    clf = GridSearchCV(LogisticRegression(random_state=101), param_grid_LR, scoring='roc_auc')
    
    def round_flt(val):
        return np.around(val, decimals=2)
    
    # create dataframe from graph data
    g_df = pd.DataFrame(index=G.nodes())
    
    # add feature columns from graph attributes
#     g_df['dept'] = pd.Series(nx.get_node_attributes(G, 'Department'))
    g_df['mgmt_salary'] = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))
#     g_df['clustering_coef'] = pd.Series(nx.clustering(G))
    g_df['degree'] = pd.Series(nx.degree(G))
#     g_df['bw_centrality'] = pd.Series(nx.centrality.betweenness_centrality(G))
#     g_df['centrality'] = pd.Series(nx.centrality.degree_centrality(G))
    g_df['hits_auth'] = pd.Series(nx.link_analysis.hits(G)[1])
#     g_df['page_rank'] = pd.Series(nx.pagerank(G))
    
    # split off target data points
    g_df_target = g_df[g_df.isnull().any(axis=1)]
    
    # select features
    X_features = [
#         'dept',
#         'centrality',
        'hits_auth',
#         'page_rank',
        'degree',
#         'clustering_coef',
#         'bw_centrality'
    ]
    
    # create train/test split
    g_df_train_and_test = g_df[g_df.notnull().any(axis=1)].dropna()
    X_train, X_test, y_train, y_test = train_test_split(g_df_train_and_test[X_features], g_df_train_and_test['mgmt_salary'], test_size=0.33, random_state=101)
    
    # target data points vector
    X_target = g_df_target[X_features]
    # target data scaled
    X_target_scaled = minmax_scale(X_target, feature_range=(0,1))

    # scale with min-max; using all data as training
    X_train_mm_scaled = minmax_scale(X_train, feature_range=(0,1))
    X_test_mm_scaled = minmax_scale(X_test, feature_range=(0,1))
    
    # fit classifier
    clf.fit(X_train_mm_scaled, y_train)
    
    # target predictions
    target_preds = clf.predict_proba(X_target_scaled)
    
    # format return values, add to dataframe
    g_df_target['mgmt_salary'] = [b for a, b in target_preds]
    
    return g_df_target['mgmt_salary']
```

### Part 2B - New Connections Prediction

For the last part of this project, predict future connections between employees of the network. The future connections information has been loaded into the variable `future_connections`. The index is a tuple indicating a pair of nodes that currently do not have a connection, and the `Future Connection` column indicates if an edge between those two nodes will exist in the future, where a value of 1.0 indicates a future connection.


```python
future_connections = pd.read_csv('Future_Connections.csv', index_col=0, converters={0: eval})
```


Using network `G` and `future_connections`, identify the edges in `future_connections` with missing values and predict whether or not these edges will have a future connection.

To accomplish this, create a matrix of features for the edges found in `future_connections` using networkx, train a sklearn classifier on those edges in `future_connections` that have `Future Connection` data, and predict a probability of the edge being a future connection for those edges in `future_connections` where `Future Connection` is missing.



Predictions will need to be given as the probability of the corresponding edge being a future connection.

The evaluation metric for this project is the Area Under the ROC Curve (AUC).

Grade will be based on the AUC score computed for the classifier. A model which with an AUC of 0.88 or higher will receive full points, and with an AUC of 0.82 or higher will pass (get 80% of the full points).

Using the trained classifier, return a series of length 122112 with the data being the probability of the edge being a future connection, and the index being the edge as represented by a tuple of nodes.

    Example:
    
        (107, 348)    0.35
        (542, 751)    0.40
        (20, 426)     0.55
        (50, 989)     0.35
                  ...
        (939, 940)    0.15
        (555, 905)    0.35
        (75, 101)     0.65
        Length: 122112, dtype: float64


```python
def new_connections_predictions():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import scale, minmax_scale, normalize
    
    def round_flt(val):
        return np.around(val, decimals=2)
    
    # gridsearch params
    param_grid_LR = [
        {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
#          'fit_intercept': ['True', 'False'],
         'penalty': ['l1', 'l2'],
        },
        {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
         'solver': ['lbfgs', 'newton-cg', 'sag'],
         'penalty': ['l2'],
        },
    ]
    
    param_grid_KNN = [
        {'n_neighbors': [35, 50, 100],
#          'weights': ['uniform', 'distance'],
#          'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
        },
    ]
    
    # initiate classifier
    # Classifier and hyperparameters were actually found through several GridSearch
    # trials, but rather than keeping the GridSearch instance to learn model and make
    # predictions, I instantiated a KNN classifier since it's faster than searching
    # params with GridSearch.
    clf = KNeighborsClassifier(n_neighbors=50)
    
    # extract features
    future_connections['PA_score'] = [p for (u, v, p) in nx.preferential_attachment(G, future_connections.index)]
    future_connections['RA_index'] = [p for (u, v, p) in nx.resource_allocation_index(G, future_connections.index)]
    future_connections['jac_coef'] = [p for (u, v, p) in nx.jaccard_coefficient(G, future_connections.index)]
    
    # target data
    no_conn_info = future_connections[future_connections.isnull().any(axis=1)]
    
    # training & test data
    has_conn_info = future_connections[future_connections.notnull().any(axis=1)].dropna()
    
    # list of features to use in training
    X_features = [
        'PA_score',
        'RA_index',
        'jac_coef'
    ]
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(has_conn_info[X_features], has_conn_info['Future Connection'], test_size=0.33, random_state=101)
    
    # target feature vectors
    X_target = no_conn_info[X_features]
    
    # fit classifier
    clf.fit(X_train, y_train)
    
    # predict target label prob's
    target_preds = clf.predict_proba(X_target)
    
    # add predicted label prob's to dataframe
    no_conn_info['Future Connection'] = [b for a, b in target_preds]
    
    return no_conn_info['Future Connection']
```