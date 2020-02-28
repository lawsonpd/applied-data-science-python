
# Loading Graphs in NetworkX


```python
import networkx as nx
import numpy as np
import pandas as pd
%matplotlib notebook

# Instantiate the graph
G1 = nx.Graph()
# add node/edge pairs
G1.add_edges_from([(0, 1),
                   (0, 2),
                   (0, 3),
                   (0, 5),
                   (1, 3),
                   (1, 6),
                   (3, 4),
                   (4, 5),
                   (4, 7),
                   (5, 8),
                   (8, 9)])

# draw the network G1
nx.draw_networkx(G1)
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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UILe559/TgDcpEWfxQelpKSwatWqvOOOOwKm96q8nTlzhuHh4Rw/frzpUjxGAVCsUgCUoHfu3Dmef/75fPrpp02XIlKkr7/+mgA4YcIE06X4pV9++YUA+Pvvv5suxWMUAMUqBUARks8++yyrV6/OM2fOmC5FpEj9+/dnREQEk5KSTJfid8j9xqgAACAASURBVIYNG8bq1aszJyfHdCkeowAoVikAipDctm0bAfCTTz4xXYpIkc6ePctLL72UTZo0YWpqquly/Mp1113HW265xXQZHqUAKFYpAIrkuuGGG3jllVeaLkOkWFu3bmWVKlV47733ajygizIyMmi32zlmzBjTpXiUAqBYpQAokmv69OkEwPXr15suRaRYeZOWJmvRcpf89ttvBMBVq1aZLsWjFADFKgVAkVyZmZmsU6cO+/fvb7oUkRI9+uijrFixIjds2GC6FJ83cuRIRkZGMisry3QpHqUAKFYpAIoUMGTIEJ533nk8ffq06VJEipWens6WLVuyWbNmeq+WokuXLuzatavpMjxOAVCsUgAUKWDnzp202Wz6eE183qZNm1ipUiU+9NBDpkvxWVlZWaxSpQpHjRpluhSPUwAUqxQARQrp1q0b27Zta7oMkVJ9/PHHBMApU6aYLsUnrVy5kgC4dOlS06V4nAKgWKUAKFLIrFmzCIBr1qwxXYpIqXr37s3KlStz8+bNpkvxOW+++SYrVqzIjIwM06V4nAKgWKUAKFJIVlYW69evz8cee8x0KSKlSktLY9OmTdmqVSstZF7IzTffzOuvv950GV6hAChWKQCKFOHll19mlSpVtOCu+IX169fTbrezb9++pkvxGdnZ2axWrRqHDx9uuhSvUAAUqxQARYqwZ88ehoSE8L333jNdiohLJk2aRAD88ssvTZfiE9atW0cAXLRokelSvEIBUKxSABQpxi233MJLLrlEOy6IX3A4HLz77rsZGRnJ7du3my7HuHHjxrFChQoB+7G4AqBYpQAoUowffviBAJiYmGi6FBGXnDp1ijExMbzssst47tw50+UY1bNnT3bs2NF0GV6jAChWKQCKFCM7O5uNGjXSOmviV9asWcMKFSpwwIABpksxxuFwsFatWhw6dKjpUrxGAVCsUgAUKUF8fDwrVqzIEydOmC5FxGXjx48nAM6cOdN0KUZs2rSJAPjTTz+ZLsVrFADFKgVA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" width="640">


### Adjacency List

`G_adjlist.txt` is the adjaceny list representation of G1.

It can be read as follows:
* `0 1 2 3 5` $\rightarrow$ node `0` is adjacent to nodes `1, 2, 3, 5`
* `1 3 6` $\rightarrow$ node `1` is (also) adjacent to nodes `3, 6`
* `2` $\rightarrow$ node `2` is (also) adjacent to no new nodes
* `3 4` $\rightarrow$ node `3` is (also) adjacent to node `4` 

and so on. Note that adjacencies are only accounted for once (e.g. node `2` is adjacent to node `0`, but node `0` is not listed in node `2`'s row, because that edge has already been accounted for in node `0`'s row).


```python
!cat G_adjlist.txt
```

    0 1 2 3 5
    1 3 6
    2
    3 4
    4 5 7
    5 8
    6
    7
    8 9
    9


If we read in the adjacency list using `nx.read_adjlist`, we can see that it matches `G1`.


```python
G2 = nx.read_adjlist('G_adjlist.txt', nodetype=int)
G2.edges()
```




    [(0, 1),
     (0, 2),
     (0, 3),
     (0, 5),
     (1, 3),
     (1, 6),
     (3, 4),
     (5, 4),
     (5, 8),
     (4, 7),
     (8, 9)]



### Adjacency Matrix

The elements in an adjacency matrix indicate whether pairs of vertices are adjacent or not in the graph. Each node has a corresponding row and column. For example, row `0`, column `1` corresponds to the edge between node `0` and node `1`.  

Reading across row `0`, there is a '`1`' in columns `1`, `2`, `3`, and `5`, which indicates that node `0` is adjacent to nodes 1, 2, 3, and 5


```python
G_mat = np.array([[0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
G_mat
```




    array([[0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
           [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
           [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])



If we convert the adjacency matrix to a networkx graph using `nx.Graph`, we can see that it matches G1.


```python
G3 = nx.Graph(G_mat)
G3.edges()
```




    [(0, 1),
     (0, 2),
     (0, 3),
     (0, 5),
     (1, 3),
     (1, 6),
     (3, 4),
     (4, 5),
     (4, 7),
     (5, 8),
     (8, 9)]



### Edgelist

The edge list format represents edge pairings in the first two columns. Additional edge attributes can be added in subsequent columns. Looking at `G_edgelist.txt` this is the same as the original graph `G1`, but now each edge has a weight. 

For example, from the first row, we can see the edge between nodes `0` and `1`, has a weight of `4`.


```python
!cat G_edgelist.txt
```

    0 1 4
    0 2 3
    0 3 2
    0 5 6
    1 3 2
    1 6 5
    3 4 3
    4 5 1
    4 7 2
    5 8 6
    8 9 1


Using `read_edgelist` and passing in a list of tuples with the name and type of each edge attribute will create a graph with our desired edge attributes.


```python
G4 = nx.read_edgelist('G_edgelist.txt', data=[('Weight', int)])

G4.edges(data=True)
```




    [('0', '1', {'Weight': 4}),
     ('0', '2', {'Weight': 3}),
     ('0', '3', {'Weight': 2}),
     ('0', '5', {'Weight': 6}),
     ('1', '3', {'Weight': 2}),
     ('1', '6', {'Weight': 5}),
     ('3', '4', {'Weight': 3}),
     ('5', '4', {'Weight': 1}),
     ('5', '8', {'Weight': 6}),
     ('4', '7', {'Weight': 2}),
     ('8', '9', {'Weight': 1})]



### Pandas DataFrame

Graphs can also be created from pandas dataframes if they are in edge list format.


```python
G_df = pd.read_csv('G_edgelist.txt', delim_whitespace=True, 
                   header=None, names=['n1', 'n2', 'weight'])
G_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n1</th>
      <th>n2</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8</td>
      <td>9</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
G5 = nx.from_pandas_dataframe(G_df, 'n1', 'n2', edge_attr='weight')
G5.edges(data=True)
```




    [(0, 1, {'weight': 4}),
     (0, 2, {'weight': 3}),
     (0, 3, {'weight': 2}),
     (0, 5, {'weight': 6}),
     (1, 3, {'weight': 2}),
     (1, 6, {'weight': 5}),
     (3, 4, {'weight': 3}),
     (5, 4, {'weight': 1}),
     (5, 8, {'weight': 6}),
     (4, 7, {'weight': 2}),
     (8, 9, {'weight': 1})]



### Chess Example

Now let's load in a more complex graph and perform some basic analysis on it.

We will be looking at chess_graph.txt, which is a directed graph of chess games in edge list format.


```python
!head -5 chess_graph.txt
```

    1 2 0	885635999.999997
    1 3 0	885635999.999997
    1 4 0	885635999.999997
    1 5 1	885635999.999997
    1 6 0	885635999.999997


Each node is a chess player, and each edge represents a game. The first column with an outgoing edge corresponds to the white player, the second column with an incoming edge corresponds to the black player.

The third column, the weight of the edge, corresponds to the outcome of the game. A weight of 1 indicates white won, a 0 indicates a draw, and a -1 indicates black won.

The fourth column corresponds to approximate timestamps of when the game was played.

We can read in the chess graph using `read_edgelist`, and tell it to create the graph using a `nx.MultiDiGraph`.


```python
chess = nx.read_edgelist('chess_graph.txt', data=[('outcome', int), ('timestamp', float)], 
                         create_using=nx.MultiDiGraph())
```


```python
chess.is_directed(), chess.is_multigraph()
```




    (True, True)




```python
chess.edges(data=True)
```




    [('1', '2', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('1', '3', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('1', '4', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('1', '5', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('1', '6', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('1', '807', {'outcome': 0, 'timestamp': 896148000.000003}),
     ('1', '454', {'outcome': 0, 'timestamp': 896148000.000003}),
     ('1', '827', {'outcome': 0, 'timestamp': 901403999.999997}),
     ('1', '1240', {'outcome': 0, 'timestamp': 906660000.0}),
     ('1', '680', {'outcome': 0, 'timestamp': 906660000.0}),
     ('1', '166', {'outcome': -1, 'timestamp': 906660000.0}),
     ('1', '1241', {'outcome': 0, 'timestamp': 906660000.0}),
     ('1', '1242', {'outcome': 0, 'timestamp': 906660000.0}),
     ('1', '808', {'outcome': 0, 'timestamp': 925055999.999997}),
     ('1', '819', {'outcome': 0, 'timestamp': 925055999.999997}),
     ('1', '448', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('1', '1214', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('1', '1217', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('1', '2454', {'outcome': 0, 'timestamp': 938196000.0}),
     ('1', '925', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('1', '91', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('1', '4477', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('1', '1363', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('1', '3644', {'outcome': 1, 'timestamp': 1098504000.0}),
     ('1', '1615', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('1', '1051', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('2', '3', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('2', '1166', {'outcome': 1, 'timestamp': 904032000.000003}),
     ('2', '303', {'outcome': 1, 'timestamp': 904032000.000003}),
     ('2', '926', {'outcome': 1, 'timestamp': 904032000.000003}),
     ('2', '1117', {'outcome': 0, 'timestamp': 904032000.000003}),
     ('2', '2010', {'outcome': -1, 'timestamp': 930312000.0}),
     ('2', '172', {'outcome': 0, 'timestamp': 930312000.0}),
     ('2', '172', {'outcome': 1, 'timestamp': 1053828000.0}),
     ('2', '2240', {'outcome': 0, 'timestamp': 930312000.0}),
     ('2', '2240', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('2', '371', {'outcome': 0, 'timestamp': 935568000.000003}),
     ('2', '14', {'outcome': -1, 'timestamp': 935568000.000003}),
     ('2', '277', {'outcome': -1, 'timestamp': 946080000.0}),
     ('2', '152', {'outcome': 1, 'timestamp': 985500000.0}),
     ('2', '23', {'outcome': 1, 'timestamp': 993384000.0}),
     ('2', '1439', {'outcome': -1, 'timestamp': 993384000.0}),
     ('2', '610', {'outcome': 0, 'timestamp': 1011780000.0}),
     ('2', '3991', {'outcome': 0, 'timestamp': 1048572000.0}),
     ('2', '421', {'outcome': 1, 'timestamp': 1059084000.0}),
     ('2', '462', {'outcome': 1, 'timestamp': 1059084000.0}),
     ('2', '483', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('2', '879', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('2', '456', {'outcome': 1, 'timestamp': 1066968000.0}),
     ('2', '456', {'outcome': 0, 'timestamp': 1109016000.0}),
     ('2', '623', {'outcome': 0, 'timestamp': 1066968000.0}),
     ('2', '477', {'outcome': 0, 'timestamp': 1074852000.0}),
     ('2', '3776', {'outcome': 1, 'timestamp': 1082736000.0}),
     ('2', '673', {'outcome': 1, 'timestamp': 1082736000.0}),
     ('2', '580', {'outcome': -1, 'timestamp': 1082736000.0}),
     ('2', '64', {'outcome': 0, 'timestamp': 1082736000.0}),
     ('2', '2028', {'outcome': 1, 'timestamp': 1085364000.0}),
     ('2', '1677', {'outcome': 1, 'timestamp': 1087992000.0}),
     ('2', '112', {'outcome': 0, 'timestamp': 1090620000.0}),
     ('2', '545', {'outcome': 1, 'timestamp': 1090620000.0}),
     ('2', '545', {'outcome': 0, 'timestamp': 1119528000.0}),
     ('2', '639', {'outcome': 0, 'timestamp': 1093248000.0}),
     ('2', '412', {'outcome': 0, 'timestamp': 1109016000.0}),
     ('2', '983', {'outcome': 0, 'timestamp': 1119528000.0}),
     ('2', '1529', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('2', '418', {'outcome': 1, 'timestamp': 1124784000.0}),
     ('2', '621', {'outcome': 0, 'timestamp': 1124784000.0}),
     ('2', '74', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('2', '3830', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('2', '1184', {'outcome': 0, 'timestamp': 1130040000.0}),
     ('2', '9', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('2', '3989', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('2', '3465', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('2', '7200', {'outcome': 1, 'timestamp': 1145808000.0}),
     ('2', '90', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('3', '89', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('3', '172', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('3', '4', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('3', '236', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('3', '6', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('3', '542', {'outcome': 0, 'timestamp': 890892000.0}),
     ('3', '602', {'outcome': 0, 'timestamp': 890892000.0}),
     ('3', '602', {'outcome': 1, 'timestamp': 956591999.999997}),
     ('3', '680', {'outcome': -1, 'timestamp': 901403999.999997}),
     ('3', '117', {'outcome': 1, 'timestamp': 901403999.999997}),
     ('3', '1035', {'outcome': 0, 'timestamp': 901403999.999997}),
     ('3', '1035', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('3', '1', {'outcome': 0, 'timestamp': 906660000.0}),
     ('3', '1', {'outcome': 0, 'timestamp': 925055999.999997}),
     ('3', '1181', {'outcome': 1, 'timestamp': 906660000.0}),
     ('3', '91', {'outcome': 1, 'timestamp': 906660000.0}),
     ('3', '91', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('3', '59', {'outcome': 0, 'timestamp': 906660000.0}),
     ('3', '1242', {'outcome': 0, 'timestamp': 906660000.0}),
     ('3', '808', {'outcome': 0, 'timestamp': 925055999.999997}),
     ('3', '808', {'outcome': 0, 'timestamp': 956591999.999997}),
     ('3', '9', {'outcome': 0, 'timestamp': 938196000.0}),
     ('3', '9', {'outcome': 0, 'timestamp': 956591999.999997}),
     ('3', '9', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('3', '1472', {'outcome': -1, 'timestamp': 938196000.0}),
     ('3', '453', {'outcome': 0, 'timestamp': 1019664000.0}),
     ('3', '453', {'outcome': 1, 'timestamp': 1032804000.0}),
     ('3', '453', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('3', '1432', {'outcome': 1, 'timestamp': 1019664000.0}),
     ('3', '1125', {'outcome': 0, 'timestamp': 1032804000.0}),
     ('3', '516', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('3', '448', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('3', '1241', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('3', '807', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('3', '1363', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('3', '1406', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('3', '1123', {'outcome': -1, 'timestamp': 1101132000.0}),
     ('3', '18', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('3', '3934', {'outcome': -1, 'timestamp': 1119528000.0}),
     ('3', '1809', {'outcome': 0, 'timestamp': 1119528000.0}),
     ('3', '4234', {'outcome': 0, 'timestamp': 1119528000.0}),
     ('3', '3058', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('3', '1066', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('3', '118', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('3', '4182', {'outcome': 0, 'timestamp': 1124784000.0}),
     ('3', '681', {'outcome': 0, 'timestamp': 1124784000.0}),
     ('3', '394', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('4', '89', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('4', '172', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('4', '236', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('4', '91', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('4', '992', {'outcome': 1, 'timestamp': 1051200000.0}),
     ('4', '1245', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('5', '4', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('5', '236', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('5', '6', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('5', '115', {'outcome': -1, 'timestamp': 998640000.000003}),
     ('5', '3318', {'outcome': -1, 'timestamp': 998640000.000003}),
     ('5', '56', {'outcome': 0, 'timestamp': 998640000.000003}),
     ('5', '2266', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('5', '4637', {'outcome': -1, 'timestamp': 1103760000.0}),
     ('5', '4628', {'outcome': -1, 'timestamp': 1135296000.0}),
     ('5', '2677', {'outcome': 0, 'timestamp': 1135296000.0}),
     ('6', '4', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('6', '91', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('6', '453', {'outcome': 1, 'timestamp': 890892000.0}),
     ('6', '3', {'outcome': 0, 'timestamp': 890892000.0}),
     ('6', '454', {'outcome': 0, 'timestamp': 890892000.0}),
     ('6', '182', {'outcome': 0, 'timestamp': 896148000.000003}),
     ('6', '118', {'outcome': 1, 'timestamp': 896148000.000003}),
     ('6', '680', {'outcome': 1, 'timestamp': 901403999.999997}),
     ('6', '142', {'outcome': 1, 'timestamp': 901403999.999997}),
     ('6', '1035', {'outcome': 0, 'timestamp': 901403999.999997}),
     ('6', '818', {'outcome': 0, 'timestamp': 906660000.0}),
     ('6', '139', {'outcome': 1, 'timestamp': 906660000.0}),
     ('6', '1262', {'outcome': 1, 'timestamp': 906660000.0}),
     ('6', '257', {'outcome': 0, 'timestamp': 946080000.0}),
     ('6', '586', {'outcome': 0, 'timestamp': 946080000.0}),
     ('6', '84', {'outcome': 1, 'timestamp': 985500000.0}),
     ('6', '1711', {'outcome': 0, 'timestamp': 985500000.0}),
     ('6', '214', {'outcome': 1, 'timestamp': 985500000.0}),
     ('6', '90', {'outcome': -1, 'timestamp': 1024920000.0}),
     ('6', '90', {'outcome': 0, 'timestamp': 1024920000.0}),
     ('6', '332', {'outcome': 1, 'timestamp': 1024920000.0}),
     ('6', '332', {'outcome': 1, 'timestamp': 1024920000.0}),
     ('6', '1166', {'outcome': 1, 'timestamp': 1032804000.0}),
     ('6', '1556', {'outcome': 1, 'timestamp': 1032804000.0}),
     ('6', '440', {'outcome': 0, 'timestamp': 1032804000.0}),
     ('6', '500', {'outcome': 1, 'timestamp': 1032804000.0}),
     ('6', '913', {'outcome': 0, 'timestamp': 1082736000.0}),
     ('6', '1308', {'outcome': -1, 'timestamp': 1082736000.0}),
     ('6', '2972', {'outcome': 0, 'timestamp': 1082736000.0}),
     ('6', '4302', {'outcome': -1, 'timestamp': 1101132000.0}),
     ('6', '4029', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('6', '4862', {'outcome': -1, 'timestamp': 1122156000.0}),
     ('6', '1282', {'outcome': -1, 'timestamp': 1122156000.0}),
     ('6', '3737', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('6', '5919', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('6', '3497', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('6', '4883', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('6', '6001', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('6', '6023', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('6', '3103', {'outcome': 0, 'timestamp': 1135296000.0}),
     ('6', '39', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('6', '3113', {'outcome': 1, 'timestamp': 1140552000.0}),
     ('6', '128', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('6', '7115', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('6', '870', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('6', '7166', {'outcome': -1, 'timestamp': 1143180000.0}),
     ('6', '2008', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('7', '8', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('7', '9', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('7', '10', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('8', '67', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('8', '67', {'outcome': -1, 'timestamp': 888264000.000003}),
     ('8', '20', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('8', '68', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('8', '69', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('8', '277', {'outcome': 0, 'timestamp': 888264000.000003}),
     ('8', '322', {'outcome': -1, 'timestamp': 888264000.000003}),
     ('8', '323', {'outcome': 0, 'timestamp': 888264000.000003}),
     ('8', '324', {'outcome': 0, 'timestamp': 888264000.000003}),
     ('8', '631', {'outcome': 0, 'timestamp': 919800000.000003}),
     ('8', '946', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('8', '4060', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('9', '113', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('9', '114', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('9', '477', {'outcome': 0, 'timestamp': 890892000.0}),
     ('9', '269', {'outcome': 1, 'timestamp': 890892000.0}),
     ('9', '200', {'outcome': 0, 'timestamp': 919800000.000003}),
     ('9', '436', {'outcome': 0, 'timestamp': 919800000.000003}),
     ('9', '436', {'outcome': 0, 'timestamp': 922428000.0}),
     ('9', '285', {'outcome': -1, 'timestamp': 922428000.0}),
     ('9', '1194', {'outcome': 0, 'timestamp': 922428000.0}),
     ('9', '448', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('9', '807', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('9', '1214', {'outcome': 1, 'timestamp': 927684000.000003}),
     ('9', '1491', {'outcome': 1, 'timestamp': 927684000.000003}),
     ('9', '1167', {'outcome': 1, 'timestamp': 938196000.0}),
     ('9', '2457', {'outcome': 1, 'timestamp': 938196000.0}),
     ('9', '2515', {'outcome': 1, 'timestamp': 938196000.0}),
     ('9', '819', {'outcome': 0, 'timestamp': 938196000.0}),
     ('9', '819', {'outcome': 1, 'timestamp': 948707999.999997}),
     ('9', '91', {'outcome': -1, 'timestamp': 938196000.0}),
     ('9', '1123', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('9', '310', {'outcome': 0, 'timestamp': 951336000.000003}),
     ('9', '1252', {'outcome': 1, 'timestamp': 956591999.999997}),
     ('9', '2247', {'outcome': 0, 'timestamp': 956591999.999997}),
     ('9', '602', {'outcome': 0, 'timestamp': 956591999.999997}),
     ('9', '736', {'outcome': 0, 'timestamp': 977616000.0}),
     ('9', '2413', {'outcome': 0, 'timestamp': 977616000.0}),
     ('9', '1466', {'outcome': 1, 'timestamp': 977616000.0}),
     ('9', '130', {'outcome': 0, 'timestamp': 985500000.0}),
     ('9', '130', {'outcome': -1, 'timestamp': 993384000.0}),
     ('9', '669', {'outcome': -1, 'timestamp': 985500000.0}),
     ('9', '712', {'outcome': 0, 'timestamp': 985500000.0}),
     ('9', '3113', {'outcome': -1, 'timestamp': 985500000.0}),
     ('9', '3113', {'outcome': 1, 'timestamp': 1145808000.0}),
     ('9', '653', {'outcome': -1, 'timestamp': 993384000.0}),
     ('9', '330', {'outcome': -1, 'timestamp': 993384000.0}),
     ('9', '330', {'outcome': 1, 'timestamp': 1135296000.0}),
     ('9', '731', {'outcome': 0, 'timestamp': 993384000.0}),
     ('9', '81', {'outcome': 1, 'timestamp': 1006524000.0}),
     ('9', '3139', {'outcome': 1, 'timestamp': 1006524000.0}),
     ('9', '127', {'outcome': 0, 'timestamp': 1006524000.0}),
     ('9', '3101', {'outcome': 1, 'timestamp': 1009152000.0}),
     ('9', '1929', {'outcome': -1, 'timestamp': 1009152000.0}),
     ('9', '889', {'outcome': -1, 'timestamp': 1024920000.0}),
     ('9', '223', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('9', '3038', {'outcome': 1, 'timestamp': 1038060000.0}),
     ('9', '3135', {'outcome': 1, 'timestamp': 1038060000.0}),
     ('9', '128', {'outcome': 1, 'timestamp': 1038060000.0}),
     ('9', '3505', {'outcome': 1, 'timestamp': 1048572000.0}),
     ('9', '635', {'outcome': -1, 'timestamp': 1048572000.0}),
     ('9', '453', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('9', '87', {'outcome': 1, 'timestamp': 1051200000.0}),
     ('9', '1035', {'outcome': -1, 'timestamp': 1051200000.0}),
     ('9', '3201', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('9', '817', {'outcome': 1, 'timestamp': 1056456000.0}),
     ('9', '1927', {'outcome': -1, 'timestamp': 1072224000.0}),
     ('9', '4543', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('9', '4545', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('9', '4544', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('9', '3524', {'outcome': 0, 'timestamp': 1135296000.0}),
     ('9', '39', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('9', '4060', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('9', '1117', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('9', '1676', {'outcome': 1, 'timestamp': 1145808000.0}),
     ('9', '4549', {'outcome': 1, 'timestamp': 1145808000.0}),
     ('9', '4542', {'outcome': 1, 'timestamp': 1145808000.0}),
     ('10', '114', {'outcome': -1, 'timestamp': 985500000.0}),
     ('11', '12', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('12', '48', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('12', '517', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('12', '1932', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('12', '115', {'outcome': 1, 'timestamp': 948707999.999997}),
     ('12', '93', {'outcome': 1, 'timestamp': 948707999.999997}),
     ('12', '278', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('12', '2070', {'outcome': 1, 'timestamp': 972359999.999997}),
     ('12', '974', {'outcome': 1, 'timestamp': 980243999.999997}),
     ('12', '3658', {'outcome': 1, 'timestamp': 1030176000.0}),
     ('12', '1796', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('12', '77', {'outcome': 0, 'timestamp': 1090620000.0}),
     ('12', '1087', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('12', '224', {'outcome': -1, 'timestamp': 1101132000.0}),
     ('12', '3465', {'outcome': 0, 'timestamp': 1101132000.0}),
     ('12', '366', {'outcome': 0, 'timestamp': 1101132000.0}),
     ('12', '34', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('12', '4886', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('12', '25', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('12', '1957', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('12', '1835', {'outcome': 1, 'timestamp': 1124784000.0}),
     ('12', '2526', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('12', '3745', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('12', '170', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('12', '1911', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('12', '1201', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('12', '287', {'outcome': 0, 'timestamp': 1135296000.0}),
     ('12', '3024', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('12', '1177', {'outcome': -1, 'timestamp': 1143180000.0}),
     ('12', '174', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('12', '357', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('13', '14', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('13', '14', {'outcome': 1, 'timestamp': 904032000.000003}),
     ('13', '14', {'outcome': 0, 'timestamp': 911916000.000003}),
     ('13', '280', {'outcome': 0, 'timestamp': 898776000.0}),
     ('13', '280', {'outcome': 0, 'timestamp': 914544000.0}),
     ('13', '280', {'outcome': 0, 'timestamp': 993384000.0}),
     ('13', '280', {'outcome': -1, 'timestamp': 1040688000.0}),
     ('13', '280', {'outcome': 0, 'timestamp': 1043316000.0}),
     ('13', '280', {'outcome': 0, 'timestamp': 1061712000.0}),
     ('13', '280', {'outcome': 1, 'timestamp': 1072224000.0}),
     ('13', '280', {'outcome': 0, 'timestamp': 1124784000.0}),
     ('13', '280', {'outcome': 0, 'timestamp': 1135296000.0}),
     ('13', '275', {'outcome': 1, 'timestamp': 898776000.0}),
     ('13', '275', {'outcome': 0, 'timestamp': 930312000.0}),
     ('13', '275', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('13', '275', {'outcome': 1, 'timestamp': 1040688000.0}),
     ('13', '275', {'outcome': -1, 'timestamp': 1043316000.0}),
     ('13', '275', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('13', '275', {'outcome': 1, 'timestamp': 1066968000.0}),
     ('13', '275', {'outcome': 0, 'timestamp': 1072224000.0}),
     ('13', '275', {'outcome': 0, 'timestamp': 1135296000.0}),
     ('13', '211', {'outcome': 0, 'timestamp': 904032000.000003}),
     ('13', '211', {'outcome': 0, 'timestamp': 1022292000.0}),
     ('13', '211', {'outcome': 1, 'timestamp': 1072224000.0}),
     ('13', '211', {'outcome': 0, 'timestamp': 1124784000.0}),
     ('13', '188', {'outcome': 0, 'timestamp': 904032000.000003}),
     ('13', '461', {'outcome': 1, 'timestamp': 911916000.000003}),
     ('13', '461', {'outcome': 1, 'timestamp': 1098504000.0}),
     ('13', '503', {'outcome': 1, 'timestamp': 911916000.000003}),
     ('13', '503', {'outcome': 1, 'timestamp': 1022292000.0}),
     ('13', '98', {'outcome': 1, 'timestamp': 914544000.0}),
     ('13', '98', {'outcome': 0, 'timestamp': 977616000.0}),
     ('13', '98', {'outcome': 1, 'timestamp': 1032804000.0}),
     ('13', '98', {'outcome': 1, 'timestamp': 1103760000.0}),
     ('13', '801', {'outcome': 0, 'timestamp': 914544000.0}),
     ('13', '801', {'outcome': -1, 'timestamp': 930312000.0}),
     ('13', '801', {'outcome': 1, 'timestamp': 1032804000.0}),
     ('13', '801', {'outcome': 0, 'timestamp': 1061712000.0}),
     ('13', '801', {'outcome': 1, 'timestamp': 1124784000.0}),
     ('13', '180', {'outcome': -1, 'timestamp': 946080000.0}),
     ('13', '731', {'outcome': 0, 'timestamp': 946080000.0}),
     ('13', '257', {'outcome': 0, 'timestamp': 946080000.0}),
     ('13', '913', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('13', '913', {'outcome': 0, 'timestamp': 980243999.999997}),
     ('13', '913', {'outcome': 0, 'timestamp': 1043316000.0}),
     ('13', '1027', {'outcome': 1, 'timestamp': 969732000.0}),
     ('13', '1027', {'outcome': 0, 'timestamp': 1066968000.0}),
     ('13', '673', {'outcome': 1, 'timestamp': 969732000.0}),
     ('13', '1244', {'outcome': 1, 'timestamp': 969732000.0}),
     ('13', '537', {'outcome': 1, 'timestamp': 977616000.0}),
     ('13', '537', {'outcome': 1, 'timestamp': 980243999.999997}),
     ('13', '537', {'outcome': 1, 'timestamp': 1040688000.0}),
     ('13', '537', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('13', '1791', {'outcome': 0, 'timestamp': 977616000.0}),
     ('13', '1791', {'outcome': 1, 'timestamp': 980243999.999997}),
     ('13', '1791', {'outcome': -1, 'timestamp': 993384000.0}),
     ('13', '1791', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('13', '1791', {'outcome': 0, 'timestamp': 1137924000.0}),
     ('13', '1202', {'outcome': 0, 'timestamp': 1003896000.0}),
     ('13', '462', {'outcome': 1, 'timestamp': 1003896000.0}),
     ('13', '467', {'outcome': 0, 'timestamp': 1009152000.0}),
     ('13', '440', {'outcome': 1, 'timestamp': 1009152000.0}),
     ('13', '53', {'outcome': 1, 'timestamp': 1009152000.0}),
     ('13', '371', {'outcome': 1, 'timestamp': 1024920000.0}),
     ('13', '371', {'outcome': 0, 'timestamp': 1040688000.0}),
     ('13', '371', {'outcome': 0, 'timestamp': 1061712000.0}),
     ('13', '371', {'outcome': 1, 'timestamp': 1103760000.0}),
     ('13', '371', {'outcome': 1, 'timestamp': 1124784000.0}),
     ('13', '439', {'outcome': 1, 'timestamp': 1061712000.0}),
     ('13', '500', {'outcome': 0, 'timestamp': 1066968000.0}),
     ('13', '1108', {'outcome': 0, 'timestamp': 1072224000.0}),
     ('13', '1108', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('13', '1108', {'outcome': -1, 'timestamp': 1137924000.0}),
     ('13', '272', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('13', '1025', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('13', '580', {'outcome': 1, 'timestamp': 1124784000.0}),
     ('13', '635', {'outcome': 0, 'timestamp': 1135296000.0}),
     ('13', '2574', {'outcome': 0, 'timestamp': 1137924000.0}),
     ('14', '13', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('14', '13', {'outcome': 0, 'timestamp': 911916000.000003}),
     ('14', '280', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('14', '280', {'outcome': 0, 'timestamp': 898776000.0}),
     ('14', '53', {'outcome': 0, 'timestamp': 898776000.0}),
     ('14', '53', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('14', '455', {'outcome': 1, 'timestamp': 904032000.000003}),
     ('14', '537', {'outcome': -1, 'timestamp': 904032000.000003}),
     ('14', '257', {'outcome': 0, 'timestamp': 904032000.000003}),
     ('14', '637', {'outcome': 0, 'timestamp': 906660000.0}),
     ('14', '731', {'outcome': 1, 'timestamp': 906660000.0}),
     ('14', '731', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('14', '207', {'outcome': 0, 'timestamp': 906660000.0}),
     ('14', '1245', {'outcome': 0, 'timestamp': 906660000.0}),
     ('14', '1178', {'outcome': 1, 'timestamp': 906660000.0}),
     ('14', '1160', {'outcome': 0, 'timestamp': 906660000.0}),
     ('14', '1160', {'outcome': 0, 'timestamp': 1116900000.0}),
     ('14', '2028', {'outcome': -1, 'timestamp': 935568000.000003}),
     ('14', '952', {'outcome': 1, 'timestamp': 935568000.000003}),
     ('14', '275', {'outcome': 0, 'timestamp': 938196000.0}),
     ('14', '275', {'outcome': 0, 'timestamp': 1032804000.0}),
     ('14', '98', {'outcome': 1, 'timestamp': 946080000.0}),
     ('14', '643', {'outcome': 0, 'timestamp': 946080000.0}),
     ('14', '211', {'outcome': 0, 'timestamp': 946080000.0}),
     ('14', '127', {'outcome': 0, 'timestamp': 946080000.0}),
     ('14', '580', {'outcome': -1, 'timestamp': 969732000.0}),
     ('14', '798', {'outcome': 0, 'timestamp': 982872000.000003}),
     ('14', '1013', {'outcome': 0, 'timestamp': 988127999.999997}),
     ('14', '3174', {'outcome': 1, 'timestamp': 988127999.999997}),
     ('14', '522', {'outcome': 1, 'timestamp': 988127999.999997}),
     ('14', '1095', {'outcome': 1, 'timestamp': 988127999.999997}),
     ('14', '573', {'outcome': 0, 'timestamp': 988127999.999997}),
     ('14', '1201', {'outcome': 0, 'timestamp': 988127999.999997}),
     ('14', '595', {'outcome': -1, 'timestamp': 988127999.999997}),
     ('14', '862', {'outcome': 0, 'timestamp': 988127999.999997}),
     ('14', '28', {'outcome': 1, 'timestamp': 998640000.000003}),
     ('14', '3195', {'outcome': 1, 'timestamp': 998640000.000003}),
     ('14', '519', {'outcome': 0, 'timestamp': 998640000.000003}),
     ('14', '500', {'outcome': 0, 'timestamp': 998640000.000003}),
     ('14', '838', {'outcome': 0, 'timestamp': 998640000.000003}),
     ('14', '238', {'outcome': 0, 'timestamp': 1001268000.0}),
     ('14', '574', {'outcome': 0, 'timestamp': 1001268000.0}),
     ('14', '828', {'outcome': 1, 'timestamp': 1011780000.0}),
     ('14', '251', {'outcome': 0, 'timestamp': 1017036000.0}),
     ('14', '510', {'outcome': 0, 'timestamp': 1017036000.0}),
     ('14', '412', {'outcome': 1, 'timestamp': 1019664000.0}),
     ('14', '228', {'outcome': 0, 'timestamp': 1032804000.0}),
     ('14', '259', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('14', '132', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('14', '586', {'outcome': 0, 'timestamp': 1069596000.0}),
     ('14', '561', {'outcome': 0, 'timestamp': 1074852000.0}),
     ('14', '2836', {'outcome': -1, 'timestamp': 1080108000.0}),
     ('14', '684', {'outcome': 0, 'timestamp': 1093248000.0}),
     ('14', '36', {'outcome': 0, 'timestamp': 1095876000.0}),
     ('14', '1430', {'outcome': 1, 'timestamp': 1095876000.0}),
     ('14', '3990', {'outcome': 1, 'timestamp': 1106388000.0}),
     ('14', '387', {'outcome': 0, 'timestamp': 1106388000.0}),
     ('14', '3326', {'outcome': 1, 'timestamp': 1106388000.0}),
     ('14', '3326', {'outcome': 0, 'timestamp': 1111644000.0}),
     ('14', '1169', {'outcome': 0, 'timestamp': 1111644000.0}),
     ('14', '421', {'outcome': 0, 'timestamp': 1114272000.0}),
     ('14', '1925', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('14', '3124', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('14', '983', {'outcome': -1, 'timestamp': 1122156000.0}),
     ('14', '2714', {'outcome': 0, 'timestamp': 1130040000.0}),
     ('14', '414', {'outcome': 0, 'timestamp': 1137924000.0}),
     ('14', '414', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('14', '139', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('14', '3989', {'outcome': 0, 'timestamp': 1137924000.0}),
     ('14', '1566', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('14', '1244', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('14', '3077', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('14', '3406', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('14', '621', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('14', '64', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('14', '1294', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('15', '16', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('15', '1266', {'outcome': 1, 'timestamp': 906660000.0}),
     ('15', '1267', {'outcome': 1, 'timestamp': 906660000.0}),
     ('15', '197', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('15', '254', {'outcome': -1, 'timestamp': 917171999.999997}),
     ('15', '256', {'outcome': -1, 'timestamp': 917171999.999997}),
     ('15', '2710', {'outcome': 1, 'timestamp': 948707999.999997}),
     ('15', '104', {'outcome': -1, 'timestamp': 948707999.999997}),
     ('15', '2371', {'outcome': 0, 'timestamp': 988127999.999997}),
     ('15', '3145', {'outcome': -1, 'timestamp': 988127999.999997}),
     ('15', '3150', {'outcome': -1, 'timestamp': 1127412000.0}),
     ('15', '5176', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('15', '921', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('15', '5177', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('15', '6814', {'outcome': -1, 'timestamp': 1140552000.0}),
     ('17', '18', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('17', '300', {'outcome': 0, 'timestamp': 888264000.000003}),
     ('17', '300', {'outcome': 0, 'timestamp': 919800000.000003}),
     ('17', '301', {'outcome': -1, 'timestamp': 888264000.000003}),
     ('17', '302', {'outcome': -1, 'timestamp': 888264000.000003}),
     ('17', '335', {'outcome': 1, 'timestamp': 919800000.000003}),
     ('17', '335', {'outcome': 1, 'timestamp': 1090620000.0}),
     ('17', '649', {'outcome': 0, 'timestamp': 919800000.000003}),
     ('17', '1221', {'outcome': 0, 'timestamp': 919800000.000003}),
     ('17', '1221', {'outcome': 1, 'timestamp': 951336000.000003}),
     ('17', '157', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('17', '158', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('17', '752', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('17', '774', {'outcome': -1, 'timestamp': 935568000.000003}),
     ('17', '2460', {'outcome': 1, 'timestamp': 938196000.0}),
     ('17', '260', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('17', '1103', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('17', '1466', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('17', '2737', {'outcome': 1, 'timestamp': 951336000.000003}),
     ('17', '334', {'outcome': -1, 'timestamp': 964475999.999997}),
     ('17', '334', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('17', '186', {'outcome': 0, 'timestamp': 964475999.999997}),
     ('17', '155', {'outcome': 1, 'timestamp': 977616000.0}),
     ('17', '700', {'outcome': 1, 'timestamp': 996011999.999997}),
     ('17', '739', {'outcome': 0, 'timestamp': 996011999.999997}),
     ('17', '3262', {'outcome': -1, 'timestamp': 996011999.999997}),
     ('17', '92', {'outcome': 0, 'timestamp': 1001268000.0}),
     ('17', '1623', {'outcome': 1, 'timestamp': 1001268000.0}),
     ('17', '42', {'outcome': 0, 'timestamp': 1003896000.0}),
     ('17', '1160', {'outcome': 1, 'timestamp': 1003896000.0}),
     ('17', '2574', {'outcome': -1, 'timestamp': 1009152000.0}),
     ('17', '2170', {'outcome': -1, 'timestamp': 1009152000.0}),
     ('17', '74', {'outcome': 1, 'timestamp': 1009152000.0}),
     ('17', '1108', {'outcome': 0, 'timestamp': 1014408000.0}),
     ('17', '447', {'outcome': 0, 'timestamp': 1019664000.0}),
     ('17', '1548', {'outcome': -1, 'timestamp': 1019664000.0}),
     ('17', '2496', {'outcome': 0, 'timestamp': 1024920000.0}),
     ('17', '3577', {'outcome': 1, 'timestamp': 1024920000.0}),
     ('17', '2319', {'outcome': -1, 'timestamp': 1024920000.0}),
     ('17', '352', {'outcome': -1, 'timestamp': 1030176000.0}),
     ('17', '2984', {'outcome': 1, 'timestamp': 1030176000.0}),
     ('17', '573', {'outcome': 0, 'timestamp': 1030176000.0}),
     ('17', '228', {'outcome': 1, 'timestamp': 1030176000.0}),
     ('17', '164', {'outcome': -1, 'timestamp': 1040688000.0}),
     ('17', '449', {'outcome': 0, 'timestamp': 1053828000.0}),
     ('17', '4244', {'outcome': 1, 'timestamp': 1072224000.0}),
     ('17', '2667', {'outcome': 1, 'timestamp': 1072224000.0}),
     ('17', '720', {'outcome': -1, 'timestamp': 1072224000.0}),
     ('17', '3137', {'outcome': 1, 'timestamp': 1072224000.0}),
     ('17', '4173', {'outcome': 1, 'timestamp': 1090620000.0}),
     ('17', '165', {'outcome': 1, 'timestamp': 1090620000.0}),
     ('17', '327', {'outcome': 0, 'timestamp': 1090620000.0}),
     ('17', '53', {'outcome': -1, 'timestamp': 1103760000.0}),
     ('17', '939', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('17', '2105', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('17', '3934', {'outcome': -1, 'timestamp': 1137924000.0}),
     ('17', '3189', {'outcome': -1, 'timestamp': 1137924000.0}),
     ('18', '170', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('18', '49', {'outcome': -1, 'timestamp': 898776000.0}),
     ('18', '49', {'outcome': -1, 'timestamp': 938196000.0}),
     ('18', '48', {'outcome': -1, 'timestamp': 898776000.0}),
     ('18', '1120', {'outcome': 1, 'timestamp': 932939999.999997}),
     ('18', '1055', {'outcome': -1, 'timestamp': 932939999.999997}),
     ('18', '2289', {'outcome': -1, 'timestamp': 932939999.999997}),
     ('18', '1283', {'outcome': 0, 'timestamp': 938196000.0}),
     ('18', '1522', {'outcome': 0, 'timestamp': 938196000.0}),
     ('18', '2537', {'outcome': 1, 'timestamp': 938196000.0}),
     ('18', '2458', {'outcome': 1, 'timestamp': 969732000.0}),
     ('18', '723', {'outcome': 1, 'timestamp': 969732000.0}),
     ('18', '2576', {'outcome': 1, 'timestamp': 1030176000.0}),
     ('18', '910', {'outcome': 1, 'timestamp': 1053828000.0}),
     ('18', '2570', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('18', '754', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('18', '3520', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('18', '2468', {'outcome': -1, 'timestamp': 1095876000.0}),
     ('18', '2765', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('18', '3201', {'outcome': 0, 'timestamp': 1101132000.0}),
     ('18', '1942', {'outcome': 1, 'timestamp': 1114272000.0}),
     ('18', '1942', {'outcome': -1, 'timestamp': 1140552000.0}),
     ('18', '5580', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('18', '115', {'outcome': 1, 'timestamp': 1145808000.0}),
     ('18', '3785', {'outcome': -1, 'timestamp': 1145808000.0}),
     ('18', '197', {'outcome': -1, 'timestamp': 1145808000.0}),
     ('19', '7', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('19', '8', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('19', '20', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('19', '21', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('19', '22', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('19', '61', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('19', '2900', {'outcome': 0, 'timestamp': 974988000.000003}),
     ('19', '153', {'outcome': 0, 'timestamp': 974988000.000003}),
     ('19', '2249', {'outcome': 0, 'timestamp': 1114272000.0}),
     ('19', '156', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('19', '220', {'outcome': 0, 'timestamp': 1137924000.0}),
     ('19', '1820', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('19', '66', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('19', '197', {'outcome': 1, 'timestamp': 1145808000.0}),
     ('20', '156', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('20', '167', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('20', '184', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('20', '147', {'outcome': -1, 'timestamp': 1103760000.0}),
     ('20', '4540', {'outcome': -1, 'timestamp': 1137924000.0}),
     ('20', '6012', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('21', '115', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('21', '129', {'outcome': 0, 'timestamp': 890892000.0}),
     ('21', '531', {'outcome': 1, 'timestamp': 890892000.0}),
     ('21', '102', {'outcome': 1, 'timestamp': 890892000.0}),
     ('21', '366', {'outcome': 0, 'timestamp': 890892000.0}),
     ('21', '851', {'outcome': 0, 'timestamp': 896148000.000003}),
     ('21', '344', {'outcome': 1, 'timestamp': 898776000.0}),
     ('21', '279', {'outcome': 0, 'timestamp': 898776000.0}),
     ('21', '999', {'outcome': 1, 'timestamp': 898776000.0}),
     ('21', '142', {'outcome': 1, 'timestamp': 904032000.000003}),
     ('21', '503', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('21', '257', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('21', '257', {'outcome': 0, 'timestamp': 1011780000.0}),
     ('21', '545', {'outcome': 0, 'timestamp': 935568000.000003}),
     ('21', '640', {'outcome': -1, 'timestamp': 935568000.000003}),
     ('21', '1177', {'outcome': 0, 'timestamp': 946080000.0}),
     ('21', '1108', {'outcome': 0, 'timestamp': 1011780000.0}),
     ('21', '1820', {'outcome': 0, 'timestamp': 1019664000.0}),
     ('21', '720', {'outcome': 0, 'timestamp': 1019664000.0}),
     ('21', '696', {'outcome': 0, 'timestamp': 1027548000.0}),
     ('21', '2041', {'outcome': 1, 'timestamp': 1027548000.0}),
     ('21', '281', {'outcome': 0, 'timestamp': 1027548000.0}),
     ('21', '4284', {'outcome': 1, 'timestamp': 1103760000.0}),
     ('21', '725', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('21', '56', {'outcome': 1, 'timestamp': 1103760000.0}),
     ('21', '4753', {'outcome': 1, 'timestamp': 1111644000.0}),
     ('21', '2476', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('21', '2317', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('21', '1428', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('21', '153', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('21', '3849', {'outcome': 0, 'timestamp': 1137924000.0}),
     ('21', '4686', {'outcome': 1, 'timestamp': 1140552000.0}),
     ('21', '4712', {'outcome': 1, 'timestamp': 1140552000.0}),
     ('21', '2796', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('21', '3530', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('21', '2676', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('21', '246', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('22', '20', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('22', '217', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('23', '1', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('23', '2', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('23', '3', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('23', '1439', {'outcome': -1, 'timestamp': 922428000.0}),
     ('23', '1439', {'outcome': -1, 'timestamp': 993384000.0}),
     ('23', '1439', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('23', '2008', {'outcome': 1, 'timestamp': 922428000.0}),
     ('23', '2008', {'outcome': 0, 'timestamp': 993384000.0}),
     ('23', '2008', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('23', '172', {'outcome': -1, 'timestamp': 922428000.0}),
     ('23', '172', {'outcome': 0, 'timestamp': 993384000.0}),
     ('23', '90', {'outcome': 0, 'timestamp': 922428000.0}),
     ('23', '90', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('23', '952', {'outcome': 0, 'timestamp': 922428000.0}),
     ('23', '352', {'outcome': -1, 'timestamp': 993384000.0}),
     ('23', '2718', {'outcome': 1, 'timestamp': 1011780000.0}),
     ('23', '165', {'outcome': -1, 'timestamp': 1048572000.0}),
     ('23', '605', {'outcome': 1, 'timestamp': 1048572000.0}),
     ('23', '2057', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('23', '3473', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('23', '4867', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('23', '4078', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('23', '232', {'outcome': -1, 'timestamp': 1137924000.0}),
     ('24', '25', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('24', '890', {'outcome': 1, 'timestamp': 898776000.0}),
     ('24', '142', {'outcome': 1, 'timestamp': 898776000.0}),
     ('24', '21', {'outcome': 0, 'timestamp': 898776000.0}),
     ('24', '1314', {'outcome': 1, 'timestamp': 906660000.0}),
     ('24', '222', {'outcome': -1, 'timestamp': 911916000.000003}),
     ('24', '372', {'outcome': 0, 'timestamp': 911916000.000003}),
     ('24', '1118', {'outcome': 0, 'timestamp': 914544000.0}),
     ('24', '2009', {'outcome': 1, 'timestamp': 922428000.0}),
     ('24', '733', {'outcome': 1, 'timestamp': 922428000.0}),
     ('24', '462', {'outcome': -1, 'timestamp': 922428000.0}),
     ('24', '1882', {'outcome': 0, 'timestamp': 938196000.0}),
     ('24', '1025', {'outcome': 0, 'timestamp': 969732000.0}),
     ('24', '160', {'outcome': -1, 'timestamp': 1003896000.0}),
     ('24', '242', {'outcome': -1, 'timestamp': 1003896000.0}),
     ('24', '1281', {'outcome': 1, 'timestamp': 1035432000.0}),
     ('24', '1578', {'outcome': 0, 'timestamp': 1035432000.0}),
     ('24', '3290', {'outcome': 1, 'timestamp': 1038060000.0}),
     ('24', '2957', {'outcome': 1, 'timestamp': 1040688000.0}),
     ('24', '710', {'outcome': -1, 'timestamp': 1098504000.0}),
     ('24', '1636', {'outcome': -1, 'timestamp': 1098504000.0}),
     ('24', '2959', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('24', '1652', {'outcome': 0, 'timestamp': 1130040000.0}),
     ('24', '849', {'outcome': 0, 'timestamp': 1130040000.0}),
     ('24', '364', {'outcome': 1, 'timestamp': 1130040000.0}),
     ('24', '2840', {'outcome': 1, 'timestamp': 1140552000.0}),
     ('24', '3814', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('24', '2614', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('24', '3150', {'outcome': 1, 'timestamp': 1145808000.0}),
     ('24', '204', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('25', '30', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('25', '93', {'outcome': -1, 'timestamp': 898776000.0}),
     ('25', '922', {'outcome': 1, 'timestamp': 898776000.0}),
     ('25', '316', {'outcome': -1, 'timestamp': 898776000.0}),
     ('25', '910', {'outcome': 1, 'timestamp': 930312000.0}),
     ('25', '315', {'outcome': 0, 'timestamp': 930312000.0}),
     ('25', '2268', {'outcome': 1, 'timestamp': 930312000.0}),
     ('25', '172', {'outcome': 1, 'timestamp': 1011780000.0}),
     ('25', '90', {'outcome': -1, 'timestamp': 1011780000.0}),
     ('25', '2416', {'outcome': 0, 'timestamp': 1011780000.0}),
     ('25', '115', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('25', '115', {'outcome': -1, 'timestamp': 1069596000.0}),
     ('25', '2681', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('25', '4018', {'outcome': 0, 'timestamp': 1053828000.0}),
     ('25', '2990', {'outcome': 1, 'timestamp': 1053828000.0}),
     ('25', '184', {'outcome': 0, 'timestamp': 1069596000.0}),
     ('25', '1125', {'outcome': -1, 'timestamp': 1093248000.0}),
     ('25', '1719', {'outcome': -1, 'timestamp': 1103760000.0}),
     ('25', '132', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('25', '3743', {'outcome': 1, 'timestamp': 1106388000.0}),
     ('25', '268', {'outcome': 0, 'timestamp': 1106388000.0}),
     ('25', '215', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('25', '4924', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('25', '4215', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('25', '3747', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('25', '6059', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('25', '5522', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('25', '1957', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('25', '1851', {'outcome': 1, 'timestamp': 1135296000.0}),
     ('25', '2673', {'outcome': -1, 'timestamp': 1135296000.0}),
     ('25', '4333', {'outcome': -1, 'timestamp': 1135296000.0}),
     ('25', '6424', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('25', '1177', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('25', '48', {'outcome': 1, 'timestamp': 1140552000.0}),
     ('25', '2152', {'outcome': 0, 'timestamp': 1143180000.0}),
     ('26', '27', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('27', '222', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('27', '107', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('28', '29', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('28', '411', {'outcome': -1, 'timestamp': 890892000.0}),
     ('28', '74', {'outcome': 0, 'timestamp': 890892000.0}),
     ('28', '261', {'outcome': 0, 'timestamp': 890892000.0}),
     ('28', '190', {'outcome': 0, 'timestamp': 890892000.0}),
     ('28', '412', {'outcome': 1, 'timestamp': 890892000.0}),
     ('28', '412', {'outcome': -1, 'timestamp': 1085364000.0}),
     ('28', '56', {'outcome': -1, 'timestamp': 890892000.0}),
     ('28', '891', {'outcome': 1, 'timestamp': 898776000.0}),
     ('28', '174', {'outcome': 1, 'timestamp': 917171999.999997}),
     ('28', '517', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('28', '182', {'outcome': -1, 'timestamp': 917171999.999997}),
     ('28', '897', {'outcome': -1, 'timestamp': 998640000.000003}),
     ('28', '53', {'outcome': -1, 'timestamp': 998640000.000003}),
     ('28', '2995', {'outcome': 1, 'timestamp': 1017036000.0}),
     ('28', '2837', {'outcome': 1, 'timestamp': 1017036000.0}),
     ('28', '2805', {'outcome': 0, 'timestamp': 1085364000.0}),
     ('28', '432', {'outcome': 0, 'timestamp': 1137924000.0}),
     ('29', '131', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('29', '77', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('30', '31', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('30', '651', {'outcome': 0, 'timestamp': 893519999.999997}),
     ('30', '653', {'outcome': 0, 'timestamp': 893519999.999997}),
     ('30', '653', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('30', '653', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('30', '653', {'outcome': 0, 'timestamp': 1114272000.0}),
     ('30', '653', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('30', '654', {'outcome': 0, 'timestamp': 893519999.999997}),
     ('30', '654', {'outcome': 0, 'timestamp': 925055999.999997}),
     ('30', '654', {'outcome': 0, 'timestamp': 1051200000.0}),
     ('30', '654', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('30', '256', {'outcome': 0, 'timestamp': 893519999.999997}),
     ('30', '54', {'outcome': 0, 'timestamp': 906660000.0}),
     ('30', '483', {'outcome': 0, 'timestamp': 906660000.0}),
     ('30', '502', {'outcome': 0, 'timestamp': 906660000.0}),
     ('30', '573', {'outcome': 0, 'timestamp': 906660000.0}),
     ('30', '232', {'outcome': 0, 'timestamp': 925055999.999997}),
     ('30', '2749', {'outcome': 1, 'timestamp': 951336000.000003}),
     ('30', '1963', {'outcome': 0, 'timestamp': 961848000.0}),
     ('30', '3519', {'outcome': 1, 'timestamp': 1011780000.0}),
     ('30', '21', {'outcome': 0, 'timestamp': 1011780000.0}),
     ('30', '3565', {'outcome': 0, 'timestamp': 1019664000.0}),
     ('30', '662', {'outcome': 0, 'timestamp': 1019664000.0}),
     ('30', '1366', {'outcome': 0, 'timestamp': 1019664000.0}),
     ('30', '663', {'outcome': 1, 'timestamp': 1019664000.0}),
     ('30', '663', {'outcome': 1, 'timestamp': 1048572000.0}),
     ('30', '227', {'outcome': 0, 'timestamp': 1032804000.0}),
     ('30', '952', {'outcome': 0, 'timestamp': 1032804000.0}),
     ('30', '449', {'outcome': -1, 'timestamp': 1032804000.0}),
     ('30', '3023', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('30', '1857', {'outcome': 1, 'timestamp': 1038060000.0}),
     ('30', '3154', {'outcome': 1, 'timestamp': 1048572000.0}),
     ('30', '2823', {'outcome': 1, 'timestamp': 1048572000.0}),
     ('30', '643', {'outcome': -1, 'timestamp': 1059084000.0}),
     ('30', '1677', {'outcome': 0, 'timestamp': 1059084000.0}),
     ('30', '3989', {'outcome': -1, 'timestamp': 1066968000.0}),
     ('30', '421', {'outcome': 1, 'timestamp': 1066968000.0}),
     ('30', '2090', {'outcome': 0, 'timestamp': 1069596000.0}),
     ('30', '2090', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('30', '330', {'outcome': 0, 'timestamp': 1069596000.0}),
     ('30', '674', {'outcome': 1, 'timestamp': 1074852000.0}),
     ('30', '789', {'outcome': 0, 'timestamp': 1098504000.0}),
     ('30', '789', {'outcome': -1, 'timestamp': 1114272000.0}),
     ('30', '3463', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('30', '3463', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('30', '874', {'outcome': 1, 'timestamp': 1101132000.0}),
     ('30', '4553', {'outcome': 1, 'timestamp': 1103760000.0}),
     ('30', '655', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('30', '4569', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('30', '3762', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('30', '3457', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('30', '1859', {'outcome': 1, 'timestamp': 1140552000.0}),
     ('30', '2796', {'outcome': -1, 'timestamp': 1145808000.0}),
     ('30', '272', {'outcome': 0, 'timestamp': 1145808000.0}),
     ('31', '1131', {'outcome': 0, 'timestamp': 914544000.0}),
     ('31', '1728', {'outcome': 0, 'timestamp': 914544000.0}),
     ('31', '333', {'outcome': -1, 'timestamp': 917171999.999997}),
     ('31', '1933', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('31', '252', {'outcome': -1, 'timestamp': 953964000.0}),
     ('31', '330', {'outcome': -1, 'timestamp': 953964000.0}),
     ('31', '239', {'outcome': -1, 'timestamp': 953964000.0}),
     ('31', '2988', {'outcome': 0, 'timestamp': 972359999.999997}),
     ('31', '2009', {'outcome': 1, 'timestamp': 972359999.999997}),
     ('31', '2990', {'outcome': 1, 'timestamp': 972359999.999997}),
     ('31', '1081', {'outcome': 1, 'timestamp': 1124784000.0}),
     ('32', '33', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('32', '34', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('32', '35', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('32', '36', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('32', '37', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('32', '884', {'outcome': 1, 'timestamp': 898776000.0}),
     ('32', '897', {'outcome': 0, 'timestamp': 898776000.0}),
     ('32', '898', {'outcome': 0, 'timestamp': 898776000.0}),
     ('32', '898', {'outcome': 0, 'timestamp': 1056456000.0}),
     ('32', '468', {'outcome': 1, 'timestamp': 898776000.0}),
     ('32', '899', {'outcome': 0, 'timestamp': 898776000.0}),
     ('32', '172', {'outcome': -1, 'timestamp': 906660000.0}),
     ('32', '61', {'outcome': 0, 'timestamp': 906660000.0}),
     ('32', '164', {'outcome': -1, 'timestamp': 906660000.0}),
     ('32', '1357', {'outcome': 1, 'timestamp': 906660000.0}),
     ('32', '922', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('32', '276', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('32', '803', {'outcome': 1, 'timestamp': 927684000.000003}),
     ('32', '803', {'outcome': 0, 'timestamp': 1116900000.0}),
     ('32', '483', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('32', '440', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('32', '886', {'outcome': 0, 'timestamp': 927684000.000003}),
     ('32', '886', {'outcome': 1, 'timestamp': 930312000.0}),
     ('32', '676', {'outcome': 1, 'timestamp': 927684000.000003}),
     ('32', '997', {'outcome': 0, 'timestamp': 969732000.0}),
     ('32', '88', {'outcome': 0, 'timestamp': 977616000.0}),
     ('32', '2184', {'outcome': 1, 'timestamp': 993384000.0}),
     ('32', '3225', {'outcome': 0, 'timestamp': 993384000.0}),
     ('32', '2269', {'outcome': 1, 'timestamp': 993384000.0}),
     ('32', '401', {'outcome': 1, 'timestamp': 1003896000.0}),
     ('32', '960', {'outcome': 0, 'timestamp': 1003896000.0}),
     ('32', '2079', {'outcome': 0, 'timestamp': 1009152000.0}),
     ('32', '1557', {'outcome': 0, 'timestamp': 1009152000.0}),
     ('32', '2230', {'outcome': 1, 'timestamp': 1014408000.0}),
     ('32', '320', {'outcome': 0, 'timestamp': 1014408000.0}),
     ('32', '1862', {'outcome': -1, 'timestamp': 1014408000.0}),
     ('32', '2594', {'outcome': -1, 'timestamp': 1027548000.0}),
     ('32', '1134', {'outcome': 0, 'timestamp': 1027548000.0}),
     ('32', '641', {'outcome': -1, 'timestamp': 1027548000.0}),
     ('32', '1099', {'outcome': 0, 'timestamp': 1030176000.0}),
     ('32', '3853', {'outcome': 1, 'timestamp': 1035432000.0}),
     ('32', '2986', {'outcome': 0, 'timestamp': 1040688000.0}),
     ('32', '471', {'outcome': 1, 'timestamp': 1040688000.0}),
     ('32', '3194', {'outcome': 1, 'timestamp': 1040688000.0}),
     ('32', '1466', {'outcome': 0, 'timestamp': 1040688000.0}),
     ('32', '3679', {'outcome': 0, 'timestamp': 1056456000.0}),
     ('32', '834', {'outcome': 1, 'timestamp': 1056456000.0}),
     ('32', '496', {'outcome': -1, 'timestamp': 1056456000.0}),
     ('32', '496', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('32', '4214', {'outcome': 1, 'timestamp': 1066968000.0}),
     ('32', '3924', {'outcome': 1, 'timestamp': 1072224000.0}),
     ('32', '2667', {'outcome': 1, 'timestamp': 1072224000.0}),
     ('32', '670', {'outcome': 1, 'timestamp': 1087992000.0}),
     ('32', '670', {'outcome': 0, 'timestamp': 1124784000.0}),
     ('32', '3548', {'outcome': 0, 'timestamp': 1087992000.0}),
     ('32', '1232', {'outcome': 0, 'timestamp': 1087992000.0}),
     ('32', '98', {'outcome': -1, 'timestamp': 1093248000.0}),
     ('32', '537', {'outcome': -1, 'timestamp': 1093248000.0}),
     ('32', '959', {'outcome': 0, 'timestamp': 1093248000.0}),
     ('32', '4445', {'outcome': 1, 'timestamp': 1095876000.0}),
     ('32', '4445', {'outcome': 0, 'timestamp': 1119528000.0}),
     ('32', '3101', {'outcome': -1, 'timestamp': 1103760000.0}),
     ('32', '188', {'outcome': 0, 'timestamp': 1103760000.0}),
     ('32', '928', {'outcome': 1, 'timestamp': 1116900000.0}),
     ('32', '153', {'outcome': 1, 'timestamp': 1119528000.0}),
     ('32', '259', {'outcome': 0, 'timestamp': 1124784000.0}),
     ('32', '2131', {'outcome': 1, 'timestamp': 1124784000.0}),
     ('32', '4368', {'outcome': 1, 'timestamp': 1135296000.0}),
     ('32', '189', {'outcome': 0, 'timestamp': 1135296000.0}),
     ('32', '3925', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('32', '3112', {'outcome': 1, 'timestamp': 1140552000.0}),
     ('32', '4060', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('32', '166', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('33', '42', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('33', '43', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('33', '773', {'outcome': 0, 'timestamp': 898776000.0}),
     ('33', '900', {'outcome': 1, 'timestamp': 898776000.0}),
     ('33', '901', {'outcome': 0, 'timestamp': 898776000.0}),
     ('33', '902', {'outcome': 0, 'timestamp': 898776000.0}),
     ('33', '468', {'outcome': 0, 'timestamp': 904032000.000003}),
     ('33', '79', {'outcome': 1, 'timestamp': 904032000.000003}),
     ('33', '200', {'outcome': 0, 'timestamp': 911916000.000003}),
     ('33', '200', {'outcome': 0, 'timestamp': 1124784000.0}),
     ('33', '1308', {'outcome': -1, 'timestamp': 911916000.000003}),
     ('33', '75', {'outcome': 1, 'timestamp': 917171999.999997}),
     ('33', '922', {'outcome': 1, 'timestamp': 917171999.999997}),
     ('33', '61', {'outcome': -1, 'timestamp': 917171999.999997}),
     ('33', '188', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('33', '1194', {'outcome': 0, 'timestamp': 925055999.999997}),
     ('33', '2061', {'outcome': 1, 'timestamp': 925055999.999997}),
     ('33', '2067', {'outcome': 0, 'timestamp': 935568000.000003}),
     ('33', '1608', {'outcome': 1, 'timestamp': 935568000.000003}),
     ('33', '2380', {'outcome': 1, 'timestamp': 1022292000.0}),
     ('33', '2380', {'outcome': 1, 'timestamp': 1093248000.0}),
     ('33', '1099', {'outcome': -1, 'timestamp': 1022292000.0}),
     ('33', '2003', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('33', '201', {'outcome': -1, 'timestamp': 1064340000.0}),
     ('33', '2138', {'outcome': 1, 'timestamp': 1064340000.0}),
     ('33', '1025', {'outcome': -1, 'timestamp': 1093248000.0}),
     ('33', '1796', {'outcome': 1, 'timestamp': 1093248000.0}),
     ('33', '3392', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('34', '47', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('34', '48', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('34', '890', {'outcome': 1, 'timestamp': 898776000.0}),
     ('34', '278', {'outcome': 0, 'timestamp': 898776000.0}),
     ('34', '1760', {'outcome': 1, 'timestamp': 932939999.999997}),
     ('34', '1177', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('34', '522', {'outcome': 1, 'timestamp': 948707999.999997}),
     ('34', '931', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('34', '423', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('34', '64', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('34', '64', {'outcome': 0, 'timestamp': 1132668000.0}),
     ('34', '256', {'outcome': 0, 'timestamp': 948707999.999997}),
     ('34', '266', {'outcome': 1, 'timestamp': 961848000.0}),
     ('34', '266', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('34', '265', {'outcome': 0, 'timestamp': 961848000.0}),
     ('34', '1327', {'outcome': 1, 'timestamp': 990756000.000003}),
     ('34', '965', {'outcome': 1, 'timestamp': 990756000.000003}),
     ('34', '25', {'outcome': 1, 'timestamp': 1038060000.0}),
     ('34', '25', {'outcome': 0, 'timestamp': 1101132000.0}),
     ('34', '234', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('34', '268', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('34', '14', {'outcome': 0, 'timestamp': 1074852000.0}),
     ('34', '215', {'outcome': 1, 'timestamp': 1122156000.0}),
     ('34', '1944', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('34', '267', {'outcome': 1, 'timestamp': 1135296000.0}),
     ('35', '12', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('35', '163', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('35', '2724', {'outcome': 1, 'timestamp': 948707999.999997}),
     ('35', '2170', {'outcome': 1, 'timestamp': 969732000.0}),
     ('35', '1809', {'outcome': 0, 'timestamp': 1003896000.0}),
     ('35', '2930', {'outcome': 1, 'timestamp': 1003896000.0}),
     ('35', '2317', {'outcome': 0, 'timestamp': 1003896000.0}),
     ('35', '1675', {'outcome': 1, 'timestamp': 1032804000.0}),
     ('35', '3799', {'outcome': 1, 'timestamp': 1032804000.0}),
     ('36', '207', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('36', '207', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('36', '441', {'outcome': 0, 'timestamp': 893519999.999997}),
     ('36', '188', {'outcome': -1, 'timestamp': 893519999.999997}),
     ('36', '775', {'outcome': -1, 'timestamp': 893519999.999997}),
     ('36', '775', {'outcome': 0, 'timestamp': 1082736000.0}),
     ('36', '776', {'outcome': 1, 'timestamp': 893519999.999997}),
     ('36', '24', {'outcome': 0, 'timestamp': 898776000.0}),
     ('36', '728', {'outcome': 0, 'timestamp': 898776000.0}),
     ('36', '573', {'outcome': -1, 'timestamp': 898776000.0}),
     ('36', '573', {'outcome': 0, 'timestamp': 988127999.999997}),
     ('36', '919', {'outcome': 1, 'timestamp': 898776000.0}),
     ('36', '579', {'outcome': 0, 'timestamp': 898776000.0}),
     ('36', '1194', {'outcome': 1, 'timestamp': 904032000.000003}),
     ('36', '1173', {'outcome': 0, 'timestamp': 904032000.000003}),
     ('36', '883', {'outcome': 0, 'timestamp': 904032000.000003}),
     ('36', '686', {'outcome': 0, 'timestamp': 906660000.0}),
     ('36', '1658', {'outcome': 1, 'timestamp': 906660000.0}),
     ('36', '1526', {'outcome': 1, 'timestamp': 906660000.0}),
     ('36', '163', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('36', '956', {'outcome': 0, 'timestamp': 917171999.999997}),
     ('36', '201', {'outcome': 0, 'timestamp': 940823999.999997}),
     ('36', '1327', {'outcome': 0, 'timestamp': 980243999.999997}),
     ('36', '2598', {'outcome': 0, 'timestamp': 980243999.999997}),
     ('36', '112', {'outcome': 0, 'timestamp': 988127999.999997}),
     ('36', '2631', {'outcome': 1, 'timestamp': 988127999.999997}),
     ('36', '465', {'outcome': 0, 'timestamp': 1001268000.0}),
     ('36', '237', {'outcome': 1, 'timestamp': 1001268000.0}),
     ('36', '1827', {'outcome': 0, 'timestamp': 1038060000.0}),
     ('36', '3456', {'outcome': 1, 'timestamp': 1038060000.0}),
     ('36', '104', {'outcome': 1, 'timestamp': 1038060000.0}),
     ('36', '635', {'outcome': 0, 'timestamp': 1043316000.0}),
     ('36', '902', {'outcome': 0, 'timestamp': 1064340000.0}),
     ('36', '721', {'outcome': 1, 'timestamp': 1080108000.0}),
     ('36', '519', {'outcome': -1, 'timestamp': 1080108000.0}),
     ('36', '3608', {'outcome': 1, 'timestamp': 1080108000.0}),
     ('36', '287', {'outcome': 1, 'timestamp': 1080108000.0}),
     ('36', '3926', {'outcome': 0, 'timestamp': 1082736000.0}),
     ('36', '3448', {'outcome': 1, 'timestamp': 1082736000.0}),
     ('36', '741', {'outcome': 0, 'timestamp': 1082736000.0}),
     ('36', '4455', {'outcome': 1, 'timestamp': 1095876000.0}),
     ('36', '2032', {'outcome': 0, 'timestamp': 1095876000.0}),
     ('36', '220', {'outcome': 1, 'timestamp': 1106388000.0}),
     ('36', '1719', {'outcome': 0, 'timestamp': 1106388000.0}),
     ('36', '259', {'outcome': 0, 'timestamp': 1106388000.0}),
     ('36', '476', {'outcome': -1, 'timestamp': 1111644000.0}),
     ('36', '4215', {'outcome': 1, 'timestamp': 1111644000.0}),
     ('36', '1117', {'outcome': 0, 'timestamp': 1111644000.0}),
     ('36', '77', {'outcome': 1, 'timestamp': 1114272000.0}),
     ('36', '89', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('36', '1191', {'outcome': 0, 'timestamp': 1127412000.0}),
     ('36', '251', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('36', '385', {'outcome': 1, 'timestamp': 1127412000.0}),
     ('36', '6053', {'outcome': 1, 'timestamp': 1132668000.0}),
     ('36', '266', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('36', '6131', {'outcome': 0, 'timestamp': 1137924000.0}),
     ('36', '41', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('36', '4018', {'outcome': 1, 'timestamp': 1143180000.0}),
     ('37', '35', {'outcome': 0, 'timestamp': 885635999.999997}),
     ('37', '1049', {'outcome': 1, 'timestamp': 917171999.999997}),
     ('37', '275', {'outcome': -1, 'timestamp': 948707999.999997}),
     ('37', '72', {'outcome': 0, 'timestamp': 1009152000.0}),
     ('37', '2894', {'outcome': -1, 'timestamp': 1032804000.0}),
     ('37', '1178', {'outcome': 0, 'timestamp': 1032804000.0}),
     ('37', '115', {'outcome': 0, 'timestamp': 1090620000.0}),
     ('37', '12', {'outcome': 1, 'timestamp': 1090620000.0}),
     ('37', '199', {'outcome': 0, 'timestamp': 1090620000.0}),
     ('37', '910', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('37', '257', {'outcome': 0, 'timestamp': 1122156000.0}),
     ('37', '79', {'outcome': -1, 'timestamp': 1122156000.0}),
     ('37', '64', {'outcome': -1, 'timestamp': 1122156000.0}),
     ('37', '4870', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('37', '5350', {'outcome': 1, 'timestamp': 1137924000.0}),
     ('37', '3147', {'outcome': 0, 'timestamp': 1140552000.0}),
     ('37', '458', {'outcome': -1, 'timestamp': 1140552000.0}),
     ('38', '39', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('38', '40', {'outcome': 1, 'timestamp': 885635999.999997}),
     ('38', '41', {'outcome': -1, 'timestamp': 885635999.999997}),
     ('38', '3101', {'outcome': -1, 'timestamp': 985500000.0}),
     ('38', '84', {'outcome': -1, 'timestamp': 985500000.0}),
     ('38', '3104', {'outcome': 1, 'timestamp': 985500000.0}),
     ...]



Looking at the degree of each node, we can see how many games each person played. A dictionary is returned where each key is the player, and each value is the number of games played.


```python
games_played = chess.degree()
games_played
```




    {'1': 48,
     '2': 112,
     '3': 85,
     '4': 12,
     '5': 18,
     '6': 95,
     '7': 9,
     '8': 20,
     '9': 142,
     '10': 4,
     '11': 2,
     '12': 70,
     '13': 148,
     '14': 153,
     '15': 23,
     '16': 3,
     '17': 115,
     '18': 45,
     '19': 27,
     '20': 12,
     '21': 65,
     '22': 6,
     '23': 41,
     '24': 72,
     '25': 72,
     '26': 2,
     '27': 3,
     '28': 41,
     '29': 8,
     '30': 115,
     '31': 23,
     '32': 146,
     '33': 67,
     '34': 52,
     '35': 22,
     '36': 118,
     '37': 30,
     '38': 39,
     '39': 79,
     '40': 26,
     '41': 71,
     '42': 200,
     '43': 10,
     '44': 77,
     '45': 74,
     '46': 41,
     '47': 92,
     '48': 45,
     '49': 58,
     '50': 9,
     '51': 90,
     '52': 4,
     '53': 167,
     '54': 80,
     '55': 35,
     '56': 87,
     '57': 48,
     '58': 29,
     '59': 111,
     '60': 43,
     '61': 107,
     '62': 74,
     '63': 7,
     '64': 171,
     '65': 43,
     '66': 203,
     '67': 39,
     '68': 23,
     '69': 4,
     '70': 15,
     '71': 55,
     '72': 10,
     '73': 19,
     '74': 144,
     '75': 22,
     '76': 20,
     '77': 71,
     '78': 18,
     '79': 122,
     '80': 4,
     '81': 29,
     '82': 29,
     '83': 58,
     '84': 32,
     '85': 35,
     '86': 13,
     '87': 52,
     '88': 61,
     '89': 123,
     '90': 58,
     '91': 100,
     '92': 257,
     '93': 24,
     '94': 10,
     '95': 14,
     '96': 41,
     '97': 25,
     '98': 276,
     '99': 28,
     '100': 20,
     '101': 49,
     '102': 31,
     '103': 8,
     '104': 95,
     '105': 6,
     '106': 18,
     '107': 14,
     '108': 75,
     '109': 100,
     '110': 12,
     '111': 42,
     '112': 208,
     '113': 142,
     '114': 96,
     '115': 87,
     '116': 14,
     '117': 26,
     '118': 51,
     '119': 10,
     '120': 52,
     '121': 36,
     '122': 106,
     '123': 28,
     '124': 138,
     '125': 38,
     '126': 85,
     '127': 166,
     '128': 118,
     '129': 82,
     '130': 211,
     '131': 66,
     '132': 207,
     '133': 73,
     '134': 3,
     '135': 13,
     '136': 9,
     '137': 24,
     '138': 108,
     '139': 44,
     '140': 38,
     '141': 10,
     '142': 65,
     '143': 62,
     '144': 21,
     '145': 64,
     '146': 18,
     '147': 66,
     '148': 111,
     '149': 78,
     '150': 20,
     '151': 51,
     '152': 69,
     '153': 91,
     '154': 2,
     '155': 35,
     '156': 63,
     '157': 110,
     '158': 93,
     '159': 53,
     '160': 103,
     '161': 73,
     '162': 65,
     '163': 29,
     '164': 150,
     '165': 140,
     '166': 87,
     '167': 16,
     '168': 8,
     '169': 30,
     '170': 15,
     '171': 114,
     '172': 126,
     '173': 40,
     '174': 36,
     '175': 22,
     '176': 6,
     '177': 39,
     '178': 88,
     '179': 48,
     '180': 25,
     '181': 8,
     '182': 39,
     '183': 43,
     '184': 37,
     '185': 82,
     '186': 52,
     '187': 99,
     '188': 180,
     '189': 131,
     '190': 36,
     '191': 15,
     '192': 11,
     '193': 108,
     '194': 21,
     '195': 49,
     '196': 24,
     '197': 88,
     '198': 7,
     '199': 89,
     '200': 127,
     '201': 99,
     '202': 127,
     '203': 20,
     '204': 37,
     '205': 11,
     '206': 4,
     '207': 185,
     '208': 8,
     '209': 2,
     '210': 4,
     '211': 194,
     '212': 28,
     '213': 21,
     '214': 30,
     '215': 20,
     '216': 22,
     '217': 5,
     '218': 9,
     '219': 81,
     '220': 54,
     '221': 14,
     '222': 37,
     '223': 93,
     '224': 100,
     '225': 7,
     '226': 71,
     '227': 112,
     '228': 146,
     '229': 60,
     '230': 4,
     '231': 41,
     '232': 64,
     '233': 22,
     '234': 80,
     '235': 62,
     '236': 36,
     '237': 197,
     '238': 125,
     '239': 88,
     '240': 17,
     '241': 31,
     '242': 138,
     '243': 3,
     '244': 13,
     '245': 8,
     '246': 104,
     '247': 29,
     '248': 83,
     '249': 8,
     '250': 40,
     '251': 191,
     '252': 76,
     '253': 15,
     '254': 24,
     '255': 58,
     '256': 148,
     '257': 149,
     '258': 46,
     '259': 107,
     '260': 78,
     '261': 86,
     '262': 21,
     '263': 34,
     '264': 88,
     '265': 21,
     '266': 39,
     '267': 21,
     '268': 45,
     '269': 85,
     '270': 63,
     '271': 1,
     '272': 140,
     '273': 1,
     '274': 2,
     '275': 192,
     '276': 21,
     '277': 54,
     '278': 154,
     '279': 14,
     '280': 129,
     '281': 64,
     '282': 4,
     '283': 34,
     '284': 11,
     '285': 146,
     '286': 15,
     '287': 107,
     '288': 10,
     '289': 41,
     '290': 1,
     '291': 24,
     '292': 1,
     '293': 2,
     '294': 13,
     '295': 46,
     '296': 21,
     '297': 7,
     '298': 28,
     '299': 122,
     '300': 112,
     '301': 47,
     '302': 61,
     '303': 39,
     '304': 38,
     '305': 17,
     '306': 37,
     '307': 16,
     '308': 28,
     '309': 19,
     '310': 107,
     '311': 4,
     '312': 50,
     '313': 50,
     '314': 79,
     '315': 21,
     '316': 36,
     '317': 16,
     '318': 47,
     '319': 9,
     '320': 44,
     '321': 28,
     '322': 54,
     '323': 20,
     '324': 27,
     '325': 26,
     '326': 14,
     '327': 74,
     '328': 78,
     '329': 15,
     '330': 272,
     '331': 7,
     '332': 37,
     '333': 78,
     '334': 179,
     '335': 67,
     '336': 35,
     '337': 124,
     '338': 31,
     '339': 113,
     '340': 59,
     '341': 18,
     '342': 9,
     '343': 4,
     '344': 26,
     '345': 41,
     '346': 7,
     '347': 26,
     '348': 4,
     '349': 26,
     '350': 9,
     '351': 8,
     '352': 149,
     '353': 26,
     '354': 37,
     '355': 55,
     '356': 42,
     '357': 119,
     '358': 45,
     '359': 13,
     '360': 66,
     '361': 9,
     '362': 64,
     '363': 16,
     '364': 51,
     '365': 20,
     '366': 57,
     '367': 9,
     '368': 2,
     '369': 23,
     '370': 26,
     '371': 258,
     '372': 72,
     '373': 67,
     '374': 6,
     '375': 39,
     '376': 75,
     '377': 2,
     '378': 7,
     '379': 3,
     '380': 4,
     '381': 9,
     '382': 9,
     '383': 152,
     '384': 70,
     '385': 113,
     '386': 2,
     '387': 187,
     '388': 23,
     '389': 5,
     '390': 5,
     '391': 9,
     '392': 59,
     '393': 130,
     '394': 42,
     '395': 35,
     '396': 86,
     '397': 17,
     '398': 23,
     '399': 52,
     '400': 180,
     '401': 17,
     '402': 18,
     '403': 38,
     '404': 11,
     '405': 13,
     '406': 3,
     '407': 29,
     '408': 42,
     '409': 45,
     '410': 9,
     '411': 9,
     '412': 141,
     '413': 6,
     '414': 92,
     '415': 34,
     '416': 95,
     '417': 4,
     '418': 78,
     '419': 26,
     '420': 3,
     '421': 150,
     '422': 3,
     '423': 92,
     '424': 10,
     '425': 21,
     '426': 17,
     '427': 4,
     '428': 4,
     '429': 9,
     '430': 10,
     '431': 8,
     '432': 46,
     '433': 23,
     '434': 3,
     '435': 69,
     '436': 117,
     '437': 12,
     '438': 4,
     '439': 88,
     '440': 171,
     '441': 46,
     '442': 14,
     '443': 83,
     '444': 78,
     '445': 62,
     '446': 112,
     '447': 201,
     '448': 116,
     '449': 126,
     '450': 30,
     '451': 125,
     '452': 96,
     '453': 94,
     '454': 22,
     '455': 252,
     '456': 221,
     '457': 45,
     '458': 54,
     '459': 62,
     '460': 76,
     '461': 280,
     '462': 196,
     '463': 45,
     '464': 24,
     '465': 161,
     '466': 96,
     '467': 270,
     '468': 130,
     '469': 6,
     '470': 15,
     '471': 61,
     '472': 22,
     '473': 59,
     '474': 4,
     '475': 26,
     '476': 113,
     '477': 87,
     '478': 21,
     '479': 26,
     '480': 58,
     '481': 30,
     '482': 121,
     '483': 92,
     '484': 23,
     '485': 66,
     '486': 42,
     '487': 132,
     '488': 46,
     '489': 3,
     '490': 15,
     '491': 42,
     '492': 48,
     '493': 16,
     '494': 23,
     '495': 38,
     '496': 136,
     '497': 112,
     '498': 17,
     '499': 61,
     '500': 148,
     '501': 8,
     '502': 64,
     '503': 96,
     '504': 25,
     '505': 18,
     '506': 2,
     '507': 68,
     '508': 97,
     '509': 17,
     '510': 131,
     '511': 28,
     '512': 4,
     '513': 31,
     '514': 71,
     '515': 33,
     '516': 131,
     '517': 14,
     '518': 3,
     '519': 178,
     '520': 14,
     '521': 53,
     '522': 81,
     '523': 50,
     '524': 15,
     '525': 17,
     '526': 22,
     '527': 7,
     '528': 12,
     '529': 5,
     '530': 56,
     '531': 3,
     '532': 4,
     '533': 3,
     '534': 20,
     '535': 12,
     '536': 89,
     '537': 162,
     '538': 112,
     '539': 53,
     '540': 25,
     '541': 19,
     '542': 53,
     '543': 142,
     '544': 6,
     '545': 158,
     '546': 51,
     '547': 26,
     '548': 18,
     '549': 51,
     '550': 9,
     '551': 15,
     '552': 48,
     '553': 4,
     '554': 5,
     '555': 8,
     '556': 21,
     '557': 125,
     '558': 17,
     '559': 1,
     '560': 125,
     '561': 41,
     '562': 36,
     '563': 178,
     '564': 8,
     '565': 10,
     '566': 78,
     '567': 17,
     '568': 29,
     '569': 52,
     '570': 10,
     '571': 4,
     '572': 28,
     '573': 216,
     '574': 112,
     '575': 4,
     '576': 26,
     '577': 92,
     '578': 24,
     '579': 155,
     '580': 207,
     '581': 21,
     '582': 19,
     '583': 78,
     '584': 5,
     '585': 9,
     '586': 171,
     '587': 9,
     '588': 37,
     '589': 49,
     '590': 21,
     '591': 1,
     '592': 2,
     '593': 45,
     '594': 4,
     '595': 136,
     '596': 26,
     '597': 6,
     '598': 2,
     '599': 16,
     '600': 38,
     '601': 65,
     '602': 41,
     '603': 20,
     '604': 23,
     '605': 137,
     '606': 8,
     '607': 15,
     '608': 17,
     '609': 83,
     '610': 193,
     '611': 3,
     '612': 31,
     '613': 5,
     '614': 18,
     '615': 17,
     '616': 5,
     '617': 1,
     '618': 2,
     '619': 1,
     '620': 13,
     '621': 164,
     '622': 82,
     '623': 211,
     '624': 7,
     '625': 6,
     '626': 10,
     '627': 38,
     '628': 8,
     '629': 8,
     '630': 3,
     '631': 9,
     '632': 15,
     '633': 46,
     '634': 33,
     '635': 155,
     '636': 27,
     '637': 192,
     '638': 52,
     '639': 43,
     '640': 60,
     '641': 74,
     '642': 47,
     '643': 130,
     '644': 57,
     '645': 6,
     '646': 71,
     '647': 66,
     '648': 52,
     '649': 105,
     '650': 3,
     '651': 100,
     '652': 90,
     '653': 158,
     '654': 131,
     '655': 61,
     '656': 132,
     '657': 48,
     '658': 33,
     '659': 255,
     '660': 22,
     '661': 83,
     '662': 54,
     '663': 54,
     '664': 99,
     '665': 11,
     '666': 7,
     '667': 9,
     '668': 43,
     '669': 20,
     '670': 93,
     '671': 42,
     '672': 27,
     '673': 232,
     '674': 86,
     '675': 10,
     '676': 54,
     '677': 15,
     '678': 3,
     '679': 1,
     '680': 42,
     '681': 57,
     '682': 15,
     '683': 13,
     '684': 126,
     '685': 147,
     '686': 36,
     '687': 85,
     '688': 13,
     '689': 39,
     '690': 59,
     '691': 59,
     '692': 123,
     '693': 15,
     '694': 14,
     '695': 11,
     '696': 34,
     '697': 24,
     '698': 3,
     '699': 6,
     '700': 41,
     '701': 42,
     '702': 8,
     '703': 27,
     '704': 75,
     '705': 18,
     '706': 60,
     '707': 62,
     '708': 13,
     '709': 24,
     '710': 84,
     '711': 9,
     '712': 110,
     '713': 61,
     '714': 59,
     '715': 56,
     '716': 29,
     '717': 11,
     '718': 4,
     '719': 28,
     '720': 191,
     '721': 120,
     '722': 39,
     '723': 25,
     '724': 25,
     '725': 134,
     '726': 50,
     '727': 6,
     '728': 118,
     '729': 1,
     '730': 33,
     '731': 195,
     '732': 1,
     '733': 27,
     '734': 53,
     '735': 2,
     '736': 188,
     '737': 37,
     '738': 18,
     '739': 51,
     '740': 27,
     '741': 166,
     '742': 155,
     '743': 28,
     '744': 32,
     '745': 13,
     '746': 39,
     '747': 52,
     '748': 114,
     '749': 53,
     '750': 7,
     '751': 2,
     '752': 61,
     '753': 87,
     '754': 34,
     '755': 21,
     '756': 27,
     '757': 18,
     '758': 10,
     '759': 85,
     '760': 5,
     '761': 30,
     '762': 17,
     '763': 29,
     '764': 18,
     '765': 34,
     '766': 8,
     '767': 24,
     '768': 4,
     '769': 9,
     '770': 24,
     '771': 35,
     '772': 40,
     '773': 49,
     '774': 24,
     '775': 48,
     '776': 9,
     '777': 67,
     '778': 130,
     '779': 4,
     '780': 11,
     '781': 4,
     '782': 8,
     '783': 50,
     '784': 28,
     '785': 19,
     '786': 8,
     '787': 78,
     '788': 45,
     '789': 155,
     '790': 47,
     '791': 17,
     '792': 46,
     '793': 46,
     '794': 7,
     '795': 17,
     '796': 14,
     '797': 30,
     '798': 126,
     '799': 24,
     '800': 19,
     '801': 185,
     '802': 10,
     '803': 164,
     '804': 1,
     '805': 7,
     '806': 63,
     '807': 87,
     '808': 68,
     '809': 16,
     '810': 20,
     '811': 6,
     '812': 1,
     '813': 13,
     '814': 3,
     '815': 57,
     '816': 54,
     '817': 49,
     '818': 60,
     '819': 57,
     '820': 13,
     '821': 4,
     '822': 57,
     '823': 3,
     '824': 73,
     '825': 80,
     '826': 63,
     '827': 84,
     '828': 38,
     '829': 25,
     '830': 46,
     '831': 109,
     '832': 30,
     '833': 184,
     '834': 36,
     '835': 5,
     '836': 6,
     '837': 14,
     '838': 63,
     '839': 12,
     '840': 43,
     '841': 17,
     '842': 61,
     '843': 8,
     '844': 31,
     '845': 3,
     '846': 6,
     '847': 21,
     '848': 78,
     '849': 129,
     '850': 2,
     '851': 33,
     '852': 42,
     '853': 22,
     '854': 7,
     '855': 18,
     '856': 6,
     '857': 10,
     '858': 4,
     '859': 37,
     '860': 33,
     '861': 28,
     '862': 139,
     '863': 39,
     '864': 35,
     '865': 13,
     '866': 5,
     '867': 14,
     '868': 36,
     '869': 6,
     '870': 21,
     '871': 28,
     '872': 70,
     '873': 20,
     '874': 42,
     '875': 9,
     '876': 28,
     '877': 8,
     '878': 11,
     '879': 46,
     '880': 117,
     '881': 6,
     '882': 78,
     '883': 78,
     '884': 9,
     '885': 102,
     '886': 39,
     '887': 6,
     '888': 12,
     '889': 11,
     '890': 9,
     '891': 56,
     '892': 17,
     '893': 37,
     '894': 2,
     '895': 36,
     '896': 14,
     '897': 66,
     '898': 36,
     '899': 12,
     '900': 43,
     '901': 29,
     '902': 45,
     '903': 2,
     '904': 20,
     '905': 5,
     '906': 11,
     '907': 19,
     '908': 25,
     '909': 5,
     '910': 92,
     '911': 3,
     '912': 3,
     '913': 195,
     '914': 82,
     '915': 25,
     '916': 5,
     '917': 14,
     '918': 17,
     '919': 6,
     '920': 29,
     '921': 47,
     '922': 15,
     '923': 5,
     '924': 15,
     '925': 44,
     '926': 26,
     '927': 26,
     '928': 31,
     '929': 34,
     '930': 24,
     '931': 70,
     '932': 13,
     '933': 64,
     '934': 6,
     '935': 19,
     '936': 9,
     '937': 4,
     '938': 24,
     '939': 13,
     '940': 7,
     '941': 34,
     '942': 24,
     '943': 14,
     '944': 5,
     '945': 22,
     '946': 48,
     '947': 13,
     '948': 13,
     '949': 9,
     '950': 106,
     '951': 4,
     '952': 65,
     '953': 32,
     '954': 12,
     '955': 53,
     '956': 70,
     '957': 14,
     '958': 44,
     '959': 101,
     '960': 64,
     '961': 60,
     '962': 5,
     '963': 14,
     '964': 7,
     '965': 48,
     '966': 4,
     '967': 3,
     '968': 100,
     '969': 1,
     '970': 1,
     '971': 43,
     '972': 9,
     '973': 98,
     '974': 26,
     '975': 33,
     '976': 40,
     '977': 9,
     '978': 19,
     '979': 2,
     '980': 3,
     '981': 61,
     '982': 10,
     '983': 154,
     '984': 7,
     '985': 1,
     '986': 3,
     '987': 74,
     '988': 12,
     '989': 66,
     '990': 4,
     '991': 27,
     '992': 14,
     '993': 30,
     '994': 8,
     '995': 18,
     '996': 8,
     '997': 45,
     '998': 10,
     '999': 22,
     '1000': 7,
     ...}



Using list comprehension, we can find which player played the most games.


```python
max_value = max(games_played.values())
max_key, = [i for i in games_played.keys() if games_played[i] == max_value]

print('player {}\n{} games'.format(max_key, max_value))
```

    player 461
    280 games


Let's use pandas to find out which players won the most games. First let's convert our graph to a DataFrame.


```python
df = pd.DataFrame(chess.edges(data=True), columns=['white', 'black', 'outcome'])
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>white</th>
      <th>black</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>{'outcome': 0, 'timestamp': 885635999.999997}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>{'outcome': 0, 'timestamp': 885635999.999997}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>4</td>
      <td>{'outcome': 0, 'timestamp': 885635999.999997}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>{'outcome': 1, 'timestamp': 885635999.999997}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>6</td>
      <td>{'outcome': 0, 'timestamp': 885635999.999997}</td>
    </tr>
  </tbody>
</table>
</div>



Next we can use a lambda to pull out the outcome from the attributes dictionary.


```python
df['outcome'] = df['outcome'].map(lambda x: x['outcome'])
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>white</th>
      <th>black</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



To count the number of times a player won as white, we find the rows where the outcome was '1', group by the white player, and sum.

To count the number of times a player won as back, we find the rows where the outcome was '-1', group by the black player, sum, and multiply by -1.

The we can add these together with a fill value of 0 for those players that only played as either black or white.


```python
won_as_white = df[df['outcome']==1].groupby('white').sum()
won_as_black = -df[df['outcome']==-1].groupby('black').sum()
win_count = won_as_white.add(won_as_black, fill_value=0)
win_count.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>7.0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



Using `nlargest` we find that player 330 won the most games at 109.


```python
win_count.nlargest(5, 'outcome')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>330</th>
      <td>109.0</td>
    </tr>
    <tr>
      <th>467</th>
      <td>103.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>94.0</td>
    </tr>
    <tr>
      <th>456</th>
      <td>88.0</td>
    </tr>
    <tr>
      <th>461</th>
      <td>88.0</td>
    </tr>
  </tbody>
</table>
</div>


