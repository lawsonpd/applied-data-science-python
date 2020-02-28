# Project 1 - Creating and Manipulating Graphs

Eight employees at a small company were asked to choose 3 movies that they would most enjoy watching for the upcoming company movie night. These choices are stored in the file `Employee_Movie_Choices.txt`.

A second file, `Employee_Relationships.txt`, has data on the relationships between different coworkers. 

The relationship score has value of `-100` (Enemies) to `+100` (Best Friends). A value of zero means the two employees haven't interacted or are indifferent.

Both files are tab delimited.


```python
import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite


# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])
```

### Question 1

Using NetworkX, load in the bipartite graph from `Employee_Movie_Choices.txt` and return that graph.

*Returns a networkx graph with 19 nodes and 24 edges*


```python
def load_graph():
        
    G = nx.read_edgelist('Employee_Movie_Choices.txt', nodetype=str, delimiter='\t')
    
    return G

```

### Question 2

Using the graph from the previous question, add nodes attributes named `'type'` where movies have the value `'movie'` and employees have the value `'employee'` and return that graph.

*Returns a networkx graph with node attributes `{'type': 'movie'}` or `{'type': 'employee'}`*


```python
def add_node_attributes():
    
    G = load_graph()
    
    for node in G:
        if node in employees:
            G.node[node]['type'] = 'employee'
        elif node in movies:
            G.node[node]['type'] = 'movie'
    
    return G

```

### Question 3

Find a weighted projection of the graph from `add_node_attributes` which tells us how many movies different pairs of employees have in common.

*Returns a weighted projected graph.*


```python
def find_weighted_projection():
        
    G = add_node_attributes()
    
    b_proj = bipartite.weighted_projected_graph(G, [node for node in G.nodes() if nx.get_node_attributes(G, 'type')[node] == 'employee'])
    
    return b_proj

```

#### Question 4

Suppose you'd like to find out if people that have a high relationship score also like the same types of movies.

Find the Pearson correlation ( using `DataFrame.corr()` ) between employee relationship scores and the number of movies they have in common. If two employees have no movies in common it should be treated as a 0, not a missing value, and should be included in the correlation calculation.

*Returns a float.*


```python
def find_correlation():
    # weighted graph from Q3
    movies_graph = find_weighted_projection()
    
    rel_df = pd.read_csv('Employee_Relationships.txt', names=['Employee 1', 'Employee 2', 'Relationship'], delim_whitespace=True, header=None)
#     movies_df = pd.read_csv('Employee_Movie_Choices.txt', names=['Employee', 'Movie'], delimiter='\t', header=0)
#     movies_df = pd.DataFrame(movies_graph.edges(data=True), columns=['Employee 1', 'Employee 2', 'Shared'])
    
    # merge DFs
    df = pd.merge(rel_df, movies_df, how='left')
    
    # pull out number of shared movies from weight dict
    df['Shared'] = df['Shared'].map(lambda x: x['weight'] if type(x) == dict else x)
    
    # fill 0 for NaN in Shared
    df.fillna(value={'Shared': 0}, inplace=True)
    
    corr = df.corr(method='pearson')
    
    return float(corr.iloc[0][1])

```
