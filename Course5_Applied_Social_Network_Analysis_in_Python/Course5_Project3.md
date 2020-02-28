# Project 3

In this project you will explore measures of centrality on two networks, a friendship network in Part 1, and a blog network in Part 2.

## Part 1

Answer parts 1-4 using the network `G1`, a network of friendships at a university department. Each node corresponds to a person, and an edge indicates friendship. 

*The network has been loaded as networkx graph object `G1`.*


```python
import networkx as nx

G1 = nx.read_gml('friendships.gml')
```

### Part 1

Find the degree centrality, closeness centrality, and normalized betweeness centrality (excluding endpoints) of node 100.

*Returns a tuple of floats `(degree_centrality, closeness_centrality, betweenness_centrality)`.*


```python
def analyze_graph():
    
    dc, cc, bc = nx.degree_centrality(G1), nx.closeness_centrality(G1), nx.betweenness_centrality(G1, normalized=True, endpoints=False)
    
    return dc[100], cc[100], bc[100]

```

<br>
#### For parts 2, 3, and 4, assume that you do not know anything about the structure of the network, except for the all the centrality values of the nodes. That is, use one of the covered centrality measures to rank the nodes and find the most appropriate candidate.
<br>

### Part 2

Suppose you are employed by an online shopping website and are tasked with selecting one user in network G1 to send an online shopping voucher to. We expect that the user who receives the voucher will send it to their friends in the network.  You want the voucher to reach as many nodes as possible. The voucher can be forwarded to multiple users at the same time, but the travel distance of the voucher is limited to one step, which means if the voucher travels more than one step in this network, it is no longer valid. Apply your knowledge in network centrality to select the best candidate for the voucher. 

*Returns an integer, the name of the node.*


```python
def get_most_central_user():
        
    user_centrality = nx.degree_centrality(G1)
    
    most_central_user = max(user_centrality, key=user_centrality.get)
    
    return most_central_user

```

### Part 3

Now the limit of the voucher’s travel distance has been removed. Because the network is connected, regardless of who you pick, every node in the network will eventually receive the voucher. However, we now want to ensure that the voucher reaches the nodes in the lowest average number of hops.

How would you change your selection strategy? Write a function to tell us who is the best candidate in the network under this condition.

*Returns an integer, the name of the node.*


```python
def get_most_between_user():
        
    user_centrality = nx.closeness_centrality(G1)
    
    most_between_user = sorted(user_centrality, key=user_centrality.get, reverse=True)[0] #max(user_centrality, key=user_centrality.get)
    
    return most_between_user

```

### Part 4

Assume the restriction on the voucher’s travel distance is still removed, but now a competitor has developed a strategy to remove a person from the network in order to disrupt the distribution of your company’s voucher. Your competitor is specifically targeting people who are often bridges of information flow between other pairs of people. Identify the single riskiest person to be removed under your competitor’s strategy?

*Returns an integer, the name of the node.*


```python
def get_riskiest_users():
        
    riskiest_users = nx.minimum_node_cut(G1)
    
    return next(iter(riskiest_users))

```

## Part 2

`G2` is a directed network of political blogs, where nodes correspond to a blog and edges correspond to links between blogs. Use PageRank and HITS to answer parts 5-9.


```python
G2 = nx.read_gml('blogs.gml')
```

## Part 5

Apply the Scaled Page Rank Algorithm to this network. Find the Page Rank of node 'realclearpolitics.com' with damping value 0.85.

*Returns a float.*


```python
def get_pagerank():
        
    scaled_pr = nx.pagerank(G2, alpha=0.85)
    
    return scaled_pr['realclearpolitics.com']

```

### Part 6

Apply the Scaled Page Rank Algorithm to this network with damping value 0.85. Find the 5 nodes with highest Page Rank. 

*Returns a list of the top 5 blogs in desending order of Page Rank.*


```python
def nodes_with_highest_pagerank():
        
    scaled_pr = nx.pagerank(G2, alpha=0.85)
    
    sorted_by_pr = sorted(scaled_pr, key=scaled_pr.get, reverse=True)
    
    return sorted_by_pr[:5]

```

### Part 7

Apply the HITS Algorithm to the network to find the hub and authority scores of node 'realclearpolitics.com'. 

*Your result should return a tuple of floats `(hub_score, authority_score)`.*


```python
def get_hits_scores():
        
    hits_score = nx.hits(G2)
    
    return hits_score[0]['realclearpolitics.com'], hits_score[1]['realclearpolitics.com']

```

### Part 8 

Apply the HITS Algorithm to this network to find the 5 nodes with highest hub scores.

*Returns a list of the top 5 blogs in desending order of hub scores.*


```python
def get_hits_scores_2():
        
    hub_scores = nx.hits(G2)[0]
    
    hub_scores_sorted = sorted(hub_scores, key=hub_scores.get, reverse=True)
    
    return hub_scores_sorted[:5]

```

### Part 9 

Apply the HITS Algorithm to this network to find the 5 nodes with highest authority scores.

*Returns a list of the top 5 blogs in desending order of authority scores.*


```python
def get_hits_scores_3():
        
    auth_scores = nx.hits(G2)[1]
    
    auth_scores_sorted = sorted(auth_scores, key=auth_scores.get, reverse=True)
    
    return auth_scores_sorted[:5]

```
