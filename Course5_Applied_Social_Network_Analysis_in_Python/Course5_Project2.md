# Project 2 - Network Connectivity

In this project you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company. 
Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient.


```python
import networkx as nx

# This line must be commented out when submitting to the autograder
# !head email_network.txt
```

### Part 1

Using networkx, load up the directed multigraph from `email_network.txt`. Make sure the node names are strings.

*Returns a directed multigraph networkx graph.*


```python
def load_multigraph():
    
    G = nx.read_edgelist('email_network.txt', data=(('time', int),), create_using=nx.MultiDiGraph())
    
    return G

```

### Part 2

How many employees and emails are represented in the graph from Part 1?

*Returns a tuple (#employees, #emails).*


```python
def count_employees_and_emails():
        
    G = load_multigraph()
    
    N = nx.number_of_nodes(G)
    E = nx.number_of_edges(G)
    
    return N, E

```

### Part 3

* Part 1. Assume that information in this company can only be exchanged through email.

    When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the receiver, but not vice versa. 

    Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?


* Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways. 

    Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?


*Returns a tuple of bools (part1, part2).*


```python
def connectivity_test():

    G = load_multigraph()
    
    return nx.is_strongly_connected(G), nx.is_weakly_connected(G)

```

### Part 4

How many nodes are in the largest (in terms of nodes) weakly connected component?

*Returns an int.*


```python
def component_node_count():
        
    G = load_multigraph()
    
    wccs = nx.weakly_connected_components(G)
    
    largest_wcc = max(wccs, key=len)
    
    return len(largest_wcc)

```

### Part 5

How many nodes are in the largest (in terms of nodes) strongly connected component?

*Returns an int*


```python
def component_node_count_2():
        
    G = load_multigraph()
    
    scc = nx.strongly_connected_components(G)
    
    largest_scc = max(scc, key=len)
    
    return len(largest_scc)

```

### Part 6

Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. 
Call this graph G_sc.

*Returns a networkx MultiDiGraph named G_sc.*


```python
def get_subgraph():
        
    G = load_multigraph()
    
    sccs = nx.strongly_connected_component_subgraphs(G)
    
    G_sc = max(sccs, key=len)
    
    return G_sc

```

### Part 7

What is the average distance between nodes in G_sc?

*Returns a float.*


```python
def subgraph_avg_dist():
        
    G_sc = get_subgraph()
    
    avg_dist = nx.average_shortest_path_length(G_sc)
    
    return avg_dist

```

### Part 8

What is the largest possible distance between two employees in G_sc?

*Returns an int.*


```python
def subgraph_diameter():
        
    G_sc = get_subgraph()
    
    G_sc_diameter = nx.diameter(G_sc)
    
    return G_sc_diameter

```

### Part 9

What is the set of nodes in G_sc with eccentricity equal to the diameter?

*Returns a set of the node(s).*


```python
def subgraph_periphery():
       
    G_sc = get_subgraph()
    
    G_sc_periphery = nx.periphery(G_sc)
    
    return set(G_sc_periphery)

```

### Part 10

What is the set of node(s) in G_sc with eccentricity equal to the radius?

*Returns a set of the node(s).*


```python
def subgraph_center():
        
    G_sc = get_subgraph()
    
    G_sc_center = nx.center(G_sc)
    
    return set(G_sc_center)

```

### Part 11

Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?

How many nodes are connected to this node?


*Returns a tuple (name of node, number of satisfied connected nodes).*


```python
def most_connected_node():
        
    G_sc = get_subgraph()
    
    # periphery of G_sc
    G_sc_periphery = nx.periphery(G_sc)
    
    # get diameter
    G_diameter = nx.diameter(G_sc)
    
    # edges consisting of peripheral nodes
    conns = set([(node1, node2) for node1, node2 in nx.edges(G_sc) if node1 in G_sc_periphery or node2 in G_sc_periphery])
    
    d = dict.fromkeys(G_sc_periphery, 0)
    for n1, n2 in conns:
        if n1 in d.keys():
            d[n1] += 1
        elif n2 in d.keys():
            d[n2] += 1
    
    most_conn_node = max(d, key=d.get)
    
    return most_conn_node, d[most_conn_node]

```

### Part 12

Suppose you want to prevent communication from flowing to the node that you found in the previous part from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous part or the center nodes)? 

*Returns an integer.*


```python
def node_removal_count():
        
    G_sc = get_subgraph()
    
    G_sc_center = nx.center(G_sc)
    
    # primary node (the "to" node)
    n, length = answer_eleven()
    
    # min node cuts
    d = [len(nx.minimum_node_cut(G_sc, m, n)) for m in G_sc_center]
    
    return min(d)

```

### Part 13

Construct an undirected graph G_un using G_sc (you can ignore the attributes).

*Returns a networkx Graph.*


```python
def make_undir_graph():
        
    G_sc = get_subgraph()
    
    G_un = nx.Graph(G_sc)
    
    return G_un

```

### Part 14

What is the transitivity and average clustering coefficient of graph G_un?

*Returns a tuple (transitivity, avg clustering).*


```python
def get_graph_details():
        
    G_un = make_undir_graph()
    
    G_un_trans = nx.transitivity(G_un)
    G_un_avgclust = nx.average_clustering(G_un)
    
    return G_un_trans, G_un_avgclust

```
