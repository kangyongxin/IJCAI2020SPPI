import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_node('a')
G.add_node('rr2')
G.add_node('rr2')
G.add_nodes_from([3, 4, 5, 6])
#G.add_cycle([1, 2, 3, 4])
G.add_edge(3, 'a', weight=0.2,visits=3)
G.add_edges_from([(3, 5), (3, 'rr2'), (6, 7)])
print("输出全部节点：{}".format(G.nodes()))
print("输出全部边：{}".format(G.edges()))
print("输出全部边的数量：{}".format(G.number_of_edges()))
nx.draw(G)
plt.show()
edge=[3,'a']
if edge in G.edges():
    print("well done")
    print(G.edges[3])
    for edgei in G.edges(3):
        print(edgei)
        print(G.edges[edgei]['weight'])
    G.add_edge(3, 'a', visits=3+0.2)
    print(G.edges[3,'a']['visits'])

G1 = nx.path_graph(8)
nx.draw(G1)
plt.show()
print(G['rr2'])