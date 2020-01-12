# main1: 在记忆中训练好策略（策略收敛），在实际中执行

#这个可以用来解决起点不同的问题

import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_node('a')
G.add_node('rr2')
G.add_node('rr2')
G.add_nodes_from([3, 4, 5, 6])
#G.add_cycle([1, 2, 3, 4])
G.add_edge(3, 'a', weight=0.2)
G.add_edges_from([(3, 5), (3, 'rr2'), (6, 7)])
print("输出全部节点：{}".format(G.nodes()))
print("输出全部边：{}".format(G.edges()))
print("输出全部边的数量：{}".format(G.number_of_edges()))
nx.draw(G)
plt.show()

G1 = nx.path_graph(8)
nx.draw(G1)
plt.show()
print(G['rr2'])