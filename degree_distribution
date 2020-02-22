import collections
import matplotlib.pyplot as plt
import networkx as nx


def plot_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.2)
    plt.plot(deg, cnt)

    plt.title("Degree histogram of labeled {}".format(args.dataset))
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 120, 10))
    # ax.set_xticklabels(deg)

    # draw graph in inset
    # plt.axes([0.4, 0.4, 0.5, 0.5])
    # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # pos = nx.spring_layout(G)
    # plt.axis('off')
    # nx.draw_networkx_nodes(G, pos, node_size=10)
    # nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.savefig('labeled_{}_degree.eps'.format(args.dataset))
    plt.show()


if __name__ == '__main__':
    import argparse
    from dgl.data import load_data
    from dgl import DGLGraph
    import numpy as np
    parser = argparse.ArgumentParser(description='SGC')
    parser.add_argument("--dataset", type=str, default='pubmed')
    args = parser.parse_args()
    data = load_data(args)
    g = DGLGraph(data.graph)
    G = g.to_networkx().to_undirected()
    labeled_degree=False
    if labeled_degree:
        print('labeled degree distribution')
        train_mask = data.train_mask
        labeled_nodes_ids = np.nonzero(train_mask)[0]
        G = G.subgraph(labeled_nodes_ids)
    print(G.number_of_edges())
    print(G.number_of_nodes())
    # G = nx.gnp_random_graph(100, 0.02)
    plot_degree_distribution(G)
