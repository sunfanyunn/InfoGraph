import networkx as nx
from karateclub import Graph2Vec

# Generate sample graphs (replace this with your actual graph dataset)
graphs = [
    nx.generators.erdos_renyi_graph(10, 0.5),
    nx.generators.watts_strogatz_graph(10, 5, 0.3),
    nx.generators.barabasi_albert_graph(10, 2),
]

# Initialize Graph2Vec with default parameters
graph2vec = Graph2Vec(dimensions=128, wl_iterations=2, min_count=5, epochs=10)

# Fit the model to the dataset
graph2vec.fit(graphs)

# Get the graph embeddings
embeddings = graph2vec.get_embedding()

print(embeddings)
with open('graph2vec_trial.log', 'a+') as f:
    f.write('embedding: \n{}'.format(embeddings))
    # f.write('embedding shape: {len(embeddings)}x{}\n{}'.format(len(emb), len(emb[0]), emb))

