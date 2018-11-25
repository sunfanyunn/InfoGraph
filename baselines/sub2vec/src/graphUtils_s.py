import networkx as nx
import random
import matplotlib.pyplot as plt


def writeGraph(filename, G):
    
    file = open(filename, 'w')
    for edge in G.edges():
        node1 = str(G.node[edge[0]]['label'])
        node2 = str(G.node[edge[1]]['label'])
        file.write(node1+'\t'+node2+'\n')
    file.close()


def getGraph(filename):
    G=nx.Graph()
    mode = 0
    f=open(filename,'r')
    lines=f.readlines()
    labels = {}
    for line in lines:
        temp=line.split()
        index1=int(temp[0])
        index2=int(temp[1])
        G.add_edge(index1,index2)         
    f.close()
    nx.set_node_attributes(G, 'label', labels)
    return G


def randomWalk(G, walkSize):
    walkList= []
    curNode = random.choice(G.nodes())

    while(len(walkList) < walkSize):
        walkList.append(G.node[curNode]['label'])
        curNode = random.choice(G.neighbors(curNode))  
    return walkList
    
def getStats(G):
    stats ={}
    stats['num_nodes'] = nx.number_of_nodes(G)
    stats['num_edges'] = nx.number_of_edges(G)
    stats['is_Connected'] = nx.is_connected(G)


def drawGraph(G):
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    plt.savefig("graph.pdf")
    plt.show()
