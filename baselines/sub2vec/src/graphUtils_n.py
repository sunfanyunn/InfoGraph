import networkx as nx
import random
import matplotlib.pyplot as plt


def getGraph(filename):
    G=nx.Graph()
    
    f=open(filename,'r')
    lines=f.readlines()
    for line in lines:
        if(line[0]=='#'):
            continue
        else:
            temp=line.split()
            index1=int(temp[0])
            index2=int(temp[1])
            G.add_edge(index1,index2)         
    f.close()
    return G


def randomWalk(G, walkSize):
    walkList= []
    curNode = random.choice(G.nodes())

    while(len(walkList) < walkSize):
        walkList.append(curNode)
        curNode = random.choice(G.neighbors(curNode))  
    return walkList
    
def getStats(G):
    stats ={}
    stats['num_nodes'] = nx.number_of_nodes(G)
    stats['num_edges'] = nx.number_of_edges(G)
    stats['is_Connected'] = nx.is_connected(G)


def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    plt.savefig("graph.pdf")
    plt.show()
