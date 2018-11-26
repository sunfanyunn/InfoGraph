import gensim.models.doc2vec as doc
import os
import graphUtils_n
from tqdm import tqdm


def arr2str(arr):
    result = ""
    for i in arr:
        result += " "+str(i)
    return result
    

def generateWalkFile(dirName, walkLength):
    walkFile = open(dirName+'.walk', 'w')
    indexToName = {}
    
    for  root, dirs, files in os.walk(dirName):
        index = 0
        for name in tqdm(files):
            # print(name)
            subgraph = graphUtils_n.getGraph(os.path.join(root, name))
            walk = graphUtils_n.randomWalk(subgraph, walkLength)
            walkFile.write(arr2str(walk) +"\n")
            indexToName[index] = name
            index += 1
    walkFile.close()
    
    return indexToName
    
def saveVectors(vectors, outputfile, IdToName):
    print(len(vectors), outputfile, IdToName)
    output = open(outputfile, 'w')
    
    output.write(str(len(vectors)) +"\n")
    for i in range(len(vectors)):
        output.write(str(IdToName[i]))
        for j in vectors[i]:
            output.write('\t'+ str(j))
        output.write('\n')
    output.close()
    
def neighborhood_embedding(args):
    inputDir = args.input
    # outputFile = args.output
    iterations = args.iter
    dimensions = args.d
    window = args.windowSize
    dm = 1 if args.model == 'dm' else 0
    indexToName = generateWalkFile(inputDir, args.walkLength)
    # print(indexToName)
    sentences = doc.TaggedLineDocument(inputDir+'.walk')

    for epochs in range(10, 210, 10):
        print('epochs', epochs)
        model = doc.Doc2Vec(sentences, vector_size = dimensions, epochs = epochs, dm = dm, window = window )
        vectors = model.docvecs
        embeddings = [[] for _ in range(len(vectors))]
        for i in range(len(vectors)):
            embeddings[int(indexToName[i])] = vectors[i]
        
        from preprocess import evaluate
        print(evaluate(args.input, embeddings))
