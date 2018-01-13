
#---------------------------
# LinkMirage.py
# Author: Changchang Liu
#---------------------------

import numpy as np

def randWalk(graph, step):
    perGraph = np.empty(shape = [0, 2], dtype = int)
    node = list(set(graph[:,0].flat))
    for i in xrange(len(node)):
        startNode = node[i]
        count = 1
        nei = graph[np.where(graph[:,0] == startNode)][:,1]
        for j in xrange(len(nei)):
            iter = 10
            for loop in xrange(iter):
                currentNode = nei[j]
                for k in xrange(step-1):
                    currentNei = graph[np.where(graph[:,0] == currentNode)][:,1]
                    ranInt = np.random.randint(0,len(currentNei),1)[0]
                    nextNode = currentNei[ranInt]
                    currentNode = nextNode
                if startNode != currentNode and [startNode, currentNode] not in perGraph:
                    break
            if startNode != currentNode:
                if count == 1:
                    perGraph = np.row_stack((perGraph, [startNode, currentNode], [currentNode, startNode]))
                else:
                    prob = float(0.5*len(nei)-1)/(len(nei)-1)
                    if prob > 0: 
                        ranNum = np.random.rand()
                        if ranNum <= prob:
                            perGraph = np.row_stack((perGraph, [startNode, currentNode],[currentNode, startNode]))
            count += 1
    temp = np.ascontiguousarray(perGraph).view(np.dtype((np.void, perGraph.dtype.itemsize * perGraph.shape[1])))
    _, idx = np.unique(temp, return_index = True)
    perGraph = perGraph[idx]
    return perGraph

    
    
def randConn(graph, comm):
    commLabel = list(set(comm[:,1].flat))
    node = list(set(comm[:,0].flat))
    leftNode = comm[np.where(comm[:,1] == commLabel[0]),0]
    rightNode = comm[np.where(comm[:,1] == commLabel[1]),0]
    perGraph = np.empty(shape = [0, 2], dtype = int)
    for i in xrange(len(leftNode[0,:])):
        leftDeg = len(graph[np.where(graph[:,0] == leftNode[0,i])][:,1])
        for j in xrange(len(rightNode[0,:])):
            rightDeg = len(graph[np.where(graph[:,0] == rightNode[0,j])][:,1])
            prob = float(leftDeg*rightDeg*len(leftNode[0,:]))/len(graph[:,0])/(len(leftNode[0,:])+len(rightNode[0,:]))
            if prob > 0: 
                ranNum = np.random.rand()
                if ranNum <= prob:
                    perGraph = np.row_stack((perGraph, [leftNode[0,i], rightNode[0,j]], [rightNode[0,j], leftNode[0,i]]))
    temp = np.ascontiguousarray(perGraph).view(np.dtype((np.void, perGraph.dtype.itemsize * perGraph.shape[1])))
    _, idx = np.unique(temp, return_index = True)
    perGraph = perGraph[idx]
    return perGraph

    
    
def staPerb(graph, comm, step):
    perGraph = np.empty(shape = [0, 2], dtype = int)
    commLabel = list(set(comm[:,1].flat))
    node = list(set(comm[:,0].flat))
    for label in commLabel:
        commGraph = np.empty(shape = [0, 2], dtype = int)
        commPerGraph = np.empty(shape = [0, 2], dtype = int)
        for i in xrange(len(graph[:,0])):
            if comm[np.where(comm[:,0] == graph[i,0]),1] == label and comm[np.where(comm[:,0] == graph[i,1]),1] == label:
                commGraph = np.row_stack((commGraph, [graph[i,0], graph[i,1]], [graph[i,1], graph[i,0]]))
        commPerGraph = randWalk(commGraph, step)
        perGraph = np.row_stack((perGraph, commPerGraph))
    for label1 in commLabel:
        for label2 in commLabel:
            if label1 < label2:
                interGraph = np.empty(shape = [0, 2], dtype = int)
                interPerGraph = np.empty(shape = [0, 2], dtype = int)
                interComm = np.empty(shape = [0, 2], dtype=int)
                for i in xrange(len(graph[:,0])):
                    if comm[np.where(comm[:,0] == graph[i,0]),1] == label1 and comm[np.where(comm[:,0] == graph[i,1]),1] == label2:
                        interGraph = np.row_stack((interGraph, [graph[i,0], graph[i,1]], [graph[i,1], graph[i,0]]))
                interComm = np.vstack((interComm, comm[np.where(comm[:,1] == label1),:][0], comm[np.where(comm[:,1] == label2),:][0]))
                interPerGraph = randConn(interGraph, interComm)
                perGraph = np.row_stack((perGraph, interPerGraph))
    temp = np.ascontiguousarray(perGraph).view(np.dtype((np.void, perGraph.dtype.itemsize * perGraph.shape[1])))
    _, idx = np.unique(temp, return_index = True)
    perGraph = perGraph[idx]
    return perGraph

   
def temPerb(graph1, graph2, comm1, comm2, step, perGraph1):
    commLabel1 = list(set(comm1[:,1].flat))
    commLabel2 = list(set(comm2[:,1].flat))
    sameLabel = list(set(commLabel2).intersection(set(commLabel1)))
    diffLabel = list(set(commLabel2).difference(set(commLabel1)))
    perGraph2 = np.empty(shape = [0, 2], dtype = int)
    for i in xrange(len(perGraph1[:,0])):
        if comm1[np.where(comm1[:,0] == perGraph1[i,0]),1] == comm1[np.where(comm1[:,0] == perGraph1[i,1]),1] and (comm1[np.where(comm1[:,0] == perGraph1[i,0]),1] in sameLabel):
            perGraph2=np.row_stack((perGraph2, [perGraph1[i,0], perGraph1[i,1]]))
    for label in diffLabel:
        commGraph = np.empty(shape = [0, 2], dtype = int)
        commPerGraph = np.empty(shape = [0, 2], dtype = int)
        for i in xrange(len(graph2[:,0])):
            if comm2[np.where(comm2[:,0] == graph2[i,0]),1] == label and comm2[np.where(comm2[:,0] == graph2[i,1]),1] == label:
                commGraph = np.row_stack((commGraph, [graph2[i,0], graph2[i,1]]))
        commPerGraph = randWalk(commGraph, step)
    perGraph2 = np.row_stack((perGraph2, commPerGraph))
    for label1 in commLabel2:
        for label2 in commLabel2:
            if label1 < label2:
                interGraph = np.empty(shape = [0, 2], dtype = int)
                interPerGraph = np.empty(shape = [0, 2], dtype = int)
                interComm = np.empty(shape = [0, 2], dtype = int)
                for i in xrange(len(graph2[:,0])):
                    if comm2[np.where(comm2[:,0] == graph2[i,0]),1] == label1 and comm2[np.where(comm2[:,0] == graph2[i,1]),1] == label2:
                        interGraph = np.row_stack((interGraph, [graph2[i,0], graph2[i,1]], [graph2[i,1], graph2[i,0]]))
                interComm = np.vstack((interComm, comm2[np.where(comm2[:,1] == label1),:][0], comm2[np.where(comm2[:,1] == label2),:][0]))
                interPerGraph = randConn(interGraph, interComm)
                perGraph2 = np.row_stack((perGraph2, interPerGraph))
    temp = np.ascontiguousarray(perGraph2).view(np.dtype((np.void, perGraph2.dtype.itemsize * perGraph2.shape[1])))
    _, idx = np.unique(temp, return_index = True)
    perGraph2 = perGraph2[idx]
    return perGraph2


