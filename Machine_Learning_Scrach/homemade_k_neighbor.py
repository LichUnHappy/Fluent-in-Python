# 手写k近邻
import numpy as np
import operator

# def createDataSet():
#     group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels

# def classify0(inX, dataSet, labels, k):
#     dataSetSize = dataSet.shape[0]
#     diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
#     sqDiffMat = diffMat ** 2
#     sqDistances = sqDiffMat.sum(axis=1)
#     distances = sqDistances ** 0.5
#     sortedDistIndices = distances.argsort()
#     print(sortedDistIndices[:])
#     classCount = {}
#     #########################################################################################
#     for i in range(k):
#         voteILabel = labels[sortedDistIndices[i]]
#         classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
#     sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
#     #########################################################################################
#     return sortedClassCount[0][0]

# group, labels = createDataSet()
# print(classify0([0,0], group, labels, 3))


def file2matrix(filename):
    fr = open(filename)
    arrayofLines = fr.readlines()
    numberofLines = len(arrayofLines)
    returnMat = np.zeros((numberofLines, 3))
    classLableVector = []
    index = 0
    for line in arrayofLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLableVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLableVector
