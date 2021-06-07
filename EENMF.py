import numpy as np
import time
import variousNMF
from sklearn import metrics
import networkx as nx

collectPath = [
    r"./dataset/birthdeath/birthdeath.t01.edges",
    r"./dataset/birthdeath/birthdeath.t02.edges",
    r"./dataset/birthdeath/birthdeath.t03.edges",
    r"./dataset/birthdeath/birthdeath.t04.edges",
    r"./dataset/birthdeath/birthdeath.t05.edges",
    r"./dataset/birthdeath/birthdeath.t06.edges",
    r"./dataset/birthdeath/birthdeath.t07.edges",
    r"./dataset/birthdeath/birthdeath.t08.edges",
    r"./dataset/birthdeath/birthdeath.t09.edges",
    r"./dataset/birthdeath/birthdeath.t10.edges",
]
startTime = time.time()
maxNode = 0
node = []
aaa = []
existNode = []
collectEdge = []
for path in collectPath:
    edge = []
    tempNode = []
    with open(path) as f:
        data = f.readlines()
        for l in data:
            edges = l.strip().split(' ')
            edge.append((int(edges[0]) - 1, int(edges[1]) - 1))
            tempNode.append(int(edges[0]) - 1)
            tempNode.append(int(edges[1]) - 1)
            if maxNode < int(edges[0]) - 1:
                maxNode = int(edges[0]) - 1
            if maxNode < int(edges[1]) - 1:
                maxNode = int(edges[1]) - 1
    node.extend(tempNode)
    aaa.append(len(set(tempNode)))
    existNode.append(list(set(tempNode)))
    collectEdge.append(edge)

nodeNum = maxNode + 1
isDirect = False
adjacency = []
count = 0
for edge in collectEdge:
    A = np.zeros([nodeNum, nodeNum], dtype=np.float)
    node = []
    for pair in edge:
        node1 = pair[0]
        node2 = pair[1]
        A[node1][node2] = 1
        if not isDirect:
            A[node2][node1] = 1
    count += 1
    adjacency.append(A)
endTime = time.time()
trueCommunityCollection = []
communityFilePath = [
    r"./dataset/birthdeath/birthdeath.t01.comm",
    r"./dataset/birthdeath/birthdeath.t02.comm",
    r"./dataset/birthdeath/birthdeath.t03.comm",
    r"./dataset/birthdeath/birthdeath.t04.comm",
    r"./dataset/birthdeath/birthdeath.t05.comm",
    r"./dataset/birthdeath/birthdeath.t06.comm",
    r"./dataset/birthdeath/birthdeath.t07.comm",
    r"./dataset/birthdeath/birthdeath.t08.comm",
    r"./dataset/birthdeath/birthdeath.t09.comm",
    r"./dataset/birthdeath/birthdeath.t10.comm",
]
count = 0
for path in communityFilePath:
    trueCommunity = [-1 for _ in range(nodeNum)]
    with open(path) as f:
        data = f.readlines()
        c = 0

        for l in data:
            edges = l.strip().split(' ')
            for node in edges:
                trueCommunity[int(node) - 1] = c
            c += 1
            # trueCommunity[int(edges[0]) - 1] = int(edges[1])

    while -1 in trueCommunity:
        trueCommunity.remove(-1)
    print(aaa[count] - len(trueCommunity))
    while len(trueCommunity) != aaa[count]:
        trueCommunity.append(0)
    trueCommunityCollection.append(trueCommunity)
    count += 1
collectG = []
for edge in collectEdge:
    G = nx.Graph()
    G.add_edges_from(edge)
    # nx.draw(G)

    # plt.show()
    collectG.append(G)
# 12, 12, 12, 12, 12, 12, 12, 12, 12, 12
# 36, 36, 36, 36, 36, 36, 36, 36, 36, 36
# 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
# 11, 11, 11, 11, 11, 11, 11, 11, 11, 11
communityNum = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
print(communityNum)
print('-----------------------------------------------------------------------------------------')
runTime = 1
print('PENMF')
p = [0.75, 0.8, 0.85, 0.9, 0.95]
p_ = [0.1, 0.2, 0.3, 0.4, 0.5]
p__ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
m_q = 0
m_n = 0
m_delta = 0
m_alpha = 0
m_beta = 0
qList = []
nList = []
for delta in p_:
    for alpha in [0.9, 0.95]:
        print(delta, alpha)
        m_q1 = np.zeros(len(adjacency), dtype=float)
        m_n1 = np.zeros(len(adjacency), dtype=float)
        for ite in range(runTime):
            startTime = time.time()
            Q = np.zeros(len(adjacency), dtype=float)
            NMI = np.zeros(len(adjacency), dtype=float)
            community = []
            for t in range(len(adjacency)):
                A = adjacency[t]
                G = nx.Graph()
                G.add_edges_from(collectEdge[t])
                H, W, community = variousNMF.test2(A, communityNum, G, existNode, t, community, delta=delta,
                                                   alpha=alpha)
                Q[t] = variousNMF.Q(A, H)
                NMI[t] = metrics.normalized_mutual_info_score(W, trueCommunityCollection[t])
            print("community num：", communityNum)
            print("modularity in each snapshot:")
            print(Q)
            print("NMI in each snapshot:")
            print(NMI)
            print("average modularity：", np.sum(Q) / len(adjacency))
            print("average NMI：", np.sum(NMI) / len(adjacency))
            m_q1 += Q
            m_n1 += NMI
            endTime = time.time()
            print("time cost：", endTime - startTime)
            print("iteration num：", ite)
            print("-----------------------------------------------------")
        if np.sum(m_q1) + np.sum(m_n1) > np.sum(m_q) + np.sum(m_n):
            m_q = m_q1
            m_n = m_n1
            m_delta = delta
            m_alpha = alpha
print('\n')
print('\n')

m_q_ECG = 0
m_n_ECG = 0
m_delta_ECG = 0
m_alpha_ECG = 0
S, D = variousNMF.similar(adjacency, collectG)
for alpha in p__:
    for delta in p__:
        print("ECG", alpha, delta)
        m_q2 = np.zeros(len(adjacency), dtype=float)
        m_n2 = np.zeros(len(adjacency), dtype=float)
        for ite in range(0):
            startTime = time.time()
            X = variousNMF.ECGNMF(adjacency, communityNum, S, D, alpha=alpha, beta=delta)
            Q = np.zeros(len(adjacency), dtype=float)
            NMI = np.zeros(len(adjacency), dtype=float)
            for t in range(len(adjacency)):
                communityIndex = [-1 for _ in range(len(adjacency[t]))]
                temp = {}
                cluster = np.zeros([len(adjacency[t]), communityNum[t]])
                for i in range(len(X[t])):
                    if i not in existNode[t]:
                        continue
                    m = 0
                    index = 0
                    for j in range(len(X[t][0])):
                        if X[t][i][j] > m:
                            m = X[t][i][j]
                            index = j
                    if index == -1:
                        print("error")
                        break
                    temp[i] = index
                    communityIndex[i] = index
                    cluster[i][index] = 1
                while -1 in communityIndex:
                    communityIndex.remove(-1)
                Q[t] = variousNMF.Q(adjacency[t], cluster)
                NMI[t] = metrics.normalized_mutual_info_score(communityIndex, trueCommunityCollection[t])

            print("community num：", communityNum)
            print("modularity in each snapshot:")
            print(Q)
            print("NMI in each snapshot:")
            print(NMI)
            print("average modularity：", np.sum(Q) / len(adjacency))
            print("平均NMI：", np.sum(NMI) / len(adjacency))
            m_q2 += Q
            m_n2 += NMI
            endTime = time.time()
            print("time cost：", endTime - startTime)
            print("iteration num：", ite)
            print("----------------------------------------------")
            if np.sum(m_q2) + np.sum(m_n2) > np.sum(m_q_ECG) + np.sum(m_n_ECG):
                m_n_ECG = m_n2
                m_q_ECG = m_q2
                m_alpha_ECG = alpha
                m_delta_ECG = delta
print('\n')
print('\n')

m_q_SE = 0
m_n_SE = 0
m_delta_SE = 0
m_alpha_SE = 0
C = variousNMF.spertal(adjacency, existNode, communityNum)
for alpha in p__:
    for delta in p__:
        print("se_NMF", alpha, delta)
        m_q4 = np.zeros(len(adjacency), dtype=float)
        m_n4 = np.zeros(len(adjacency), dtype=float)
        for ite in range(0):
            startTime = time.time()
            X = variousNMF.SE_NMF(adjacency, communityNum, collectG, existNode, C, alpha=alpha, gamma=delta)
            Q = np.zeros(len(adjacency), dtype=float)
            NMI = np.zeros(len(adjacency), dtype=float)
            for t in range(len(adjacency)):
                communityIndex = [-1 for _ in range(len(adjacency[t]))]
                temp = {}
                cluster = np.zeros([len(adjacency[t]), communityNum[t]])
                for i in range(len(X[t])):
                    if i not in existNode[t]:
                        continue

                    m = 0
                    index = 0
                    for j in range(len(X[t][0])):
                        if X[t][i][j] > m:
                            m = X[t][i][j]
                            index = j
                    if index == -1:
                        print("error")
                        break
                    temp[i] = index
                    communityIndex[i] = index
                    cluster[i][index] = 1
                while -1 in communityIndex:
                    communityIndex.remove(-1)
                Q[t] = variousNMF.Q(adjacency[t], cluster)
                NMI[t] = metrics.normalized_mutual_info_score(communityIndex, trueCommunityCollection[t])
            print("community num：", communityNum)
            print("modularity in each snapshot:")
            print(Q)
            print("NMI in each snapshot:")
            print(NMI)
            print("average modularity：", np.sum(Q) / len(adjacency))
            print("平均NMI：", np.sum(NMI) / len(adjacency))
            m_q4 += Q
            m_n4 += NMI
            endTime = time.time()
            print("time cost：", endTime - startTime)
            print("iteration num：", ite)
            print("----------------------------------------------")
            if np.sum(m_q4) + np.sum(m_n4) > np.sum(m_q_SE) + np.sum(m_n_SE):
                m_n_SE = m_n4
                m_q_SE = m_q4
                m_alpha_SE = alpha
                m_delta_SE = delta


print(m_delta, m_alpha, m_beta)
print("modularity(EENMF)：", m_q / runTime)
print("average modularity(EENMF)：", np.sum(m_q / runTime) / len(adjacency))
print("NMI(EENMF)：", m_n / runTime)
print("average NMI(EENMF)：", np.sum(m_n / runTime) / len(adjacency))
print("----------------------------------------------")
print(m_alpha_ECG, m_delta_ECG)
print("modularity(ECGNMF)：", m_q_ECG / runTime)
print("average modularity(ECGNMF)：", np.sum(m_q_ECG / runTime) / len(adjacency))
print("NMI(ECGNMF)：", m_n_ECG / runTime)
print("average NMI(ECGNMF)：", np.sum(m_n_ECG / runTime) / len(adjacency))
print("----------------------------------------------")
print(m_alpha_SE, m_delta_SE)
print("modularity(se-NMF)：", m_q_SE / runTime)
print("average modularity(se-NMF)：", np.sum(m_q_SE / runTime) / len(adjacency))
print("NMI(se-NMF)：", m_n_SE / runTime)
print("average NMI(se-NMF)：", np.sum(m_n_SE / runTime) / len(adjacency))
