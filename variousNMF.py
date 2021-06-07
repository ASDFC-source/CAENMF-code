import math
import numpy as np
import copy
import networkx as nx
from numpy.core.fromnumeric import *


def Q(array, cluster):
    # 总边数
    m = sum(sum(array)) / 2
    k1 = np.sum(array, axis=1)
    k2 = k1.reshape(k1.shape[0], 1)
    # 节点度数积
    k1k2 = k1 * k2
    # 任意两点连接边数的期望值
    Eij = k1k2 / (2 * m)
    # 节点v和w的实际边数与随机网络下边数期望之差
    B = array - Eij
    # 获取节点、社区矩阵
    node_cluster = np.dot(cluster, np.transpose(cluster))
    results = np.dot(B, node_cluster)
    # 求和
    sum_results = np.trace(results)
    # 模块度计算
    Q = sum_results / (2 * m)
    return Q


def test2(A, CommunityNum, cG, eN, t, community, delta=1, alpha=0):
    def density(C):
        edgeNum = 0
        maxNum = len(C) * (len(C) - 1) / 2
        for (key, values) in cDegree.items():
            edgeNum += key * len(values)
        if maxNum == 0:
            return 0
        return float(edgeNum / maxNum) / 2

    def nodeRep():
        path = [
            r"./dataset/birthdeath/birthdeath.t01.emb",
            r"./dataset/birthdeath/birthdeath.t02.emb",
            r"./dataset/birthdeath/birthdeath.t03.emb",
            r"./dataset/birthdeath/birthdeath.t04.emb",
            r"./dataset/birthdeath/birthdeath.t05.emb",
            r"./dataset/birthdeath/birthdeath.t06.emb",
            r"./dataset/birthdeath/birthdeath.t07.emb",
            r"./dataset/birthdeath/birthdeath.t08.emb",
            r"./dataset/birthdeath/birthdeath.t09.emb",
            r"./dataset/birthdeath/birthdeath.t10.emb",
        ]

        with open(path[t]) as f:
            data = f.readlines()
            temp = np.zeros([len(A[t]), len(data[1].strip().split(' ')) - 1], dtype=float)
            sign = 0
            for e in data:
                if sign == 0:
                    sign = 1
                    continue
                r = e.strip().split(' ')
                idx = int(r[0]) - 1
                for j in range(1, len(r)):
                    temp[idx][j - 1] = float(r[j])
        rep = temp
        return rep

    if t != 0:
        idx = 0
        while idx < len(community):
            if len(community[idx]) < 3:
                del community[idx]
            else:
                idx += 1
        commDis = {}
        commDis2 = {}
        nodeC = {}
        edgeNum = {}
        for idx in range(len(community)):
            commDis[idx] = 0
            commDis2[idx] = 0
            edgeNum[idx] = 0
            for node in community[idx]:
                nodeC[node] = idx
        rep = nodeRep()
        O = np.zeros([len(A[t]), len(A[t])])
        D = np.zeros([len(A[t]), len(A[t])])
        for idx in range(len(community)):
            for i in community[idx]:
                for j in community[idx]:
                    O[i][j] = 1
                    commDis[nodeC[i]] += np.sum((rep[i] - rep[j]) ** 2)
                    edgeNum[nodeC[i]] += 1
        for k in commDis.keys():
            if commDis[k] != 0:
                commDis[k] /= (edgeNum[k] + 0.0001)
        for node in range(len(A[t])):
            if node in nodeC.keys() or node not in eN[t]:
                continue
            s = 0
            m = 1000000
            index = -1
            for k in range(len(community)):
                for n_ in community[k]:
                    s += np.sum((rep[node] - rep[n_]) ** 2)
                s /= len(community[k])
                s *= commDis[k]
                if s < m:
                    m = s
                    index = k
            if index == -1:
                continue
            for n_ in community[index]:
                O[node][n_] = 1 / (m + 0.00000001)
                O[n_][node] = 1 / (m + 0.00000001)
                if O[node][n_] > 1:
                    O[node][n_] = 1
                    O[n_][node] = 1
        temp = np.sum(O, axis=1)
        for i in range(len(temp)):
            D[i][i] = temp[i]
    W = np.random.random([len(A), CommunityNum[t]])
    H = np.random.random([CommunityNum[t], len(A)])
    for i in range(60):
        if t == 0:
            W *= A.dot(H.T) / (0.0000000000001 + W.dot(H).dot(H.T))
            H *= W.T.dot(A) / (0.0000000000001 + W.T.dot(W).dot(H))
        if t != 0:
            W *= A.dot(H.T) / (0.0000000000001 + W.dot(H).dot(H.T))
            H *= (W.T.dot(A) + delta * H.dot(O.T)) / (
                    0.0000000000001 + W.T.dot(W).dot(H) + delta * H.dot(D.T))
    H = H.T
    nodeCommunity = {}
    community = [[] for _ in range(CommunityNum[t])]
    X_ = np.zeros([len(A), CommunityNum[t]])
    C = [-1 for _ in range(len(A))]
    for row in range(len(A)):
        if row not in eN[t]:
            continue
        m = 0
        index = -1
        for col in range(CommunityNum[t]):
            if H[row][col] > m:
                m = H[row][col]
                index = col
        nodeCommunity[row] = index
        X_[row][index] = 1
        community[index].append(row)
        C[row] = index
    while -1 in C:
        C.remove(-1)
    dealCommunity = []
    for idx in range(len(community)):
        c = community[idx]
        subG = cG.subgraph(c)
        subD = subG.degree()
        if len(c) == 0:
            continue
        cDegree = {}
        for pair in subD:
            if pair[1] in cDegree.keys():
                cDegree[pair[1]].append(pair[0])
            else:
                cDegree[pair[1]] = [pair[0]]
        sortedC = sorted(cDegree)
        index = 0
        while 1:
            value = sortedC[index]
            n = cDegree[value][0]
            if density(c) > alpha or len(c) <= 3:
                break
            c.remove(n)
            subG = nx.subgraph(cG, c)
            subD = subG.degree()
            if len(c) == 0:
                continue
            cDegree = {}

            for pair in subD:
                if pair[1] in cDegree.keys():
                    cDegree[pair[1]].append(pair[0])
                else:
                    cDegree[pair[1]] = [pair[0]]
            sortedC = sorted(cDegree)
        dealCommunity.append(c)
    return X_, C, dealCommunity


def similar(A, cG):
    sim = []
    degree = []
    for t in range(len(A)):
        d_ = list(cG[t].degree())
        d1 = {}
        for pair in d_:
            d1[pair[0]] = pair[1]
        s = np.zeros([len(A[t]), len(A[t])])
        d = np.zeros([len(A[t]), len(A[t])])
        for i in range(len(A[t])):
            if not cG[t].has_node(i):
                continue
            for j in range(i, len(A[t])):
                if not cG[t].has_node(j):
                    continue
                su = 0
                a = set(list(cG[t].neighbors(i))) & set(list(cG[t].neighbors(j)))
                for node in a:
                    su += 1 / (math.log(d1[node]) + 0.0001)
                s[i][j] = su
                s[j][i] = su
        temp = np.sum(s, axis=1)
        for j in range(len(A[t])):
            d[j][j] = temp[j]
        sim.append(s)
        degree.append(d)
    return sim, degree


def ECGNMF(A, CommunityNum, S, D, alpha=0.2, beta=0.2):
    H = [np.random.random([len(A[t]), CommunityNum[t]]) for t in range(len(A))]
    W = [np.random.random([len(A[t]), CommunityNum[t]]) for t in range(len(A))]
    Z = [np.random.random([CommunityNum[t - 1], CommunityNum[t]]) for t in range(1, len(A))]
    for t in range(len(A)):
        for run in range(60):
            if t == 0:
                W[t] *= A[t].dot(H[t]) / (W[t].dot(H[t].T).dot(H[t]) + 0.0000001)
                H[t] *= A[t].T.dot(W[t]) / (H[t].dot(W[t].T).dot(W[t]) + 0.0000001)
            else:
                W[t] *= A[t].dot(H[t]) / (W[t].dot(H[t].T).dot(H[t]) + 0.0000001)
                H[t] *= (A[t].T.dot(W[t]) + alpha * H[t - 1].dot(Z[t - 1]) + beta * S[t - 1].dot(H[t])) / (
                        H[t].dot(W[t].T).dot(W[t]) + alpha * H[t] + beta * D[t - 1].dot(H[t]) + 0.0000001)
                Z[t - 1] *= H[t - 1].T.dot(H[t]) / (H[t - 1].T.dot(H[t - 1]).dot(Z[t - 1]))
    return H


def spertal(A, eN, CommunityNum):
    C = []
    for t in range(len(A)):
        nodeCount = len(A[t])
        D = np.zeros([nodeCount, nodeCount])
        D1 = np.zeros([nodeCount, nodeCount])
        temp = np.sum(A[t], axis=1)
        for i in range(len(A[t])):
            D[i][i] = temp[i]
            if i in eN[t] and temp[i] != 0:
                D1[i][i] = 1 / temp[i] ** 0.5
        L = D1.dot(A[t]).dot(D1)
        x, v = np.linalg.eig(L)
        x = np.real(x)
        v = np.real(v)
        idxVec = np.argsort(-x)
        M = np.zeros([len(A[t]), CommunityNum[t]])
        for i in range(CommunityNum[t]):
            M[:, i] = v[:, idxVec[i]]
        C.append(M)
    return C


def SE_NMF(A, CommunityNum, cG, eN, com, alpha=0.8, gamma=0.1):
    from sklearn.cluster import KMeans

    def density(C):
        edgeNum = 0
        maxNum = len(C) * (len(C) - 1) / 2
        for (key, values) in cDegree.items():
            edgeNum += key * len(values)
        if maxNum == 0:
            return 0
        return float(edgeNum / maxNum) / 2

    h = []
    for t in range(len(A)):
        if t != 0:
            adja = alpha * A[t] + (1 - alpha) * (A[t] - A[t - 1])
        else:
            adja = copy.deepcopy(A[t])
        nodeCount = len(adja)
        M = com[t]
        kmeans = KMeans(n_clusters=CommunityNum[t]).fit(M).labels_
        community = [[] for _ in range(CommunityNum[t])]
        for i in range(nodeCount):
            if i in eN[t]:
                community[kmeans[i]].append(i)

        for idx in range(len(community)):
            if len(community[idx]) == 0:
                continue
            c = community[idx]
            subG = cG[t].subgraph(c)
            subD = subG.degree()
            if len(c) == 0:
                continue
            cDegree = {}
            for pair in subD:
                if pair[1] in cDegree.keys():
                    cDegree[pair[1]].append(pair[0])
                else:
                    cDegree[pair[1]] = [pair[0]]
            sortedC = sorted(cDegree)
            index = 0
            while 1:
                value = sortedC[index]
                n = cDegree[value][0]
                if len(c) <= 5 or density(c) > 0.95:
                    break
                c.remove(n)
                subG = cG[t].subgraph(c)
                subD = subG.degree()
                if len(c) == 0:
                    continue
                cDegree = {}
                for pair in subD:
                    if pair[1] in cDegree.keys():
                        cDegree[pair[1]].append(pair[0])
                    else:
                        cDegree[pair[1]] = [pair[0]]
                sortedC = sorted(cDegree)
        Z = np.zeros([len(A[t]), CommunityNum[t]], dtype=np.float)
        for i in range(len(community)):
            for node in community[i]:
                Z[node][i] = 1
        adja += gamma * Z.dot(Z.T)
        B = np.random.random([len(A[t]), CommunityNum[t]])
        F = np.random.random([CommunityNum[t], len(A[t])])
        for i in range(30):
            B *= adja.dot(F.T) / (B.dot(F).dot(F.T) + 0.0000001)
            F *= B.T.dot(adja) / (B.T.dot(B).dot(F) + 0.0000001)
        h.append(F.T)
    return h
