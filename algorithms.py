import itertools
import networkx as nx
import numpy as np

class BipartiteMatching:
    def __init__(self, G):
        self.G = G
        self.n = len(G.nodes)
        #L, R = nx.bipartite.sets(G)
        L, R = list(range(0, int(self.n/2))), list(range(int(self.n/2), self.n))
        self.L = list(L)
        self.R = list(R)

    def Lpm_get_feasible(self, pred):
        eps = max([self.G[e[0]][e[1]]['w'] - pred[e[0]] + pred[e[1]] for e in self.G.edges()])    
        proj = np.zeros(self.n)
        proj[self.L], proj[self.R] = pred[self.L] + (eps/2) * np.ones(len(self.L)), pred[self.R] - (eps/2) * np.ones(len(self.R))
        return np.round(proj)
    
    def L1_get_feasible(self, pred):
        pred = np.round(pred)
        proj = np.zeros(self.n)
        delta = np.zeros(self.n)
        H = self.G.copy()
        while len(H.edges()) > 0:
            i = list(H.nodes())[0]
            if len(H.adj[i]) == 0:
                H.remove_node(i)
            else:
                while len(H.adj[i]) > 0:
                    rmax = -np.inf
                    jmax = -1
                    for j in H.adj[i]:
                        eps = H[i][j]['w'] - pred[i] + pred[j]
                        if eps > rmax:
                            rmax = eps
                            jmax = j
                    H.remove_node(i)
                    delta[i] = max(0, rmax)
                    i = jmax
        proj[self.L], proj[self.R] = pred[self.L] + delta[self.L], pred[self.R] - delta[self.L]
        return proj
    
    def steepest_descent(self, p):
        itr = 0
        while True:
            itr += 1
        
            UE = [(e[0], e[1]) for e in self.G.edges() if abs(p[e[0]] - p[e[1]] - self.G[e[0]][e[1]]['w']) < 1e-10] # tight edges for unweighted graph
            UG = nx.Graph()
            UG.add_edges_from(UE)
        
            ans = dict()
            cover = set()
            for SG in [UG.subgraph(C).copy() for C in nx.connected_components(UG)]:
                matching = nx.bipartite.hopcroft_karp_matching(SG)
                ans |= matching
                cover |= nx.bipartite.to_vertex_cover(SG, matching)
        
            if len(cover) >= self.n/2:
                break
            S, T = cover & set(self.L), cover & set(self.R)
        
            LmS = list(set(self.L) - S)
            RmT = list(set(self.R) - T)
            lam = min([p[i] - p[j] - self.G[i][j]['w'] for i, j in itertools.product(LmS, RmT) if (i, j) in self.G.edges()])
        
            S = list(S)
            p[S], p[RmT] = p[S] + lam * np.ones(len(S)), p[RmT] + lam * np.ones(len(RmT))

        return {'dual': p, 'num_iter': itr}