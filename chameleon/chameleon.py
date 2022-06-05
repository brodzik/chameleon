import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community
from scipy.io import arff
from sklearn.metrics import adjusted_rand_score, rand_score
from sklearn.metrics.pairwise import euclidean_distances
from tqdm.auto import tqdm


class Chameleon:
    def __init__(self, k=10, remove_edges_coef=0.5, min_ri=0, min_rc=0, min_ri_rc=0.5):
        self.k_ = k
        self.remove_edges_coef_ = remove_edges_coef
        self.min_ri_ = min_ri
        self.min_rc_ = min_rc
        self.min_ri_rc_ = min_ri_rc

        self.distance_matrix_ = None
        self.knng_ = None
        self.clusters_ = None
        self.cache_ = None
        self.cluster_history_ = None

        self.debug_draw_knng_ = False
        self.debug_draw_partition_ = False

    def fit(self, X):
        self.distance_matrix_ = euclidean_distances(X, X)
        self._build_knn_graph(range(len(X)))

        if self.debug_draw_knng_:
            plt.figure(figsize=(5, 5))
            nx.draw(self.knng_, pos=X, node_size=1, node_color="black")
            plt.show()

        self._partition_knn_graph(len(X))
        self._merge_clusters()

        return self

    def get_clusters(self, idx=-1):
        return self._simplify_clusters(self.cluster_history_[idx])

    def _build_knn_graph(self, X):
        self.knng_ = nx.Graph()

        for i in X:
            self.knng_.add_node(i)

        for a in X:
            for b in np.argsort(self.distance_matrix_[a])[1:self.k_ + 1]:
                self.knng_.add_edge(a, b, weight=self.distance_matrix_[a][b], similarity=1/self.distance_matrix_[a][b])

    def _partition_knn_graph(self, N):
        self.clusters_ = np.zeros(N, dtype=np.uint64)
        cluster_idx = 1

        curr = set(range(N))
        blacklist = set()

        knng = self.knng_.copy()

        while len(knng.nodes()) > 0:
            knng_copy = knng.copy()

            if self.debug_draw_partition_:
                plt.figure(figsize=(5, 5))
                nx.draw(knng, pos=X, node_size=1, node_color="black")
                plt.show()

            N_remove = int(self.remove_edges_coef_ * len(knng.edges()))
            i = 0
            for a, b, w in sorted(knng.edges(data=True), key=lambda x: x[2].get("weight", 1), reverse=True):
                if i >= N_remove:
                    break

                if knng.has_edge(a, b):
                    if knng.degree(a) > 1:
                        knng.remove_edge(a, b)
                        i += 1

                if knng.has_edge(b, a):
                    if knng.degree(b) > 1:
                        knng.remove_edge(b, a)
                        i += 1

            subgraphs = sorted(list(nx.connected_components(knng)), key=lambda t: len(t), reverse=True)

            if len(subgraphs) > 0:
                subgraph_vertices = subgraphs[0]
                knng_copy.remove_nodes_from(subgraph_vertices)
                self.clusters_[list(subgraph_vertices)] = cluster_idx

            cluster_idx += 1
            knng = knng_copy.copy()

    def _absolute_metrics(self, i, j):
        if (i, j) in self.cache_:
            return self.cache_[(i, j)]

        if (j, i) in self.cache_:
            return self.cache_[(j, i)]

        nodes_i = np.where(self.clusters_ == i)[0].tolist()
        nodes_j = np.where(self.clusters_ == j)[0].tolist()

        edges_i = self.knng_.subgraph(nodes_i).copy().edges(data=True)
        edges_j = self.knng_.subgraph(nodes_j).copy().edges(data=True)

        cluster = self.knng_.subgraph(nodes_i + nodes_j).copy()
        cluster.remove_edges_from(edges_i)
        cluster.remove_edges_from(edges_j)

        if cluster.number_of_nodes() <= 0 or cluster.number_of_edges() <= 0:
            EC = 0
            SEC = 0
        else:
            EC = cluster.size(weight="similarity")
            SEC = cluster.size(weight="similarity") / cluster.number_of_edges()

        result = (EC, SEC)

        self.cache_[(i, j)] = result

        return result

    def _internal_metrics(self, i):
        if i in self.cache_:
            return self.cache_[i]

        nodes = np.where(self.clusters_ == i)[0].tolist()

        cluster = self.knng_.subgraph(nodes).copy()

        if cluster.number_of_nodes() <= 0 or cluster.number_of_edges() <= 0:
            result = (0, 0, 0)
        else:
            Ci = cluster.size(weight="similarity")

            a, b = community.kernighan_lin_bisection(cluster, max_iter=20, weight="similarity", seed=42)

            edges_a = cluster.subgraph(a).copy().edges(data=True)
            edges_b = cluster.subgraph(b).copy().edges(data=True)

            cluster.remove_edges_from(edges_a)
            cluster.remove_edges_from(edges_b)

            if cluster.number_of_nodes() <= 0 or cluster.number_of_edges() <= 0:
                ECci = 0
                SECci = 0
            else:
                ECci = cluster.size(weight="similarity")
                SECci = cluster.size(weight="similarity") / cluster.number_of_edges()

            result = (ECci, SECci, Ci)

        self.cache_[i] = result

        return result

    def _merge_score(self, i, j):
        EC, SEC = self._absolute_metrics(i, j)
        ECci, SECci, Ci = self._internal_metrics(i)
        ECcj, SECcj, Cj = self._internal_metrics(j)

        ri = 2 * EC / (ECci + ECcj + 1e-6)
        rc = SEC / ((Ci / (Ci + Cj + 1e-6) * SECci) + (Cj / (Ci + Cj + 1e-6) * SECcj) + 1e-6)

        return ri, rc

    def _delete_cache(self, i, j):
        for x in np.unique(self.clusters_):
            if (i, x) in self.cache_:
                del self.cache_[(i, x)]

            if (x, i) in self.cache_:
                del self.cache_[(x, i)]

            if (j, x) in self.cache_:
                del self.cache_[(j, x)]

            if (x, j) in self.cache_:
                del self.cache_[(x, j)]

        if i in self.cache_:
            del self.cache_[i]

        if j in self.cache_:
            del self.cache_[j]

    def _merge_clusters(self):
        self.cache_ = {}
        self.cluster_history_ = [self.clusters_.copy()]

        while True:
            merged = False
            metrics = []

            for i, j in tqdm(list(itertools.combinations(np.unique(self.clusters_), 2))):
                ri, rc = self._merge_score(i, j)
                metrics.append((i, j, ri, rc, ri * rc**2))

            metrics.sort(key=lambda x: x[4], reverse=True)

            for i, j, ri, rc, ri_rc in metrics:
                if ri > self.min_ri_ and rc > self.min_rc_ and ri_rc > self.min_ri_rc_:
                    self.clusters_[np.where(self.clusters_ == j)] = i
                    merged = True
                    self._delete_cache(i, j)
                    break

            if merged:
                self.cluster_history_.append(self.clusters_.copy())
            else:
                break

    def _simplify_clusters(self, c):
        c2 = np.zeros_like(c)

        for i, x in enumerate(np.unique(c)):
            c2[np.where(c == x)] = i

        return c2


"""
if __name__ == "__main__":
    df = pd.DataFrame(arff.loadarff("../data/artificial/3-spiral.arff")[0])
    df["CLASS"] = df["class"]
    df["CLASS"] = df["CLASS"].map({x: i for i, x in enumerate(df["CLASS"].unique())})
    X = df[["x", "y"]].to_numpy()

    model = Chameleon(k=10).fit(X)

    c = model.get_clusters(-1)

    print(len(np.unique(c)))

    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=c, s=10)
    plt.show()

    print(rand_score(df["CLASS"].to_numpy(), c), adjusted_rand_score(df["CLASS"].to_numpy(), c))
"""
