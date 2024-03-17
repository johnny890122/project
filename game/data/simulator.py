import networkx as nx
import numpy as np
import copy
import itertools
from tqdm import trange
from scipy.stats import lognorm
from pathlib import Path
from argparse import ArgumentParser, Namespace
from datetime import datetime
from scipy import sparse

def hxa(g, method):
    G = g.copy()
    if method == 'HDA':
        dc = nx.degree_centrality(G)
    elif method == 'HBA':
        dc = nx.betweenness_centrality(G)
    elif method == 'HCA':
        dc = nx.closeness_centrality(G)
    elif method == 'HPRA':
        dc = nx.pagerank(G)
    keys = list(dc.keys())
    values = list(dc.values())
    maxTag = np.argmax(values)
    node = keys[maxTag]

    return node

def hxa(g, method):
    G = g.copy()
    if method == 'HDA':
        dc = nx.degree_centrality(G)
    elif method == 'HBA':
        dc = nx.betweenness_centrality(G)
    elif method == 'HCA':
        dc = nx.closeness_centrality(G)
    elif method == 'HPRA':
        dc = nx.pagerank(G)
    keys = list(dc.keys())
    values = list(dc.values())
    maxTag = np.argmax(values)
    node = keys[maxTag]
    
    return node

class CovertGenerator():
	def __init__(self, min_n, max_n, density, exposed_type="uniform", info_type="avg"):
		self.min_n = min_n
		self.max_n = max_n

		self.size = np.random.randint(self.max_n - self.min_n + 1) + self.min_n

		# parameter for G_mu
		self.exposed_type = exposed_type # uniform, ...
		self.info_type = info_type # avg, worst
		self.density = density

		# parameter for (1) brutal search
		self.total_iter = 10000
		self.search_patience = self.total_iter*0.1

		# for debuging
		self.G = None

	def G_total_dist(self, G):
		path = dict( nx.all_pairs_shortest_path_length(G) )
		dist_mat = [list(d.values()) for d in path.values()]
		return np.sum(list(itertools.chain(*dist_mat)))

	def G_diameter(self, G):
		return nx.diameter(G)

	def G_secrecy(self, G):
		degree = np.array(G.degree)

		if self.exposed_type == "uniform":
			return np.mean( 1 - ((degree[:,1] + 1)/G.size()) )
		else:
			raise Exception("Exposed type other than uniform hasn't implemented yet.")

	def G_information(self, G):
		if self.info_type == "avg":
			total_dist = self.G_total_dist(G)
			if total_dist == 0:
				return -1
			return G.size() * (G.size() - 1) / total_dist

		elif self.info_type == "worst":
			return G.size() * (G.size() - 1) / self.G_diameter(G)
		else:
			raise Exception("info_type only allowed: avg & worst.")

	def G_mu(self, G):
		return self.G_secrecy(G) * self.G_information(G)

	def simulate(self):
		best_score, tmp_score, patience = 0, 0, 0

		plot_lst = list()
		for i in range(self.total_iter):
			adjacency = np.random.rand(self.size, self.size) <= (self.density / 2)
			tmpG = nx.from_numpy_matrix(adjacency, nx.Graph)
			tmp_score = self.G_mu(tmpG)
			if tmp_score > best_score:
				self.G = copy.deepcopy(tmpG)
				best_score = tmp_score
				patience = 0
			else:
				patience += 1

			if patience > self.search_patience:
				break
		self.G.remove_edges_from(nx.selfloop_edges(self.G))

class DarkGenerator():
	def __init__(self, min_n, max_n, density):
        # parameter
		self.min_n = min_n
		self.max_n = max_n
		self.size = np.random.randint(self.max_n - self.min_n + 1) + self.min_n
		self.density = density

		self.beta = 1
		self.gamma = 1
		self.n0 = 5
		self.e0 = 5
		self.M = int((self.size * (self.size - 1) - self.n0) * self.density / (self.size-self.n0)) + 1

		self.adjacency_mat = np.zeros(self.size**2).reshape(self.size, self.size)

		self.timp_stamp = 0

		self.heavy_tail_dis = "log_normal"

		self.G = nx.from_numpy_array(self.adjacency_mat)

	def A_sample(self):
		lst = [self.size for _ in range(self.n0)]
		for i in range(self.size - self.n0, 0, -1):
			lst.append(i)
		return np.array(lst)

	def B_sample(self):
		if self.heavy_tail_dis == "log_normal":
			return lognorm.rvs(1, size=self.size)
		else:
			raise Exception("Dist other than log-norm hasn't implemented")

	def startG(self):
		adjacency = np.random.rand(self.n0, self.n0) < (self.e0 / (self.n0**2))
		for i, j in np.argwhere(adjacency == True):
			self.adjacency_mat[i][j] = 1
			self.adjacency_mat[j][i] = 1

	def dynamic_evolve(self, time):
		val = self.A_sample()[:time]**self.beta * self.B_sample()[:time]**self.gamma
		prob = val / np.sum(val)

		M1 = np.random.choice(np.arange(0, prob.size, 1), self.M, p=prob)
		M2 = np.random.choice(np.arange(0, prob.size, 1), self.M, p=prob)

		for i, j in zip(M1, M2):
			self.adjacency_mat[i][j] = 1
			self.adjacency_mat[j][i] = 1

	def rewire_node(self):
		degree = np.sum(self.adjacency_mat, axis=1)
		np.random.shuffle(degree)
		rewire_G = nx.expected_degree_graph(degree, selfloops=True)

		self.adjacency_mat = nx.adjacency_matrix(rewire_G).toarray()
		return degree

	def simulate(self):
		# step 1
		self.startG()

		# step 2
		for _ in range(self.size - self.n0):
			self.timp_stamp += 1
			self.dynamic_evolve(self.timp_stamp)
			self.rewire_node()

		self.G = nx.from_numpy_matrix(self.adjacency_mat)
		self.G.remove_edges_from(nx.selfloop_edges(self.G))

def fintuing_realG_generator(data_dir, file_name):
    choice = np.random.choice([1, 2, 3], 1, [0.2, 0.4, 0.4]).item()
    g = nx.read_gml(data_dir + file_name)
    
    node_mapping = {node: int(i) for i, node in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, node_mapping)
    G = g.copy()
    # num_removal = np.random.randint(1, int(g.number_of_nodes()*0.75))

    # if choice == 1: # Use whole graph
    #     pass
    # elif choice == 2: # Pure HXA-based removal
    #     method = np.random.choice(['HDA', 'HBA', 'HCA', 'HPRA'])
    #     while G.number_of_nodes() > g.number_of_nodes() - num_removal:
    #         node = hxa(G, method)
    #         G.remove_node(int(node))
    # elif choice == 3:
    #     while G.number_of_nodes() > g.number_of_nodes() - num_removal:
    #         method = np.random.choice(['HDA', 'HBA', 'HCA', 'HPRA', "RANDOM"])
    #         if method == "RANDOM":
    #             node = np.random.choice(list(G.nodes()))
    #         else:
    #             node = hxa(G, method)
    #         G.remove_node(int(node))

    # node_mapping = {node: int(i) for i, node in enumerate(G.nodes())}
    # G = nx.relabel_nodes(G, node_mapping)
    return G
