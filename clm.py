import graph_tool
import graph_tool.stats
import graph_tool.collection
import graph_tool.centrality
import graph_tool.spectral
import networkx as nx
import numpy as np

import argparse
import logging
import time

logger = logging.getLogger()

parser = argparse.ArgumentParser(
    description='Run CLM model of cascading power grid failures.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('NET', type=str, 
    help='Which grid network to use', choices=['western','karate','ieee300'])
parser.add_argument('--alpha', type=float, help="Initial capacity to load ratio", default=1.0) 
parser.add_argument('--r', type=float, help="Heterogenenity of loads", default=0.1) 
parser.add_argument('--num_iters', type=int, help="Number of cascade iterations to perform", default=15)
parser.add_argument('--num_perts', type=int, help="Number of initial perturbations to sample", default=10)

#parser.add_argument('--debug', help='Output debugging information', action='store_true')
args = parser.parse_args()

#if args.debug:
#  logger.setLevel(logging.DEBUG)


class CLMLoadings(object):
    # Class that computes loadings for CLM model
    def __init__(self, G):
        self.G = G
        self.N = G.num_vertices()
        self.notdiag = (1-np.eye(self.N)).astype('bool')

    def __str__(self):
        return "N=%d, <degree>=%0.2f, class=%s" % \
           (self.N, graph_tool.stats.vertex_average(self.G, 'total')[0], 
            self.__class__.__name__)

    def get_loadings(self, dists):
        vprop, _ =graph_tool.centrality.betweenness(self.G, weight=dists)
        return vprop.a

    def _get_shortest_paths(self, dists):
        dmap = graph_tool.topology.shortest_distance(self.G, weights=dists)
        return dmap.get_2d_array(range(self.N))

    def get_efficiency(self, dists):
        mx = self._get_shortest_paths(dists)
        return (1./mx[self.notdiag]).mean()


class CLMLoadingsGensVsLoads(CLMLoadings):
    # Class that computes loadings for CLM while differentiating between
    # generator and load buses
    def __init__(self, G, is_generator_bus):
        self.is_generator_bus = is_generator_bus
        self.is_gen_set       = np.flatnonzero(is_generator_bus)
        self.is_not_gen_set   = np.flatnonzero(~is_generator_bus)
        super(CLMLoadingsGensVsLoads,self).__init__(G)

    def get_loadings(self, dists):
        # Convert from graph-tool to networkx format
        mx = graph_tool.spectral.adjacency(G, weight=dists).T
        nxG = nx.from_scipy_sparse_matrix(mx, create_using=nx.Graph())

        cnts = np.zeros(self.N)
        d=nx.centrality.betweenness_centrality_subset(nxG, 
            sources=self.is_gen_set, targets=self.is_not_gen_set, weight='weight')
        ixs, vals = map(list, zip(*d.iteritems()))
        cnts[ixs] = vals
        return cnts

    def get_efficiency(self, dists):
        mx = self._get_shortest_paths(dists)
        return (1./mx[self.is_generator_bus,:][:,~self.is_generator_bus]).mean()



if args.NET == 'ieee300':
    from pypower import case300
    d=case300.case300()
    bus_idmap = { busid : ix for ix, busid in enumerate(d['bus'][:,0].astype('int')) }
    edgelist = [(bus_idmap[s], bus_idmap[e]) for s, e in d['branch'][:,0:2].astype('int')]
    G = graph_tool.Graph(directed=False)
    verts = list(G.add_vertex(len(bus_idmap)))
    for s, e in edgelist:
        G.add_edge(verts[s],verts[e])
    c = CLMLoadingsGensVsLoads(G, d['bus'][:,1]==2)
    
elif args.NET == 'western':
    c = CLMLoadings(graph_tool.collection.data["power"])

elif args.NET == 'karate':
    c = CLMLoadings(graph_tool.collection.data["karate"])

else:
    raise Exception("Dont know how to load network %s"%args.NET)


print "# Running %s network: %s" % (args.NET, str(c))

row_format ="{:>10}{:>7}{:>7}{:>7}{:>6}{:>15.5}{:>15.5}{:>8.4}"
print row_format.format("Network", "Alpha", "R", "PertId", "t", "Eff", "Damage", "RunT")

np.set_printoptions(precision=3)

init_time = time.time()
init_effs = c.G.new_edge_property("float") 
init_effs.set_value(1.0)

dists = c.G.new_edge_property("float") 
dists.a = 1.0/(init_effs.a + 1e-6)

heterogeneity_q = 1 + (np.random.uniform(size=c.N)*2*args.r-args.r)
loadings = c.get_loadings(dists)
capacities = args.alpha * heterogeneity_q * loadings

edgesources = np.array([int(e.source()) for e in c.G.edges()]) 
edgetargets = np.array([int(e.target()) for e in c.G.edges()]) 
edge_ones   = np.ones(len(edgetargets))

init_efficiency = c.get_efficiency(dists)
run_time = time.time() - init_time

print row_format.format(args.NET, args.alpha, args.r, -1, 0, init_efficiency, 0.0, run_time)

for ndx in range(args.num_perts):
    c_init_effs = init_effs.copy()

    cutnode = np.random.randint(c.N)
    v = c.G.vertex(cutnode)
    for e in v.all_edges():
        c_init_effs[e] = 0.0
            
    c_effs = c_init_effs.copy()

    for t in range(args.num_iters):
        iter_init_time = time.time()
        dists.a = 1.0/(c_effs.a + 1e-6)
        loadings = c.get_loadings(dists)
        mult = edge_ones.copy()
        ixs1 = loadings[edgesources]>capacities[edgesources]
        ixs2 = loadings[edgetargets]>capacities[edgetargets]
        mult[ixs1] = capacities[edgesources][ixs1]/loadings[edgesources][ixs1]
        mult[ixs2] = np.minimum(mult[ixs2], capacities[edgetargets][ixs2]/loadings[edgetargets][ixs2])
        c_effs.a = np.multiply(c_init_effs.a, mult)

        cur_efficiency = c.get_efficiency(dists)
        iter_time = time.time() - iter_init_time
        cur_damage = (init_efficiency-cur_efficiency)/init_efficiency
        print row_format.format(args.NET, args.alpha, args.r, cutnode, t, cur_efficiency, cur_damage, iter_time)
