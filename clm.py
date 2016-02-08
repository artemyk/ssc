import numpy as np
import networkx as nx

import argparse
import logging
import time

try:
    import graph_tool
    import graph_tool.centrality
    import graph_tool.stats
    import graph_tool.spectral
except ImportError:
    print "# graph_tool not found"

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
parser.add_argument('--rndseed', type=int, help="Random seed") 
parser.add_argument('--engine', type=str, help='Which network package to use', choices=['graphtool','networkx'])

#parser.add_argument('--debug', help='Output debugging information', action='store_true')
args = parser.parse_args()

#if args.debug:
#  logger.setLevel(logging.DEBUG)

class gBase(object):
    def __str__(self):
        return "N=%d, <degree>=%0.2f" % (self.N, self.avg_degree())

    def get_efficiency(self, dists):
        mx = self.shortest_paths(dists)
        mx[np.eye(mx.shape[0]).astype('bool')] = np.nan
        mx = 1.0/mx
        if self.is_generator_bus is not None:
            mx = mx[self.is_generator_bus,:][:,~self.is_generator_bus]
        return np.nanmean(mx)

    def _init_edges(self, edgelist):
        self.num_edges = len(edgelist)
        self.edgesources = np.array([e[0] for e in edgelist]) 
        self.edgetargets = np.array([e[1] for e in edgelist]) 
        self._edge_ixs = {}
        for ndx, e in enumerate(edgelist):
            self._edge_ixs[e] = ndx
            self._edge_ixs[(e[1],e[0])] = ndx

        
class gtGraph(gBase):
    def __init__(self, graph, is_generator_bus=None):
        #if is_generator_bus is not None:
        #    raise Exception('is_generator_bus not supported')
        self.graph = graph
        self.N = graph.num_vertices()
        edgelist = [(int(e.source()), int(e.target())) for e in graph.edges()]
        self._init_edges(edgelist)
        self.is_generator_bus = is_generator_bus
        if is_generator_bus is not None:
            self.gn_set      = np.flatnonzero(is_generator_bus)
            self.gn_set_ndx  = { vid:ndx for ndx, vid in enumerate(self.gn_set)}
            self.ld_set      = np.flatnonzero(~is_generator_bus)
            self.ld_set_ndx  = { vid:ndx for ndx, vid in enumerate(self.ld_set)}
            self.gn_subgraph = graph_tool.Graph()
            self.ld_subgraph = graph_tool.Graph()
            verts1 = list(self.gn_subgraph.add_vertex(len(self.gn_set)))
            verts2 = list(self.ld_subgraph.add_vertex(len(self.ld_set)))
            for e in edgelist:
                if e[0] in self.gn_set and e[1] in self.gn_set:
                    self.gn_subgraph.add_edge(verts1[self.gn_set_ndx[e[0]]],verts1[self.gn_set_ndx[e[1]]])
                if e[0] in self.ld_set and e[1] in self.ld_set:
                    self.ld_subgraph.add_edge(verts2[self.ld_set_ndx[e[0]]],verts2[self.ld_set_ndx[e[1]]])



    @classmethod
    def construct(cls, N, edgelist, is_generator_bus=None):
        #if is_generator_bus is not None:
        #    raise Exception('is_generator_bus not supported')
        graph = graph_tool.Graph(directed=False)
        verts = list(graph.add_vertex(len(bus_idmap)))
        for s, e in edgelist:
            graph.add_edge(verts[s],verts[e])
        return cls(graph, is_generator_bus)

    def avg_degree(self):
        return graph_tool.stats.vertex_average(self.graph, 'total')[0]

    def shortest_paths(self, dists):
        dmap = graph_tool.topology.shortest_distance(self.graph, weights=self.graph.new_ep('float', vals=dists))
        return dmap.get_2d_array(range(self.N))

    def get_loadings(self, dists):
        vprop, _ =graph_tool.centrality.betweenness(self.graph, weight=self.graph.new_ep('float', vals=dists), norm=False)
        ret = vprop.a
        if self.is_generator_bus is not None:
            vpropG, _ =graph_tool.centrality.betweenness(self.gn_subgraph, weight=self.graph.new_ep('float', vals=dists[self.gn_set]), norm=False)
            ret[self.is_generator_bus] -= vpropG.a
            vpropL, _ =graph_tool.centrality.betweenness(self.ld_subgraph, weight=self.graph.new_ep('float', vals=dists[self.ld_set]), norm=False)
            ret[~self.is_generator_bus] -= vpropL.a
        return ret

    def toNx(self):
        # Convert from graph-tool to networkx format
        mx = graph_tool.spectral.adjacency(self.graph).T
        nxG = nx.from_scipy_sparse_matrix(mx, create_using=nx.Graph())
        return nxG

    def neighbor_edges(self, node):
        v = self.graph.vertex(node)
        return [self._edge_ixs[(int(e.source()),int(e.target()))] for e in v.all_edges()]


class nxGraph(gBase):
    def __init__(self, graph, is_generator_bus=None):
        self.graph = graph
        self.N = len(graph)
        self.is_generator_bus = is_generator_bus
        if is_generator_bus is not None:
            self.gn_set   = np.flatnonzero(is_generator_bus)
            self.ld_set   = np.flatnonzero(~is_generator_bus)
        self._init_edges(graph.edges())

    @classmethod
    def construct(cls, N, edgelist, is_generator_bus=None):
        graph = nx.Graph()
        graph.add_nodes_from(range(N))
        graph.add_edges_from(edgelist)
        return cls(graph, is_generator_bus)

    def avg_degree(self):
        return np.mean(nx.degree(self.graph).values())

    def shortest_paths(self, dists):
        mx = np.inf * np.ones((self.N, self.N))
        nx.set_edge_attributes(self.graph, 'weight', dict(zip(self.graph.edges(), dists)))
        d = nx.shortest_path_length(self.graph, weight='weight')
        for i, v in d.iteritems():
            mx[i, v.keys()] = v.values()
        return mx

    def get_loadings(self, dists):
        cnts = np.zeros(self.N)
        args = {}
        nx.set_edge_attributes(self.graph, 'weight', dict(zip(self.graph.edges(), dists)))
        if self.is_generator_bus is not None:
            d=nx.centrality.betweenness_centrality_subset(self.graph, weight='weight', 
                sources=self.gn_set, targets=self.ld_set)
        else:
            d=nx.centrality.betweenness_centrality(self.graph, 
                weight='weight', normalized=False)
        ixs, vals = map(list, zip(*d.iteritems()))
        cnts[ixs] = vals
        return cnts        

    def neighbor_edges(self, node):
        return [self._edge_ixs[(node,n)] for n in self.graph.neighbors(node)]



class igGraph(gBase): # igraph
    def __init__(self, graph, is_generator_bus=None):
        self.graph = graph
        self.N = len(graph)
        self.is_generator_bus = is_generator_bus
        if is_generator_bus is not None:
            self.gn_set   = np.flatnonzero(is_generator_bus)
            self.ld_set   = np.flatnonzero(~is_generator_bus)
        self._init_edges(graph.edges())

    @classmethod
    def construct(cls, N, edgelist, is_generator_bus=None):
        graph = nx.Graph()
        graph.add_nodes_from(range(N))
        graph.add_edges_from(edgelist)
        return cls(graph, is_generator_bus)

    def avg_degree(self):
        return np.mean(nx.degree(self.graph).values())

    def shortest_paths(self, dists):
        mx = np.inf * np.ones((self.N, self.N))
        nx.set_edge_attributes(self.graph, 'weight', dict(zip(self.graph.edges(), dists)))
        d = nx.shortest_path_length(self.graph, weight='weight')
        for i, v in d.iteritems():
            mx[i, v.keys()] = v.values()
        return mx

    def get_loadings(self, dists):
        cnts = np.zeros(self.N)
        args = {}
        nx.set_edge_attributes(self.graph, 'weight', dict(zip(self.graph.edges(), dists)))
        if self.is_generator_bus is not None:
            d=nx.centrality.betweenness_centrality_subset(self.graph, weight='weight', 
                sources=self.gn_set, targets=self.ld_set)
        else:
            d=nx.centrality.betweenness_centrality(self.graph, 
                weight='weight', normalized=False)
        ixs, vals = map(list, zip(*d.iteritems()))
        cnts[ixs] = vals
        return cnts        

    def neighbor_edges(self, node):
        return [self._edge_ixs[(node,n)] for n in self.graph.neighbors(node)]



"""
class CLMLoadings(object):
    # Class that computes loadings for CLM model
    @staticmethod
    def get_loadings(G, dists):
        import graph_tool.centrality
        vprop, _ =graph_tool.centrality.betweenness(G.graph, weight=dists)
        return vprop.a

    @classmethod
    def get_efficiency(cls, G, dists):
        mx = G.shortest_paths(dists)
        return (1./mx[G.notdiag]).mean()


class CLMLoadingsGensVsLoads(CLMLoadings):
    # Class that computes loadings for CLM while differentiating between
    # generator and load buses
    def __init__(self, G, is_generator_bus):
        self.is_generator_bus = is_generator_bus
        self.gn_set       = np.flatnonzero(is_generator_bus)
        self.ld_set   = np.flatnonzero(~is_generator_bus)
        super(CLMLoadingsGensVsLoads,self).__init__(G)

    def get_loadings(self, dists):
        # Convert from graph-tool to networkx format
        mx = graph_tool.spectral.adjacency(self.G, weight=dists).T
        nxG = nx.from_scipy_sparse_matrix(mx, create_using=nx.Graph())

        cnts = np.zeros(self.N)
        d=nx.centrality.betweenness_centrality_subset(nxG, 
            sources=self.gn_set, targets=self.ld_set, weight='weight')
        ixs, vals = map(list, zip(*d.iteritems()))
        cnts[ixs] = vals
        return cnts

    def get_efficiency(self, dists):
        mx = self._get_shortest_paths(dists)
        return (1./mx[self.is_generator_bus,:][:,~self.is_generator_bus]).mean()
"""

if args.NET == 'ieee300':
    from pypower import case300
    d=case300.case300()
    bus_idmap = { busid : ix for ix, busid in enumerate(d['bus'][:,0].astype('int')) }
    edgelist = [(bus_idmap[s], bus_idmap[e]) for s, e in d['branch'][:,0:2].astype('int')]
    cls = nxGraph if args.engine == 'networkx' else gtGraph
    G = cls.construct(len(bus_idmap), edgelist, d['bus'][:,1]==2)
    
else:
    import graph_tool.collection
    if args.NET == 'western':
        G = gtGraph(graph_tool.collection.data["power"])
    elif args.NET == 'karate':
        G = gtGraph(graph_tool.collection.data["karate"])
    else:
        raise Exception("Dont know how to load network %s"%args.NET)

    if args.engine == 'networkx':
        G=nxGraph(G.toNx())


print "# Running %s network: %s / %s" % (args.NET, str(G), G.__class__.__name__)

def print_row(net, alpha, r, pertid, t, eff, damage, runtime):
    row_format ="{:>10}{:>7}{:>7}{:>7}{:>6}{:>15}{:>15}{:>9}"
    if type(eff) is not str:
        eff = '%0.5f'% eff
    if type(damage) is not str:
        damage = '%0.5f'% damage
    if type(runtime) is not str:
        runtime = '%0.3f'% runtime
    print row_format.format(net, alpha, r, pertid, t, eff, damage, runtime)

print_row("Network", "Alpha", "R", "PertId", "t", "Eff", "Damage", "RunT")
np.set_printoptions(precision=3)

init_time = time.time()

diag = np.eye(G.N).astype('bool')
init_effs = np.ones(G.num_edges)
#init_effs = G.graph.new_edge_property("float") 
#init_effs.set_value(1.0)

#dists = G.graph.new_edge_property("float") 
#dists.a = 1.0/(init_effs.a + 1e-6)
NONZEROEPS = 1e-8
dists = 1.0/(init_effs + NONZEROEPS)

if args.rndseed is not None:
    np.random.seed(args.rndseed)

heterogeneity_q = 1 + (np.random.uniform(size=G.N)*2*args.r-args.r)
loadings = G.get_loadings(dists)
capacities = args.alpha * heterogeneity_q * loadings


init_efficiency = G.get_efficiency(dists)
run_time = time.time() - init_time

print_row(args.NET, args.alpha, args.r, -1, 0, init_efficiency, 0.0, run_time)

for ndx in range(args.num_perts):
    c_init_effs = init_effs.copy()

    cutnode = np.random.randint(G.N)
    #print "# Cutting edges", G.neighbor_edges(cutnode)
    c_init_effs[G.neighbor_edges(cutnode)] = 0.0
            
    c_effs = c_init_effs.copy()

    for t in range(args.num_iters):
        iter_init_time = time.time()
        dists = 1.0/(c_effs + NONZEROEPS)
        loadings = G.get_loadings(dists)
        print 'loadings:', np.sort(loadings)[-1:-20:-1]

        mult = np.ones(G.num_edges)
        ixs1 = loadings[G.edgesources]>capacities[G.edgesources]
        ixs2 = loadings[G.edgetargets]>capacities[G.edgetargets]
        mult[ixs1] = capacities[G.edgesources][ixs1]/loadings[G.edgesources][ixs1]
        mult[ixs2] = np.minimum(mult[ixs2], capacities[G.edgetargets][ixs2]/loadings[G.edgetargets][ixs2])
        c_effs = np.multiply(c_init_effs, mult)
        cur_efficiency = G.get_efficiency(dists)
        iter_time = time.time() - iter_init_time
        cur_damage = (init_efficiency-cur_efficiency)/init_efficiency
        print_row(args.NET, args.alpha, args.r, cutnode, t, cur_efficiency, cur_damage, iter_time)

print "# Total runtime %0.4f" % (time.time() - init_time)
