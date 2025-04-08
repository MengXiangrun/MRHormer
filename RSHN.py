# use DGL environment
import os
from dgl.data.utils import load_graphs, save_graphs
import dgl
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from abc import ABCMeta
import torch.nn as nn


# Relation-based Heterogeneous GNN Baselines############################################################################
# RSHN[ICDM 2019]
# Relation Structure-Aware Heterogeneous Graph Neural Network
# https://ieeexplore.ieee.org/abstract/document/8970828
# from openhgnn.models import RSHN
class coarsened_line_graph():

    def __init__(self, rw_len, batch_size, n_dataset, symmetric=True):
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.n_dataset = n_dataset
        self.symmetric = symmetric  # which means the original graph had inverse edges

        return

    def get_cl_graph(self, hg):
        fname = './openhgnn/output/RSHN/{}_cl_graoh_{}_{}.bin'.format(
            self.n_dataset, self.rw_len, self.batch_size)
        if os.patorch.exists(fname):
            g, _ = load_graphs(fname)
            return g[0]
        else:
            g = self.build_cl_graph(hg)
            save_graphs(fname, g)
            return g

    def init_cl_graph(self, cl_graph):
        cl_graph = give_one_hot_feats(cl_graph, 'h')

        cl_graph = dgl.remove_self_loop(cl_graph)
        edge_attr = cl_graph.edata['w'].type(torch.FloatTensor).to(cl_graph.device)

        row, col = cl_graph.edges()
        for i in range(cl_graph.num_nodes()):
            mask = torch.eq(row, i)
            edge_attr[mask] = torch.nn.functional.normalize(edge_attr[mask], p=2, dim=0)

        # add_self_loop, set 1 as edge feature
        cl_graph = dgl.add_self_loop(cl_graph)
        edge_attr = torch.cat([edge_attr, torch.ones(cl_graph.num_nodes(), device=edge_attr.device)], dim=0)
        cl_graph.edata['w'] = edge_attr
        return cl_graph

    def build_cl_graph(self, hg):
        if not hg.is_homogeneous:
            self.num_edge_type = len(hg.etypes)
            g = dgl.to_homogeneous(hg).to('cpu')

        traces = self.random_walks(g)
        edge_batch = self.rw_map_edge_type(g, traces)
        cl_graph = self.edge2graph(edge_batch)
        return cl_graph

    def random_walks(self, g):
        source_nodes = torch.randint(0, g.number_of_nodes(), (self.batch_size,))
        traces, _ = dgl.sampling.random_walk(g, source_nodes, length=self.rw_len - 1)
        return traces

    def rw_map_edge_type(self, g, traces):
        edge_type = g.edata[dgl.ETYPE].long()
        edge_batch = []
        first_flag = True
        for t in traces:
            u = t[:-1]
            v = t[1:]
            edge_path = edge_type[g.edge_ids(u, v)].unsqueeze(0)
            if first_flag == True:
                edge_batch = edge_path
                first_flag = False
            else:
                edge_batch = torch.cat((edge_batch, edge_path), dim=0)
        return edge_batch

    def edge2graph(self, edge_batch):

        u = edge_batch[:, :-1].reshape(-1)
        v = edge_batch[:, 1:].reshape(-1)
        if self.symmetric:
            tmp = u
            u = torch.cat((u, v), dim=0)
            v = torch.cat((v, tmp), dim=0)

        g = dgl.graph((u, v))
        sg = dgl.to_simple(g, return_counts='w')
        return sg


def give_one_hot_feats(g, ntype='h'):
    # if the nodes are featureless, the input feature is then the node id.
    num_nodes = g.num_nodes()
    # g.ndata[ntype] = torch.arange(num_nodes, dtype=torch.float32, device=g.device)
    g.ndata[ntype] = torch.eye(num_nodes).to(g.device)
    return g


class RSHN_BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(RSHN_BaseModel, self).__init__()

    def forward(self, *args):
        raise NotImplementedError

    def extra_loss(self):
        raise NotImplementedError

    def h2dict(self, h, hdict):
        pre = 0
        out_dict = {}
        for i, value in hdict.items():
            out_dict[i] = h[pre:value.shape[0] + pre]
            pre += value.shape[0]
        return out_dict

    def get_emb(self):
        raise NotImplementedError


class AGNNConv(nn.Module):
    def __init__(self,
                 eps=0.,
                 train_eps=False,
                 learn_beta=True):
        super(AGNNConv, self).__init__()
        self.initial_eps = eps
        if learn_beta:
            self.beta = nn.Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.Tensor(1))
        self.learn_beta = learn_beta
        if train_eps:
            self.eps = torch.nn.Parameter(torch.ones([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)
        if self.learn_beta:
            self.beta.data.fill_(1)

    def forward(self, graph, feat, edge_weight):

        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata['norm_h'] = F.normalize(feat_src, p=2, dim=-1)

            e = self.beta * edge_weight
            # graph.edata['p'] = e
            graph.edata['p'] = edge_softmax(graph, e, norm_by='src')
            graph.update_all(fn.u_mul_e('norm_h', 'p', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata.pop('h')
            rst = (1 + self.eps) * feat + rst
            return rst


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats, dropout,
                 activation=None,
                 ):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.weight1 = nn.Parameter(torch.Tensor(in_feats, out_feats))
        # self.weight2 = nn.Parameter(torch.Tensor(in_feats, out_feats))

        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1)
        # nn.init.xavier_uniform_(self.weight2)

    def forward(self, hg, feat, edge_weight=None):
        with hg.local_scope():
            outputs = {}
            norm = {}
            aggregate_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                # assert edge_weight.shape[0] == graph.number_of_edges()
                hg.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            for e in hg.canonical_etypes:
                if e[0] == e[1]:
                    hg = dgl.remove_self_loop(hg, etype=e)
            feat_src, feat_dst = expand_as_pair(feat, hg)

            # aggregate first then mult W
            hg.srcdata['h'] = feat_src
            for e in hg.canonical_etypes:
                stype, etype, dtype = e
                sub_graph = hg[stype, etype, dtype]
                sub_graph.update_all(aggregate_fn, fn.sum(msg='m', out='out'))
                temp = hg.ndata['out'].pop(dtype)
                degs = sub_graph.in_degrees().float().clamp(min=1)
                if isinstance(temp, dict):
                    temp = temp[dtype]
                if outputs.get(dtype) is None:
                    outputs[dtype] = temp
                    norm[dtype] = degs
                else:
                    outputs[dtype].add_(temp)
                    norm[dtype].add_(degs)

            def _apply(ntype, h, norm):
                h = torch.matmul(h + feat[ntype], self.weight1)

                if self.activation:
                    h = self.activation(h)
                return self.dropout(h)

            return {ntype: _apply(ntype, h, norm) for ntype, h in outputs.items()}


class RSHN(RSHN_BaseModel):
    def __init__(self, config):
        super(RSHN, self).__init__()
        # map the edge feature
        self.num_node_layer = config.num_local_layer

        dim = config.encoder_hidden_dim
        dropout = config.local_dropout_probability
        out_dim = config.encoder_out_dim

        self.AGNNConvs = nn.ModuleList()
        for i in range(config.num_local_layer):
            self.AGNNConvs.append(AGNNConv())

        self.GraphConvs = nn.ModuleList()
        for i in range(config.num_local_layer):
            self.GraphConvs.append(GraphConv(in_feats=dim, out_feats=dim, dropout=dropout, activation=torch.tanh))

        self.linear = nn.Linear(in_features=dim, out_features=out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.init_para()

    def init_para(self):
        return

    def forward(self, edge_dict, node_emb_dict):
        hg = {}
        for edge_type, edge in edge_dict.items():
            edge_type = (edge_type[0], edge_type[0] + edge_type[-1], edge_type[-1])
            edge = (edge[0], edge[1])
            hg[edge_type] = edge

        hg = dgl.heterograph(data_dict=hg)

        h = self.cl_graph.ndata['h']
        h_e = self.cl_graph.edata['w']
        for layer in self.AGNNConvs:
            h = torch.relu(layer(self.cl_graph, h, h_e))
            h = self.dropout(h)

        h = self.linear_e1(h)
        edge_weight = {}
        for i, e in enumerate(hg.canonical_etypes):
            edge_weight[e] = h[i].expand(hg.num_edges(e), -1)
        if hasattr(hg, 'ntypes'):
            for layer in self.GraphConvs:
                node_emb_dict = layer(hg, node_emb_dict, edge_weight)
        else:
            # minibatch training
            pass
        for n in node_emb_dict:
            # n_feats[n] = self.dropout(self.linear(n_feats[n]))
            node_emb_dict[n] = self.linear(node_emb_dict[n])
        return node_emb_dict