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


# RHGNN[TKDE 2021]
# Heterogeneous Graph Representation Learning with Relation Awareness
# https://arxiv.org/abs/2105.11122
# from openhgnn.models import RHGNN
import dgl
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch
from dgl.ops import edge_softmax
import dgl.function as fn


class RHGNN_BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(RHGNN_BaseModel, self).__init__()

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


class RHGNN(RHGNN_BaseModel):
    def __init__(self, config):
        super(RHGNN, self).__init__()
        self.input_dim_dict = config.encoder_in_dim
        self.num_layers = config.num_local_layer
        self.hidden_dim = config.encoder_hidden_dim // config.num_local_head
        self.relation_input_dim = config.encoder_hidden_dim
        self.relation_hidden_dim = config.encoder_hidden_dim
        self.n_heads = config.num_local_head
        self.dropout = config.local_dropout_probability
        self.negative_slope = 0.2
        self.residual = True
        self.out_dim = config.encoder_out_dim
        self.norm = True

        # relation embedding dictionary
        self.relation_embedding = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(self.relation_input_dim, 1)) for etype in config.edge_type_list
        })

        # align the dimension of different types of nodes
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(self.input_dim_dict[ntype], self.hidden_dim * self.n_heads)
            for ntype in self.input_dim_dict
        })

        # each layer takes in the heterogeneous graph as input
        self.layers = nn.ModuleList()

        # for each relation_layer
        self.layers.append(R_HGNN_Layer(config=config))

        for _ in range(1, self.num_layers):
            self.layers.append(R_HGNN_Layer(config=config))

        # transformation matrix for target node representation under each relation
        self.node_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(self.n_heads, self.hidden_dim, self.hidden_dim))
            for etype in config.edge_type_list
        })

        # transformation matrix for relation representation
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(self.n_heads, self.relation_hidden_dim, self.hidden_dim))
            for etype in config.edge_type_list
        })

        # different relations fusing module
        self.relation_fusing = RelationFusing(node_hidden_dim=self.hidden_dim,
                                              relation_hidden_dim=self.relation_hidden_dim,
                                              num_heads=self.n_heads,
                                              dropout=self.dropout,
                                              negative_slope=self.negative_slope)
        self.out = nn.Linear(self.hidden_dim * self.n_heads, self.out_dim)  #### todo

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')

        for etype in self.relation_embedding:
            nn.init.xavier_normal_(self.relation_embedding[etype], gain=gain)
        for ntype in self.projection_layer:
            nn.init.xavier_normal_(self.projection_layer[ntype].weight, gain=gain)
        for etype in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[etype], gain=gain)
        for etype in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[etype], gain=gain)

    def forward(self, edge_dict, node_emb_dict,
                relation_target_node_features=None, relation_embedding: dict = None):
        hg = {}
        for edge_type, edge in edge_dict.items():
            edge_type = (edge_type[0], edge_type[0] + edge_type[-1], edge_type[-1])
            edge = (edge[0], edge[1])
            hg[edge_type] = edge
        hg = dgl.heterograph(data_dict=hg)
        for node_type, node_emb in node_emb_dict.items():
            hg.nodes[node_type].data['h'] = node_emb

        relation_target_node_features = {}
        for stype, etype, dtype in hg.canonical_etypes:
            relation_target_node_features[(stype, etype, dtype)] = hg.srcnodes[dtype].data.get('h').to(torch.float32)

        # target relation feature projection
        for stype, reltype, dtype in relation_target_node_features:
            relation_target_node_features[(stype, reltype, dtype)] = self.projection_layer[dtype](
                relation_target_node_features[(stype, reltype, dtype)])

        # each relation is associated with a specific type, if no semantic information is given,
        # then the one-hot representation of each relation is assign with trainable hidden representation
        if relation_embedding is None:
            relation_embedding = {}
            for etype in self.relation_embedding:
                relation_embedding[etype] = self.relation_embedding[etype].flatten()

        # graph convolution
        for layer in self.layers:
            relation_target_node_features, relation_embedding = layer(hg,
                                                                      relation_target_node_features,
                                                                      relation_embedding)

        relation_fusion_embedding_dict = {}
        # relation_target_node_features -> {(srctype, etype, dsttype): target_node_features}
        for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):
            relation_target_node_features_dict = {etype: relation_target_node_features[(stype, etype, dtype)]
                                                  for stype, etype, dtype in relation_target_node_features}
            etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
            dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]
            dst_relation_embeddings = [relation_embedding[etype] for etype in etypes]
            dst_node_feature_transformation_weight = [self.node_transformation_weight[etype] for etype in etypes]
            dst_relation_embedding_transformation_weight = [self.relation_transformation_weight[etype] for etype in
                                                            etypes]

            # Tensor, shape (heads_num * hidden_dim)
            dst_node_relation_fusion_feature = self.relation_fusing(dst_node_features,
                                                                    dst_relation_embeddings,
                                                                    dst_node_feature_transformation_weight,
                                                                    dst_relation_embedding_transformation_weight)

            relation_fusion_embedding_dict[dsttype] = self.out(dst_node_relation_fusion_feature)

        return relation_fusion_embedding_dict

    def inference(self, graph: dgl.DGLHeteroGraph, relation_target_node_features: dict, relation_embedding: dict = None,
                  device: str = 'cuda:0'):
        r"""
        mini-batch inference of final representation over all node types. Outer loop: Interate the layers, Inner loop: Interate the batches

        Parameters
        ----------
        graph: dgl.DGLHeteroGraph
            The whole relational graphs
        relation_target_node_features:  dict
            target node features under each relation, e.g {(srctype, etype, dsttype): features}
        relation_embedding: dict
            embedding for each relation, e.g {etype: feature} or None
        device: str
            device

        """

        with torch.no_grad():

            if relation_embedding is None:
                relation_embedding = {}
                for etype in self.relation_embedding:
                    relation_embedding[etype] = self.relation_embedding[etype].flatten()

            # interate over each layer
            for index, layer in enumerate(self.layers):
                # Tensor, features of all relation embeddings of the target nodes, store on cpu
                y = {
                    (stype, etype, dtype): torch.zeros(graph.number_of_nodes(dtype), self.hidden_dim * self.n_heads) for
                    stype, etype, dtype in graph.canonical_etypes}

                # full sample for each type of nodes
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    graph,
                    {ntype: torch.arange(graph.number_of_nodes(ntype)) for ntype in graph.ntypes},
                    sampler,
                    batch_size=1280,
                    shuffle=True,
                    drop_last=False,
                    num_workers=4)

                tqdm_dataloader = tqdm(dataloader, ncols=120)
                for batch, (input_nodes, output_nodes, blocks) in enumerate(tqdm_dataloader):
                    block = blocks[0].to(device)

                    # for relational graphs that only contain a single type of nodes, construct the input and output node dictionary
                    if len(set(blocks[0].ntypes)) == 1:
                        input_nodes = {blocks[0].ntypes[0]: input_nodes}
                        output_nodes = {blocks[0].ntypes[0]: output_nodes}

                    input_features = {(stype, etype, dtype): relation_target_node_features[(stype, etype, dtype)][
                        input_nodes[dtype]].to(device)
                                      for stype, etype, dtype in relation_target_node_features.keys()}

                    input_relation_features = relation_embedding

                    if index == 0:
                        # target relation feature projection for the first layer in the full batch inference
                        for stype, reltype, dtype in input_features:
                            input_features[(stype, reltype, dtype)] = self.projection_layer[dtype](
                                input_features[(stype, reltype, dtype)])
                    h, input_relation_features = layer(block, input_features, input_relation_features)
                    for stype, reltype, dtype in h.keys():
                        y[(stype, reltype, dtype)][output_nodes[dtype]] = h[(stype, reltype, dtype)].cpu()

                    tqdm_dataloader.set_description(f'inference for the {batch}-th batch in model {index}-th layer')

                # update the features of all the nodes (after the graph convolution) in the whole graph
                relation_target_node_features = y
                # relation embedding is updated after each layer
                relation_embedding = input_relation_features

            for stype, etype, dtype in relation_target_node_features:
                relation_target_node_features[(stype, etype, dtype)] = relation_target_node_features[
                    (stype, etype, dtype)].to(device)

            relation_fusion_embedding_dict = {}
            # relation_target_node_features -> {(srctype, etype, dsttype): target_node_features}
            for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):

                relation_target_node_features_dict = {etype: relation_target_node_features[(stype, etype, dtype)]
                                                      for stype, etype, dtype in relation_target_node_features}
                etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
                dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]
                dst_relation_embeddings = [relation_embedding[etype] for etype in etypes]
                dst_node_feature_transformation_weight = [self.node_transformation_weight[etype] for etype in etypes]
                dst_relation_embedding_transformation_weight = [self.relation_transformation_weight[etype] for etype in
                                                                etypes]

                # use mini-batch to avoid out of memory in inference
                relation_fusion_embedding = []
                index = 0
                batch_size = 2560
                while index < dst_node_features[0].shape[0]:
                    # Tensor, shape (heads_num * hidden_dim)
                    relation_fusion_embedding.append(self.relation_fusing(
                        [dst_node_feature[index: index + batch_size, :] for dst_node_feature in dst_node_features],
                        dst_relation_embeddings,
                        dst_node_feature_transformation_weight,
                        dst_relation_embedding_transformation_weight))
                    index += batch_size
                relation_fusion_embedding_dict[dsttype] = torch.cat(relation_fusion_embedding, dim=0)

            # relation_fusion_embedding_dict, {ntype: tensor -> (nodes, n_heads * hidden_dim)}
            # relation_target_node_features, {ntype: tensor -> (num_relations, nodes, n_heads * hidden_dim)}
            return relation_fusion_embedding_dict, relation_target_node_features


# hetetoConv
class HeteroGraphConv(nn.Module):
    def __init__(self, mods: dict):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)

    def forward(self, graph: dgl.DGLHeteroGraph, input_src: dict, input_dst: dict, relation_embedding: dict,
                node_transformation_weight: nn.ParameterDict, relation_transformation_weight: nn.ParameterDict):
        r"""
        call the forward function with each module.

        Parameters
        ----------
        graph: DGLHeteroGraph
            The Heterogeneous Graph.
        input_src: dict[tuple, Tensor]
            Input source node features {relation_type: features, }
        input_dst: dict[tuple, Tensor]
            Input destination node features {relation_type: features, }
        relation_embedding: dict[etype, Tensor]
            Input relation features {etype: feature}
        node_transformation_weight: nn.ParameterDict
            weights {ntype, (inp_dim, hidden_dim)}
        relation_transformation_weight: nn.ParameterDict
            weights {etype, (n_heads, 2 * hidden_dim)}

        Returns
        -------
        outputs: dict[tuple, Tensor]
            Output representations for every relation -> {(stype, etype, dtype): features}.
        """

        # find reverse relation dict
        reverse_relation_dict = {}
        for srctype, reltype, dsttype in list(input_src.keys()):
            for stype, etype, dtype in input_src:
                if stype == dsttype and dtype == srctype and etype != reltype:
                    reverse_relation_dict[reltype] = etype
                    break

        # dictionary, {(srctype, etype, dsttype): representations}
        outputs = dict()

        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            # for example, (author, writes, paper) relation, take author as src_nodes, take paper as dst_nodes
            dst_representation = self.mods[etype](rel_graph,
                                                  (input_src[(dtype, reverse_relation_dict[etype], stype)],
                                                   input_dst[(stype, etype, dtype)]),
                                                  node_transformation_weight[dtype],
                                                  node_transformation_weight[stype],
                                                  relation_embedding[etype],
                                                  relation_transformation_weight[etype])

            # dst_representation (dst_nodes, hid_dim)
            outputs[(stype, etype, dtype)] = dst_representation

        return outputs


# relation crossing
class RelationCrossing(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        r"""
        Relation crossing layer

        Parameters
        ----------
        in_feats : pair of ints
            input feature size
        out_feats : int
            output feature size
        num_heads : int
            number of heads in Multi-Head Attention
        dropout : float
            optional, dropout rate, defaults: 0.0
        negative_slope : float
            optional, negative slope rate, defaults: 0.2
        """
        super(RelationCrossing, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dsttype_node_features: torch.Tensor, relations_crossing_attention_weight: nn.Parameter):
        r"""
        Parameters
        ----------
        dsttype_node_features:
            a tensor of (dsttype_node_relations_num, num_dst_nodes, n_heads * hidden_dim)
        relations_crossing_attention_weight:
            Parameter the shape is (n_heads, hidden_dim)
        Returns:
        ----------
        output_features: Tensor

        """
        if len(dsttype_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dsttype_node_features = dsttype_node_features.squeeze(dim=0)
        else:
            # (dsttype_node_relations_num, num_dst_nodes, n_heads, hidden_dim)
            dsttype_node_features = dsttype_node_features.reshape(dsttype_node_features.shape[0], -1, self._num_heads,
                                                                  self._out_feats)
            # shape -> (dsttype_node_relations_num, dst_nodes_num, n_heads, 1),  (dsttype_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (n_heads, hidden_dim)
            dsttype_node_relation_attention = (dsttype_node_features * relations_crossing_attention_weight).sum(dim=-1,
                                                                                                                keepdim=True)
            dsttype_node_relation_attention = F.softmax(self.leaky_relu(dsttype_node_relation_attention), dim=0)
            # shape -> (dst_nodes_num, n_heads, hidden_dim),  (dsttype_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (dsttype_node_relations_num, dst_nodes_num, n_heads, 1)
            dsttype_node_features = (dsttype_node_features * dsttype_node_relation_attention).sum(dim=0)
            dsttype_node_features = self.dropout(dsttype_node_features)
            # shape -> (dst_nodes_num, n_heads * hidden_dim)
            dsttype_node_features = dsttype_node_features.reshape(-1, self._num_heads * self._out_feats)

        return dsttype_node_features


# relation fusing
class RelationFusing(nn.Module):
    def __init__(self, node_hidden_dim: int, relation_hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        super(RelationFusing, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dst_node_features: list, dst_relation_embeddings: list,
                dst_node_feature_transformation_weight: list,
                dst_relation_embedding_transformation_weight: list):
        if len(dst_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_features[0]
        else:
            # (num_dst_relations, nodes, n_heads, node_hidden_dim)
            dst_node_features = torch.stack(dst_node_features, dim=0).reshape(len(dst_node_features), -1,
                                                                              self.num_heads, self.node_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim)
            dst_relation_embeddings = torch.stack(dst_relation_embeddings, dim=0).reshape(len(dst_node_features),
                                                                                          self.num_heads,
                                                                                          self.relation_hidden_dim)
            # (num_dst_relations, n_heads, node_hidden_dim, node_hidden_dim)
            dst_node_feature_transformation_weight = torch.stack(dst_node_feature_transformation_weight, dim=0).reshape(
                len(dst_node_features), self.num_heads,
                self.node_hidden_dim, self.node_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim, relation_hidden_dim)
            dst_relation_embedding_transformation_weight = torch.stack(dst_relation_embedding_transformation_weight,
                                                                       dim=0).reshape(len(dst_node_features),
                                                                                      self.num_heads,
                                                                                      self.relation_hidden_dim,
                                                                                      self.node_hidden_dim)
            # shape (num_dst_relations, nodes, n_heads, hidden_dim)
            dst_node_features = torch.einsum('abcd,acde->abce', dst_node_features,
                                             dst_node_feature_transformation_weight)

            # shape (num_dst_relations, n_heads, hidden_dim)
            dst_relation_embeddings = torch.einsum('abc,abcd->abd', dst_relation_embeddings,
                                                   dst_relation_embedding_transformation_weight)

            # shape (num_dst_relations, nodes, n_heads, 1)
            attention_scores = (dst_node_features * dst_relation_embeddings.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
            attention_scores = F.softmax(self.leaky_relu(attention_scores), dim=0)
            # (nodes, n_heads, hidden_dim)
            dst_node_relation_fusion_feature = (dst_node_features * attention_scores).sum(dim=0)
            dst_node_relation_fusion_feature = self.dropout(dst_node_relation_fusion_feature)
            # (nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_relation_fusion_feature.reshape(-1,
                                                                                        self.num_heads * self.node_hidden_dim)

        return dst_node_relation_fusion_feature


# relationGraphConv
class RelationGraphConv(nn.Module):
    def __init__(self, in_feats: tuple, out_feats: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        super(RelationGraphConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.relu = nn.ReLU()

    def forward(self, graph: dgl.DGLHeteroGraph, feat: tuple, dst_node_transformation_weight: nn.Parameter,
                src_node_transformation_weight: nn.Parameter, relation_embedding: torch.Tensor,
                relation_transformation_weight: nn.Parameter):
        graph = graph.local_var()
        # Tensor, (N_src, input_src_dim)
        feat_src = self.dropout(feat[0])
        # Tensor, (N_dst, input_dst_dim)
        feat_dst = self.dropout(feat[1])
        # Tensor, (N_src, n_heads, hidden_dim) -> (N_src, input_src_dim) * (input_src_dim, n_heads * hidden_dim)
        feat_src = torch.matmul(feat_src, src_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        # Tensor, (N_dst, n_heads, hidden_dim) -> (N_dst, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        feat_dst = torch.matmul(feat_dst, dst_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        # Tensor, (n_heads, 2 * hidden_dim) -> (1, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        relation_attention_weight = torch.matmul(relation_embedding.unsqueeze(dim=0),
                                                 relation_transformation_weight).view(self._num_heads,
                                                                                      2 * self._out_feats)

        # first decompose the weight vector into [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j, This implementation is much efficient
        # Tensor, (N_dst, n_heads, 1),   (N_dst, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_dst = (feat_dst * relation_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
        # Tensor, (N_src, n_heads, 1),   (N_src, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_src = (feat_src * relation_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
        # (N_src, n_heads, hidden_dim), (N_src, n_heads, 1)
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        # (N_dst, n_heads, 1)
        graph.dstdata.update({'e_dst': e_dst})
        # compute edge attention, e_src and e_dst are a_src * Wh_src and a_dst * Wh_dst respectively.
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
        # shape (edges_num, heads, 1)
        e = self.leaky_relu(graph.edata.pop('e'))

        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)

        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'feat'))
        # (N_dst, n_heads * hidden_dim), reshape (N_dst, n_heads, hidden_dim)
        dst_features = graph.dstdata.pop('feat').reshape(-1, self._num_heads * self._out_feats)

        dst_features = self.relu(dst_features)

        return dst_features


class R_HGNN_Layer(nn.Module):
    def __init__(self, config):
        super(R_HGNN_Layer, self).__init__()
        self.input_dim = config.encoder_hidden_dim
        self.hidden_dim = config.encoder_hidden_dim
        self.relation_input_dim = config.encoder_hidden_dim
        self.relation_hidden_dim = config.encoder_hidden_dim
        self.n_heads = config.num_local_head
        self.dropout = config.local_dropout_probability
        self.negative_slope = 0.2
        self.residual = True
        self.norm = False

        # node transformation parameters of each type
        self.node_transformation_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(self.input_dim, self.n_heads * self.hidden_dim))
            for ntype in config.node_type_list
        })

        # relation transformation parameters of each type, used as attention queries
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(self.relation_input_dim, self.n_heads * 2 * self.hidden_dim))
            for etype in config.edge_type_list
        })

        # relation propagation layer of each relation
        self.relation_propagation_layer = nn.ModuleDict({
            etype: nn.Linear(self.relation_input_dim, self.n_heads * self.relation_hidden_dim)
            for etype in config.edge_type_list
        })

        # hetero conv modules, each RelationGraphConv deals with a single type of relation
        self.hetero_conv = HeteroGraphConv({
            etype: RelationGraphConv(in_feats=(self.input_dim, self.input_dim), out_feats=self.hidden_dim,
                                     num_heads=self.n_heads, dropout=self.dropout, negative_slope=self.negative_slope)
            for etype in config.edge_type_list
        })

        if self.residual:
            # residual connection
            self.res_fc = nn.ModuleDict()
            self.residual_weight = nn.ParameterDict()
            for ntype in config.node_type_list:
                self.res_fc[ntype] = nn.Linear(self.input_dim, self.n_heads * self.hidden_dim)
                self.residual_weight[ntype] = nn.Parameter(torch.randn(1))

        if self.norm:
            self.layer_norm = nn.ModuleDict(
                {ntype: nn.LayerNorm(self.n_heads * self.hidden_dim) for ntype in config.node_type_list})

        # relation type crossing attention trainable parameters
        self.relations_crossing_attention_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(self.n_heads, self.hidden_dim))
            for etype in config.node_type_list
        })
        # different relations crossing layer
        self.relations_crossing_layer = RelationCrossing(in_feats=self.n_heads * self.hidden_dim,
                                                         out_feats=self.hidden_dim,
                                                         num_heads=self.n_heads,
                                                         dropout=self.dropout,
                                                         negative_slope=self.negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[weight], gain=gain)
        for etype in self.relation_propagation_layer:
            nn.init.xavier_normal_(self.relation_propagation_layer[etype].weight, gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for weight in self.relations_crossing_attention_weight:
            nn.init.xavier_normal_(self.relations_crossing_attention_weight[weight], gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, relation_target_node_features: dict, relation_embedding: dict):
        # in each relation, target type of nodes has an embedding
        # dictionary of {(srctype, etypye, dsttype): target_node_features}
        input_src = relation_target_node_features

        if graph.is_block:
            input_dst = {}
            for srctype, etypye, dsttype in relation_target_node_features:
                input_dst[(srctype, etypye, dsttype)] = relation_target_node_features[(srctype, etypye, dsttype)][
                                                        :graph.number_of_dst_nodes(dsttype)]
        else:
            input_dst = relation_target_node_features

        # output_features, dict {(srctype, etypye, dsttype): target_node_features}
        output_features = self.hetero_conv(graph, input_src, input_dst, relation_embedding,
                                           self.node_transformation_weight, self.relation_transformation_weight)

        # residual connection for the target node
        if self.residual:
            for srctype, etype, dsttype in output_features:
                alpha = torch.sigmoid(self.residual_weight[dsttype])
                output_features[(srctype, etype, dsttype)] = output_features[(srctype, etype, dsttype)] * alpha + \
                                                             self.res_fc[dsttype](
                                                                 input_dst[(srctype, etype, dsttype)]) * (1 - alpha)

        output_features_dict = {}
        # different relations crossing layer
        for srctype, etype, dsttype in output_features:
            # (dsttype_node_relations_num, dst_nodes_num, n_heads * hidden_dim)
            dst_node_relations_features = torch.stack([output_features[(stype, reltype, dtype)]
                                                       for stype, reltype, dtype in output_features if
                                                       dtype == dsttype], dim=0)

            output_features_dict[(srctype, etype, dsttype)] = self.relations_crossing_layer(dst_node_relations_features,
                                                                                            self.relations_crossing_attention_weight[
                                                                                                etype])

        # layer norm for the output
        if self.norm:
            for srctype, etype, dsttype in output_features_dict:
                output_features_dict[(srctype, etype, dsttype)] = self.layer_norm[dsttype](
                    output_features_dict[(srctype, etype, dsttype)])

        relation_embedding_dict = {}
        for etype in relation_embedding:
            relation_embedding_dict[etype] = self.relation_propagation_layer[etype](relation_embedding[etype])

        # relation features after relation crossing layer, {(srctype, etype, dsttype): target_node_features}
        # relation embeddings after relation update, {etype: relation_embedding}
        return output_features_dict, relation_embedding_dict
