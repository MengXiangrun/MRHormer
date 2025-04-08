import torch_geometric as PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def to_homogeneous(node_emb_dict, edge_dict):
    # node number
    num_node_dict = dict()
    for node_type, node_emb in node_emb_dict.items():
        num_node = node_emb.shape[0]
        num_node_dict[node_type] = num_node

    # offset distance
    offset_dict = dict()
    offset = 0
    for node_type, num_node in num_node_dict.items():
        offset_dict[node_type] = offset
        offset += num_node

    # save node type mapping index range
    node_range_dict = dict()
    start_index = 0
    end_index = 0
    for node_type, num_node in num_node_dict.items():
        start_index = end_index
        end_index = end_index + num_node - 1
        node_range_dict[node_type] = (start_index, end_index)
        end_index = end_index + 1

    # stack node emb
    node_emb = torch.cat(list(node_emb_dict.values()), dim=0)

    # add offset distance to original heterogeneous subgraph edge index
    offset_edge_list = list()
    for edge_type, edge in edge_dict.items():
        source_type, target_type = edge_type[0], edge_type[-1]

        source_offset = offset_dict[source_type]
        target_offset = offset_dict[target_type]

        offset_edge = edge.clone()

        offset_edge[0] = offset_edge[0] + source_offset
        offset_edge[1] = offset_edge[1] + target_offset

        offset_edge_list.append(offset_edge)

    offset_edge = torch.cat(offset_edge_list, dim=1)

    return node_emb, node_range_dict, offset_edge


def to_heterogeneous_node_embedding(node_emb, node_range_dict):
    z_dict = dict()
    for node_type, node_range in node_range_dict.items():
        start_index, end_index = node_range
        z = node_emb[start_index:end_index + 1, :]
        z_dict[node_type] = z

    return z_dict


class Linear(torch.nn.Module):
    def __init__(self, out_dim, bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.linear = PyG.nn.Linear(in_channels=-1,
                                    out_channels=self.out_dim,
                                    weight_initializer='kaiming_uniform',
                                    bias=bias,
                                    bias_initializer='zero')
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


# CoB
class GCN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_hidden_dim)

        self.local = PyG.nn.GCN(in_channels=config.encoder_in_dim,
                                hidden_channels=config.encoder_hidden_dim,
                                out_channels=config.encoder_out_dim,
                                num_layers=config.num_local_layer)

    def forward(self, x_dict, edge_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_dict)
        h = self.local.forward(x=h, edge_index=edge)

        z_dict = to_heterogeneous_node_embedding(node_emb=h, node_range_dict=node_range_dict)

        return z_dict


def BGA_sparse_dropout(x: torch.Tensor, p: float, training: bool):
    x = x.coalesce()
    return torch.sparse_coo_tensor(x.indices(), F.dropout(x.values(), p=p, training=training),
                                   size=x.size())


class BGA_FFN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(BGA_FFN, self).__init__()
        self.lin1 = Linear(hidden_channels)
        self.lin2 = Linear(hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.dropout(x)
        x = F.relu(self.lin1(x))
        return x
        # return self.lin2(x)


class BGA_SparseFFN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(BGA_SparseFFN, self).__init__()
        self.lin1 = Linear(hidden_channels)
        self.lin2 = Linear(hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = BGA_sparse_dropout(x, 0.5, self.training)
        x = F.relu(self.lin1(x))
        return self.lin2(x) + x


class BGALayer_ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(BGALayer_ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.label_same_matrix = torch.load('analysis/label_same_matrix_citeseer.pt').float()

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # self.label_same_matrix = self.label_same_matrix.to(attn.device)
        # attn = attn * self.label_same_matrix * 2 + attn * (1-self.label_same_matrix)
        attn = self.dropout(F.softmax(attn, dim=-1))
        # attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn


class BGALayer_MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(BGALayer_MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = Linear(channels, bias=False)
        self.w_ks = Linear(channels, bias=False)
        self.w_vs = Linear(channels, bias=False)
        self.fc = Linear(channels, bias=False)

        self.attention = BGALayer_ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0)
        N_q = q.size(1)
        B_k = k.size(0)
        N_k = k.size(1)
        B_v = v.size(0)
        N_v = v.size(1)

        residual = q
        # x = self.dropout(q)

        # Pass through the pre-attention projection: B * N x (h*dv)
        # Separate different heads: B * N x h x dv
        q = self.w_qs(q).view(B_q, N_q, n_head, d_q)
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        # Transpose for attention dot product: B * h x N x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # For head axis broadcasting.
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: B x N x h x dv
        # Combine the last two dimensions to concatenate all the heads together: B x N x (h*dv)
        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1)
        q = self.fc(q)
        q = q + residual

        return q, attn


class BGALayer_FFN(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, channels, dropout=0.1):
        super(BGALayer_FFN, self).__init__()
        self.lin1 = Linear(channels, channels)  # position-wise
        self.lin2 = Linear(channels, channels)  # position-wise
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x) + residual

        return x


class BGALayer(nn.Module):
    def __init__(self, n_head, channels, use_patch_attn=True, dropout=0.1):
        super(BGALayer, self).__init__()
        self.node_norm = nn.LayerNorm(channels)
        self.node_transformer = BGALayer_MultiHeadAttention(n_head, channels, dropout)
        self.patch_norm = nn.LayerNorm(channels)
        self.patch_transformer = BGALayer_MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = BGALayer_FFN(channels, dropout)
        self.patch_ffn = BGALayer_FFN(channels, dropout)
        self.fuse_lin = Linear(2 * channels, channels)
        self.use_patch_attn = use_patch_attn
        self.attn = None

    def forward(self, x, patch, attn_mask=None, need_attn=False):
        x = self.node_norm(x)
        patch_x = x[patch]
        patch_x, attn = self.node_transformer(patch_x, patch_x, patch_x, attn_mask)
        patch_x = self.node_ffn(patch_x)
        if need_attn:
            self.attn = torch.zeros((x.shape[0], x.shape[0]))
            for i in tqdm(range(patch.shape[0])):
                p = patch[i].tolist()
                row = torch.tensor([p] * len(p)).T.flatten()
                col = torch.tensor(p * len(p))
                a = attn[i].mean(0).flatten().cpu()
                self.attn = self.attn.index_put((row, col), a)

            self.attn = self.attn[:-1][:, :-1].detach().cpu()

        if self.use_patch_attn:
            p = self.patch_norm(patch_x.mean(dim=1, keepdim=False)).unsqueeze(0)
            p, _ = self.patch_transformer(p, p, p)
            p = self.patch_ffn(p).permute(1, 0, 2)
            #
            p = p.repeat(1, patch.shape[1], 1)
            z = torch.cat([patch_x, p], dim=2)
            patch_x = F.relu(self.fuse_lin(z)) + patch_x

        x[patch] = patch_x

        return x


class BGA(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, use_patch_attn=True, dropout1=0.5, dropout2=0.1, need_attn=False):
        super(BGA, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = BGA_FFN(in_channels, hidden_channels)
        self.BGALayers = nn.ModuleList()
        for _ in range(0, layers):
            self.BGALayers.append(BGALayer(n_head, hidden_channels, use_patch_attn, dropout=dropout2))
        self.classifier = Linear(out_channels)
        self.attn = []

    def forward(self, x: torch.Tensor, patch: torch.Tensor, need_attn=False):
        patch_mask = (patch != self.num_nodes - 1).float().unsqueeze(-1)
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int()

        x = self.attribute_encoder(x)
        for i in range(0, self.layers):
            x = self.BGALayers[i](x, patch, attn_mask, need_attn)
            if need_attn:
                self.attn.append(self.BGALayers[i].attn)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class CoBFormer(torch.nn.Module):
    def __init__(self, config,
                 dropout1=0.5,
                 dropout2=0.1,
                 alpha=0.8,
                 tau=0.5,
                 use_patch_attn=True):
        super(CoBFormer, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.layers = config.num_global_self_layer
        self.n_head = config.num_global_head
        self.num_nodes = config.num_node
        self.dropout = nn.Dropout(dropout1)

        self.gcn = GCN(config)
        self.bga = BGA(self.num_nodes,
                       self.in_channels,
                       self.hidden_channels,
                       self.out_channels,
                       self.ayers,
                       self.n_head,
                       use_patch_attn,
                       dropout1,
                       dropout2)
        self.attn = None

    def forward(self, x: torch.Tensor, patch: torch.Tensor, edge_index: torch.Tensor, need_attn=False):
        z1 = self.gcn(x, edge_index)
        z2 = self.bga(x, patch, need_attn)
        if need_attn:
            self.attn = self.beyondformer.attn

        return z1, z2

    def loss(self, pred1, pred2, label, mask):
        l1 = F.cross_entropy(pred1[mask], label[mask])
        l2 = F.cross_entropy(pred2[mask], label[mask])
        pred1 *= self.tau
        pred2 *= self.tau
        l3 = F.cross_entropy(pred1[~mask], F.softmax(pred2, dim=1)[~mask])
        l4 = F.cross_entropy(pred2[~mask], F.softmax(pred1, dim=1)[~mask])
        loss = self.alpha * (l1 + l2) + (1 - self.alpha) * (l3 + l4)
        return loss
