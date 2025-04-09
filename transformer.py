import torch_geometric
import torch
import math


class Linear(torch.nn.Module):
    def __init__(self, out_dim, bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.linear = torch_geometric.nn.Linear(in_channels=-1,
                                                out_channels=self.out_dim,
                                                weight_initializer='kaiming_uniform',
                                                bias=bias,
                                                bias_initializer='zeros')
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class MHA(torch.nn.Module):
    def __init__(self,
                 emb_dim,
                 num_head,
                 dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.dropout = dropout
        self.head_dim = self.emb_dim // self.num_head
        assert self.head_dim * self.num_head == self.emb_dim

        self.q_linear = Linear(self.emb_dim, bias=False)
        self.k_linear = Linear(self.emb_dim, bias=False)
        self.v_linear = Linear(self.emb_dim, bias=False)
        self.out_linear = Linear(self.emb_dim, bias=True)

    def forward(self,
                source_emb,
                target_emb,
                attention_mask=None):
        if source_emb.ndim == 2:
            num_source_token, source_emb_dim = source_emb.shape
            source_emb = source_emb.view(1, num_source_token, source_emb_dim)
        if target_emb.ndim == 2:
            num_target_token, target_emb_dim = target_emb.shape
            target_emb = target_emb.view(1, num_target_token, target_emb_dim)

        batch_size, num_source_token, source_emb_dim = source_emb.shape
        batch_size, num_target_token, target_emb_dim = target_emb.shape

        num_head = self.num_head
        head_dim = self.head_dim

        q_emb = target_emb.transpose(1, 0)
        q_emb = self.q_linear(q_emb)
        q_emb = q_emb.view(num_target_token, batch_size, num_head, head_dim)
        q_emb = q_emb.view(num_target_token, batch_size * num_head, head_dim)
        q_emb = q_emb.transpose(0, 1)

        k_emb = source_emb.transpose(1, 0)
        num_source_token, batch_size, source_emb_dim = k_emb.shape
        k_emb = self.k_linear(k_emb)
        k_emb = k_emb.view(num_source_token, batch_size, num_head, head_dim)
        k_emb = k_emb.view(num_source_token, batch_size * num_head, head_dim)
        k_emb = k_emb.transpose(0, 1)
        k_emb_transpose = k_emb.transpose(-2, -1)

        v_emb = source_emb.transpose(1, 0)
        num_source_token, batch_size, source_emb_dim = v_emb.shape
        v_emb = self.v_linear(v_emb)
        v_emb = v_emb.view(num_source_token, batch_size, num_head, head_dim)
        v_emb = v_emb.view(num_source_token, batch_size * num_head, head_dim)
        v_emb = v_emb.transpose(0, 1)

        # attention_mask
        # (batch_size * num_head, num_target_token, num_source_token)
        if attention_mask is not None:
            q_emb = q_emb / math.sqrt(float(head_dim))  # scaling
            attention = torch.baddbmm(attention_mask, q_emb, k_emb_transpose)
        else:
            q_emb = q_emb / math.sqrt(float(head_dim))  # scaling
            attention = torch.bmm(q_emb, k_emb.transpose(-2, -1))

        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = torch.nn.functional.dropout(input=attention, p=self.dropout, training=self.training)

        out_emb = torch.bmm(attention, v_emb)
        out_emb = out_emb.transpose(dim0=0, dim1=1).contiguous()
        out_emb = out_emb.view(num_target_token, batch_size, num_head, head_dim)
        out_emb = out_emb.view(num_target_token, batch_size, target_emb_dim)
        out_emb = out_emb.transpose(dim0=0, dim1=1).contiguous()
        out_emb = self.out_linear(out_emb)

        attention = attention.view(batch_size, num_head, num_target_token, num_source_token)

        if batch_size == 1:
            mean_attention = attention.mean(dim=1)
            attention = mean_attention.squeeze(0)
            out_emb = out_emb.squeeze(0)
        return out_emb, attention


class FFN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, dropout_p=0.1):
        super().__init__()
        self.linear1 = Linear(hidden_dim)
        self.linear2 = Linear(input_dim)  # Output dimension should match input
        self.activation = torch.nn.ReLU()  # Or any other activation function you prefer
        self.dropout = torch.nn.Dropout(dropout_p)
        self.dropout3 = torch.nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class SimpleTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layer = config.num_global_self_layer

        self.source_layers = torch.nn.ModuleList()
        for _ in range(self.num_layer ):
            source_mha = MHA(emb_dim=config.encoder_hidden_dim, num_head=config.num_global_head)
            self.source_layers.append(source_mha)
        self.source_ffn = FFN(input_dim=config.encoder_hidden_dim)
        self.source_norm1 = torch.nn.LayerNorm(config.encoder_hidden_dim, eps=1e-5, bias=True)
        self.source_norm2 = torch.nn.LayerNorm(config.encoder_hidden_dim, eps=1e-5, bias=True)

        self.target_layers = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            target_mha = MHA(emb_dim=config.encoder_hidden_dim, num_head=config.num_global_head)
            self.target_layers.append(target_mha)
        self.target_ffn = FFN(input_dim=config.encoder_hidden_dim)
        self.target_norm1 = torch.nn.LayerNorm(config.encoder_hidden_dim, eps=1e-5, bias=True)
        self.target_norm2 = torch.nn.LayerNorm(config.encoder_hidden_dim, eps=1e-5, bias=True)

        self.cross_layers = torch.nn.ModuleList()
        for _ in range(self.num_layer ):
            cross = MHA(emb_dim=config.encoder_hidden_dim, num_head=config.num_global_head)
            self.cross_layers.append(cross)
        self.cross_ffn = FFN(input_dim=config.encoder_hidden_dim)
        self.cross_norm1 = torch.nn.LayerNorm(config.encoder_hidden_dim, eps=1e-5, bias=True)
        self.cross_norm2 = torch.nn.LayerNorm(config.encoder_hidden_dim, eps=1e-5, bias=True)

    def forward(self, source_emb, target_emb):
        for layer_index in range(self.num_layer):
            residual = source_emb.clone()
            source_emb, _ = self.source_layers[layer_index].forward(source_emb=source_emb, target_emb=source_emb)
            source_emb += residual
            source_emb = self.source_norm1(source_emb)

            residual = source_emb.clone()
            source_emb = self.source_ffn(source_emb)
            source_emb += residual
            source_emb = self.source_norm2(source_emb)

        for layer_index in range(self.num_layer):
            # residual = target_emb.clone()
            # target_emb, _ = self.target_layers[layer_index].forward(source_emb=target_emb, target_emb=target_emb)
            # target_emb += residual
            # target_emb = self.target_norm1(target_emb)
            #
            # residual = target_emb.clone()
            # target_emb = self.target_ffn(target_emb)
            # target_emb += residual
            # target_emb = self.target_norm2(target_emb)
            #
            # for layer_index in range(self.num_layer):
            residual = target_emb.clone()
            target_emb, attention = self.cross_layers[layer_index].forward(source_emb=source_emb, target_emb=target_emb)
            target_emb += residual
            target_emb = self.cross_norm1(target_emb)

            residual = target_emb.clone()
            target_emb = self.cross_ffn(target_emb)
            target_emb += residual
            target_emb = self.cross_norm2(target_emb)

        return target_emb, attention

class VanillaTransformer(torch.nn.Transformer):
    def __init__(self):
        super().__init__()

class DecoderTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layer = config.num_global_self_layer

        self.cross_layers = torch.nn.ModuleList()
        for _ in range(self.num_layer ):
            cross = MHA(emb_dim=config.encoder_hidden_dim, num_head=config.num_global_head)
            self.cross_layers.append(cross)
        self.cross_ffn = FFN(input_dim=config.encoder_hidden_dim)
        self.cross_norm1 = torch.nn.LayerNorm(config.encoder_hidden_dim, eps=1e-5, bias=True)
        self.cross_norm2 = torch.nn.LayerNorm(config.encoder_hidden_dim, eps=1e-5, bias=True)

    def forward(self, source_emb, target_emb):
        for layer_index in range(self.num_layer):
            residual = target_emb.clone()
            target_emb, attention = self.cross_layers[layer_index].forward(source_emb=source_emb, target_emb=target_emb)
            target_emb += residual
            target_emb = self.cross_norm1(target_emb)

            residual = target_emb.clone()
            target_emb = self.cross_ffn(target_emb)
            target_emb += residual
            target_emb = self.cross_norm2(target_emb)

        return target_emb, attention
