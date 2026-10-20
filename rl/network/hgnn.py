import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F

class AddAndNorm(nn.Module):
    """Normalize after the residual connection."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddAndNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.rmsnorm = nn.RMSNorm(normalized_shape)

    def forward(self, X, Y):
        return self.rmsnorm(self.dropout(Y) + X)

class Ffn(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(Ffn, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.elu = nn.ELU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.elu(self.dense1(X)))
    

# Machine node embedding
class GATedge(nn.Module):
    """
    Machine node embedding
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_head,
        dropout=0.0,
        negative_slope=0.2,
        residual=False,
    ):
        """
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        """
        super(GATedge, self).__init__()
        self._num_heads = num_head
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats
        
        self.addnorm1 = AddAndNorm(out_feats, 0)
        self.ffn = Ffn(out_feats, 2 * out_feats, out_feats)
        self.addnorm2 = AddAndNorm(out_feats, dropout)

        # Embedding layers for oper (src), machine (dst), and processing time + operation parameter (edge)
        self.fc_src = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_dst = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_edge = nn.Linear(2, out_feats, bias=False)

        # Attention parameters: attn_l: oper | attn_r: machine | attn_e: O-M pair
        self.attn_l = nn.Parameter(torch.rand(size=(1, 1, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, 1, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, 1, out_feats), dtype=torch.float))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.residual = residual
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        
        nn.init.xavier_normal_(self.ffn.dense1.weight, gain=gain)
        nn.init.xavier_normal_(self.ffn.dense2.weight, gain=gain)

    def attention(self, ope_ma_adj_batch, feat_src, feat_dst, feat_edge, attn_l, attn_r, attn_e):
        # Calculate attention coefficients
        # feat_src: (B, num_opers, d)
        # self.attn_l: (1, 1, d)
        
        # el: (B, num_opers, 1), mu_ij
        # er: (B, num_machines, 1), nu_k
        # ee: (B, num_opers, nums_machines, 1), lamta_ijk
        el = (feat_src * attn_l).sum(dim=-1).unsqueeze(-1)  
        er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(-1)
        ee = (feat_edge * attn_e.unsqueeze(1)).sum(dim=-1).unsqueeze(-1)

        # ope_ma_adj_batch：(len(batch_size), num_opers, num_machines)
        el_add_ee = ope_ma_adj_batch.unsqueeze(-1) * el.unsqueeze(-2) + ee
        el_add_ee_add_er = el_add_ee + ope_ma_adj_batch.unsqueeze(-1) * er.unsqueeze(-3)
        eijk = self.leaky_relu(el_add_ee_add_er)
        ekk = self.leaky_relu(er + er)

        # Normalize attention coefficients
        mask = torch.cat(
            (
                ope_ma_adj_batch.unsqueeze(-1) == 1,
                torch.full(
                    size=(ope_ma_adj_batch.size(0), 1, ope_ma_adj_batch.size(2), 1),
                    dtype=torch.bool,
                    fill_value=True,
                ),
            ),
            dim=-3,
        )

        # e: (batch_size, num_opers + 1, num_machines, 1)
        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)
        expanded_mask = mask.unsqueeze(0).expand(e.shape)
        e[~expanded_mask] = float("-inf")

        alpha = F.softmax(e.squeeze(-1), dim=-2)
        alpha = self.dropout(alpha)
        alpha_ijk = alpha[..., :-1, :]
        # alpha_kk: (num_head, batch_size, 1, num_machines)
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)

        # Calculate an return machine embedding
        # (batch_size，num_opers, num_mas，d)
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)
        b = torch.sum(Wmu_ijk * alpha_ijk.unsqueeze(-1), dim=-3)
        c = feat_dst * alpha_kk.squeeze(-2).unsqueeze(-1)
        return b, c

    def transpose_node(self, X, num_head):
        # (b，unique_dim(oper/mas)，num_heads，d / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_head, -1)
        return X.permute(2, 0, 1, 3)

    def transpose_edge(self, X, num_head):
        # (b，num_opers, num_mas，num_heads，d / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], num_head, -1)
        return X.permute(3, 0, 1, 2, 4)
    
    def transpose_attn(self, X, num_head):
        # (b，unique_dim(oper/mas)，num_heads，d / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_head, -1)
        return X.permute(2, 0, 1, 3)

    def inverse_transpose(self, X):
        X = X.permute(1, 2, 0, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    
    def forward(self, ope_ma_adj_batch, feat):
        # Project operations and machines separately; concatenate processing time and operation parameter into a 2-D edge input, then project it to out_feats.

        # feat_src: (b, num_opers, d)
        # feat_dst: (b, num_machines, d)
        # feat_edge: (b, num_opers, num_machines, out_feats)
        feat_src = self.fc_src(self.dropout(feat[0]))
        feat_dst = self.fc_dst(self.dropout(feat[1]))
        feat_edge = self.fc_edge(torch.cat([feat[2].unsqueeze(-1), feat[3].unsqueeze(-1)], dim=-1))
        split_feat_src = self.transpose_node(feat_src, self._num_heads)
        split_feat_dst = self.transpose_node(feat_dst, self._num_heads)
        split_feat_edge = self.transpose_edge(feat_edge, self._num_heads)  
        split_attn_l = self.transpose_attn(self.attn_l, self._num_heads)
        split_attn_r = self.transpose_attn(self.attn_r, self._num_heads)
        split_attn_e = self.transpose_attn(self.attn_e, self._num_heads)
        
        b, c = self.attention(ope_ma_adj_batch, split_feat_src, split_feat_dst, split_feat_edge, split_attn_l, split_attn_r, split_attn_e)

        b = self.inverse_transpose(b)
        c = self.inverse_transpose(c)
        
        # nu_k_prime: shape(batch_size, num_mas, d)
        if self.residual:
            Y = self.addnorm1(feat_dst, (b + c))
            Y = self.addnorm2(Y, self.ffn(Y))
            nu_k_prime = torch.sigmoid(Y)
        else:
            nu_k_prime = torch.sigmoid(b + c)
        
        return nu_k_prime


# Part of the operation feature embedding
class MLP_theta_i(nn.Module):
    """
    Part of operation node embedding
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_dim,
        num_head,
        dropout=0.0,
        negative_slope=0.2,
        residual=False,
    ):
        """
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        """
        super(MLP_theta_i, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout), # Add dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout), # Add dropout
            nn.Linear(hidden_dim, self._out_feats),
        )

    def forward(self, feat, adj):
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        



        b = torch.bmm(adj.float(), feat) # Save memory
        c = self.project(b)
        return c


# Operation node embedding
class MLPs(nn.Module):
    """
    MLPs in operation node embedding
    """

    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout, residual=False):
        """

        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        """
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.gnn_layers = nn.ModuleList()
        self.residual = residual

        # A total of 4 MLPs and MLP_0 (self.mlp_theta_0) aggregates information from other MLPs
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(
                MLP_theta_i(
                    self.in_sizes_ope[i],
                    self.out_size_ope,
                    self.hidden_size_ope,
                    self.num_head,
                    dropout,
                    residual=self.residual,
                )
            )
        
        self.mlp_theta_0 = nn.Sequential(
            nn.ELU(),
            nn.Dropout(dropout), # Add dropout
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Dropout(dropout), # Add dropout
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Dropout(dropout), # Add dropout
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        """
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        """
        h = (feats[1], feats[0], feats[0], feats[0])


        self_adj = torch.eye(feats[0].size(-2), dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch)








        adj = (
            ope_ma_adj_batch,
            ope_pre_adj_batch,
            ope_sub_adj_batch,
            self_adj,
        )


        mlp_embeddings = []
        for i in range(len(adj)):
            mlp_embeddings.append(self.gnn_layers[i](h[i], adj[i]))


        mu_ij_prime = self.mlp_theta_0(torch.cat(mlp_embeddings, dim=-1))
        return mu_ij_prime
