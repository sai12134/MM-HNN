import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
import torch_geometric.utils
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_scatter import scatter

from models.conv import MHNNConv
from models.mlp import MLP


class MHNN(nn.Module):
    def __init__(self, num_target, args, task_type='reg'):
        """ Molecular Hypergraph Neural Network (MHNN)
        (Shared parameters between all message passing layers)

        Args:
            num_target (int): number of output targets
            args (NamedTuple): global args
            task_type (str): 'reg' or 'cls'
        """
        super().__init__()

        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.mlp1_layers = args.MLP1_num_layers
        self.mlp2_layers = args.MLP2_num_layers
        self.mlp3_layers = args.MLP3_num_layers
        self.mlp4_layers = args.MLP4_num_layers
        self.nlayer = args.All_num_layers
        self.task_type = task_type

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.bond_encoder = nn.Embedding(6, args.MLP_hidden)

        self.conv = MHNNConv(args.MLP_hidden, mlp1_layers=self.mlp1_layers, mlp2_layers=self.mlp2_layers,
                             mlp3_layers=self.mlp3_layers, mlp4_layers=self.mlp4_layers, aggr=args.aggregate,
                             dropout=args.dropout, normalization=args.normalization)

        self.mlp_out = MLP(in_channels=args.MLP_hidden * 2,
                           hidden_channels=args.output_hidden * 2,
                           out_channels=num_target,
                           num_layers=args.output_num_layers,
                           dropout=args.dropout,
                           Normalization=args.normalization,
                           InputNorm=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):

        V, E = data.edge_index0, data.edge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.x.device)
        he_batch = e_batch[data.e_order > 2]

        x = self.atom_encoder(data.x)
        e = self.bond_encoder(data.edge_attr.squeeze(-1))

        for i in range(self.nlayer):
            x, e = self.conv(x, e, V, E)
            if i == self.nlayer - 1:
                # remove relu for the last layer
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        x = global_add_pool(x, data.batch)
        e = global_add_pool(e[data.e_order > 2], he_batch)
        out = self.mlp_out(torch.cat((x, e), -1))
        if self.task_type == 'cls':
            out = self.sigmoid(out)
        return out.squeeze(-1) if out.size(-1) == 1 else out


class MMHNN(nn.Module):
    """ MHNN with attention-based readout for both nodes and hyperedges.

    Instead of simple global sum pooling, uses learned attention weights
    to focus on important atoms and conjugated structures. This improves
    classification performance and better leverages hypergraph representations.
    (Shared parameters between all message passing layers)
    """
    def __init__(self, num_target, args, task_type='reg'):
        super().__init__()

        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.mlp1_layers = args.MLP1_num_layers
        self.mlp2_layers = args.MLP2_num_layers
        self.mlp3_layers = args.MLP3_num_layers
        self.mlp4_layers = args.MLP4_num_layers
        self.nlayer = args.All_num_layers
        self.task_type = task_type

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.bond_encoder = nn.Embedding(6, args.MLP_hidden)

        self.conv = MHNNConv(args.MLP_hidden, mlp1_layers=self.mlp1_layers, mlp2_layers=self.mlp2_layers,
                             mlp3_layers=self.mlp3_layers, mlp4_layers=self.mlp4_layers, aggr=args.aggregate,
                             dropout=args.dropout, normalization=args.normalization)

        # attention readout for nodes
        self.node_attn = nn.Linear(args.MLP_hidden, 1)

        self.mlp_out = MLP(in_channels=args.MLP_hidden * 2,
                           hidden_channels=args.output_hidden * 2,
                           out_channels=num_target,
                           num_layers=args.output_num_layers,
                           dropout=args.dropout,
                           Normalization=args.normalization,
                           InputNorm=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        V, E = data.edge_index0, data.edge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.x.device)
        he_batch = e_batch[data.e_order > 2]

        x = self.atom_encoder(data.x)
        e = self.bond_encoder(data.edge_attr.squeeze(-1))

        for i in range(self.nlayer):
            x, e = self.conv(x, e, V, E)
            if i == self.nlayer - 1:
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        # node attention pooling
        node_attn_score = self.node_attn(x)
        node_attn_weight = torch_geometric.utils.softmax(node_attn_score, data.batch, dim=0)
        x_pool = scatter(node_attn_weight * x, data.batch, dim=0, reduce='sum')

        # hyperedge global sum pooling
        num_graphs = data.batch.max().item() + 1
        e_pool = scatter(e[data.e_order > 2], he_batch, dim=0, reduce='sum', dim_size=num_graphs)

        out = self.mlp_out(torch.cat((x_pool, e_pool), -1))
        if self.task_type == 'cls':
            out = self.sigmoid(out)
        return out.squeeze(-1) if out.size(-1) == 1 else out