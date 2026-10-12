import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from .model import BaseRGCN


class GNN(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "rgcn":
            return RGCNBlockLayer(
                self.h_dim,
                self.h_dim,
                self.num_rels,
                self.num_bases,
                activation=act,
                self_loop=self.self_loop,
                dropout=self.dropout,
                skip_connect=sc,
            ).cuda()
        elif self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb).cuda()
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "rgcn":
            g.ndata['h'] = g.ndata['feat']
            prev_h = []
            for _, layer in enumerate(self.layers):
                prev_h = layer(g, prev_h)
            return g.ndata.pop('h')
        elif self.encoder_name == "uvrgcn":
            g.ndata['h'] = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                layer(g, [], init_rel_emb)
            return g.ndata.pop('h')
        else:
            raise NotImplementedError


'''

self.rgcn = GNN(num_ents,
                     h_dim == 200,  # 是num_bases和num_basis的整数倍
                     h_dim == 200,
                     num_rels,
                     num_bases == 100,
                     num_basis == 100,
                     num_hidden_layers == 2,
                     dropout == 0.2,
                     self_loop == True,
                     skip_connect == False,
                     encoder_name == 'rgcn',
                     self.opn == 'sub',
                     self.emb_rel,
                     use_cuda == True,
                     analysis == False)
                     
'''
