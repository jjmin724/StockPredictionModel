"""
Graph augmentation utilities for GraphCL-style contrastive learning
-------------------------------------------------------------------
author : you
"""
from __future__ import annotations
import random
from copy import deepcopy
import torch
from torch_geometric.data import Data


class GraphAugmentor:
    """Collection of simple graph augmentations."""

    def __init__(self, strength: float = 0.2):
        self.p = strength

    # ---------- public API ----------
    def __call__(self, g: Data) -> Data:
        # Compose two random augmentations à la GraphCL
        funcs = [self.node_drop, self.edge_perturb, self.attr_mask]
        aug1, aug2 = random.sample(funcs, 2)
        return aug2(aug1(deepcopy(g)))

    # ---------- individual transforms ----------
    def node_drop(self, g: Data) -> Data:
        """Randomly drop nodes by probability p."""
        num_nodes = g.num_nodes
        mask = torch.rand(num_nodes) > self.p
        g.x = g.x[mask]
        # rebuild edge_index
        node_map = torch.arange(num_nodes)[mask]
        idx_map = -torch.ones(num_nodes, dtype=torch.long)
        idx_map[mask] = torch.arange(mask.sum())
        src, dst = g.edge_index
        keep = mask[src] & mask[dst]
        src, dst = src[keep], dst[keep]
        g.edge_index = torch.vstack([idx_map[src], idx_map[dst]])
        return g

    def edge_perturb(self, g: Data) -> Data:
        """Randomly remove and add edges."""
        src, dst = g.edge_index
        num_edges = src.size(0)
        keep_mask = torch.rand(num_edges) > self.p
        src, dst = src[keep_mask], dst[keep_mask]
        # random add
        num_add = int(self.p * num_edges)
        rand_src = torch.randint(0, g.num_nodes, (num_add,))
        rand_dst = torch.randint(0, g.num_nodes, (num_add,))
        g.edge_index = torch.vstack([torch.cat([src, rand_src]),
                                     torch.cat([dst, rand_dst])])
        return g

    def attr_mask(self, g: Data) -> Data:
        """Mask node attributes with probability p."""
        mask = torch.rand_like(g.x) < self.p
        g.x = g.x.masked_fill(mask, 0.)
        return g
