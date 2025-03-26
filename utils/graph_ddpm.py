#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File (Python):  'graph_ddpm.py'
author:         Julien Straubhaar
date:           may-2024

Functions for Graph DDPM.
"""

import numpy as np
import networkx

import torch
import torch_geometric

# -----------------------------------------------------------------------------
# Data set to be used with data loader:
#   torch_geometric.loader.DataLoader(data_set, batch_size=batch_size, shuffle=True)
class Graph_geom_sampler_data_set(object):
    """
    Class defining a data set from a list of graphs (torch_geometric).

    The data set delivers (by `__getitem__`) item G_geom, a torch_geometric
    graph.

    Parameters
    ----------
    G_geom_list : list of `torch_geometric.data.Data`
        list of graphs (torch_geometric)
    
    G_nsample : sequence of ints (>=1)
        sequence of same length as `G_list`, of ints >= 1, 
        number of times that each graph in `G_list` is sampled, i.e.
        G_list[i] will be sampled G_nsample[i] times;
        hence, the length of "data set" is the cumulative sum of `G_nsample`

    Notes
    -----
    - methods `__len__` and `__getitem__` must be defined, so that instanciated \
    data set can be used with data loader from `pytorch_geometric` \
    (`torch_geometric.loader.DataLoader`)
    - before using a data loader, use `torch.random.manual_seed()` to ensure \
    reproducibility of batches delivered by the data loader (if needed)
    """
    def __init__(self, G_geom_list, G_nsample):
        """Constructor method.
        """
        # List of graphs 
        self.G_geom_list = G_geom_list

        # List of indices of sampled graph
        # - G_nsample[i] times i, for i 0, 1, ... len(G_list)-1
        self.G_index_list = np.repeat(range(len(G_nsample)), G_nsample)

        # Number of features
        self.n_node_features = G_geom_list[0].x.size(1)

        # Length of the data set
        self.len = len(self.G_index_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Select the graph 
        G_ind = self.G_index_list[idx]
        G_geom = self.G_geom_list[G_ind]

        return G_geom
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class Graph_DDPM(torch.nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) for graph.
    
    Ref: 
    
    - Ho et al. (2020) (https://doi.org/10.48550/arXiv.2006.11239)
    - Lin et al. (2024) (https://doi.org/10.48550/arXiv.2305.08891)
    
    Parameters
    ----------
    net : class :class:`Graph_DDPM_net_model` (torch.nn.Module)
        network for prediction of noise used between original data and
        its representation at any time step t or for prediction
        of original data from its representation at any time step t
        (see `learn_noise` below)
    
    n_steps : int
        number of time steps (typically in the order of thousands)
    
    n_node_features : int
        number of node features (length of the considered attribute
        attached to graph nodes) onto which diffusion process is applied
    
    betas : float or list of floats of length n_steps
        noise schedule, for each time step (if single float, it is repeated), 
        betas[t]: standard deviation of random gaussian noise from time step 
        t to time step t+1 (in forward process)
    
    force_snr_zero : bool, default: False
        if `True`, the noise schedule (`betas`) are modified to ensure that
        the "signal-to-noise-ration (snr)" is zero at final time step, where
        snr(t) = alpha_bars[t]**(1/2)/(1-alpha_bars[t])**(1/2), with 
        alpha_bars[t] the product of betas[s] with s <= t; the definition of
        snr(t) comes from the fact that after diffusion of x up to step t, we 
        get z(t) = alpha_bars[t]**(1/2) x + (1-alpha_bars[t])**(1/2) epsilon,
        where epsilon is a gaussian with noise (centered at 0 with variance 1);
        to ensure that the last snr is zero, the alpha_bars are linearly 
        rescaled by keeping alpha_bars[0] unchanged and sending 
        alpha_bars[n_steps-1] onto 0; see Lin et al. (2024)

    learn_noise : bool, default: True
        - if `True`: the network predicts noise used between original data and \
        its representation at any time step t
        - if `False`: the network predicts original data from \
        its representation at any time step t
        
    device : torch device, optional
        device on which the network is stored;
        by default (`None`): torch.device('cpu') will be used

    Notes
    -----
    if `learn_noise = False`, then `force_snr_zero` must be set to `False`, otherwise
    the reconstruction fails (see reconstruct when computing image representation at 
    previous time step).    
    """
    def __init__(self, 
                 net,
                 n_node_features,
                 n_steps=1000, 
                 betas=1.e-4,
                 force_snr_zero=False,
                 learn_noise=True,
                 device=None):
        """Constructor method.
        """
        super().__init__()
        self.n_steps = n_steps
        self.n_node_features = n_node_features
        self.net = net
        betas = torch.as_tensor(betas).view(-1)
        if len(betas) == 1:
            betas = torch.repeat_interleave(betas, n_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        if force_snr_zero:
            sqrt_alpha_bars_1 = torch.sqrt(alpha_bars[0])
            sqrt_alpha_bars_T = torch.sqrt(alpha_bars[-1])
            sqrt_alpha_bars = sqrt_alpha_bars_1/(sqrt_alpha_bars_1 - sqrt_alpha_bars_T) * (torch.sqrt(alpha_bars) - sqrt_alpha_bars_T)
            alpha_bars = sqrt_alpha_bars**2
            alphas[1:] = alpha_bars[1:] / alpha_bars[:-1]
            betas = 1.0 - alphas

        self.betas = betas    # noise schedule
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = torch.cat((alpha_bars[:1], alpha_bars[:-1]))
        self.learn_noise = learn_noise
        self.device = device

        self.to_device(device)

    def forward(self, G_batch, t, eta=None):
        """
        Forward process of the diffustion model. 
        
        Direct computation, from original data (`G_batch.x`) to any time step (`t`).

        Parameters
        ----------
        G_batch : `torch_geometric.data.batch.DataBatch`
            batch of graphs, with:
            
                - `G_batch.x` : tensor of size `(G_batch.num_nodes, G_batch.num_node_features)`, \
                attribute onto which the operations are applied

        t : tensor of Long, of size `(G_batch.num_graphs, )`
            target time steps for each graph in the batch (`G_batch`)
        
        eta : tensor, optional
            gaussian white noise (~N(0,1)) used to compute target data,
            tensor of same size as `G_batch.x`
 
        Returns
        -------
        G_batch : `torch_geometric.data.batch.DataBatch`
            target data after forward (applied on attribute x)
        
        Notes 
        -----
        In place operations are done, `G_batch` is modified.
        """
        # Make input data more noisy (we can directly skip to the desired step)
        if eta is None:
            eta = torch.randn_like(G_batch.x)

        # Define time step for all nodes
        n_nodes = G_batch.ptr[1:] - G_batch.ptr[:-1] # number of nodes of each graph
        t_batch = torch.repeat_interleave(t, n_nodes, dim=0)

        alpha_bar = self.alpha_bars[t_batch].view(G_batch.x.size(0), *((G_batch.x.ndim-1)*[1]))
        G_batch.x = alpha_bar.sqrt() * G_batch.x + (1.0 - alpha_bar).sqrt() * eta
        
        return G_batch

    def backward(self, G_batch, t):
        """
        Backward process of the diffustion model, estimation at step t,
        obtained by running the network.

        Parameters
        ----------
        G_batch : `torch_geometric.data.batch.DataBatch`
            batch of graphs, with:
            
                - `G_batch.x` : tensor of size `(G_batch.num_nodes, G_batch.num_node_features)`, \
                attribute onto which the operations are applied

        t : tensor of Long, of size `(G_batch.num_graphs,)`
            time steps for each graph in the batch (`G_batch`)
        
        Returns
        -------
        out : tensor
            estimation, tensor of same size as `G_batch.x`
        """
        # Define time step for all nodes
        n_nodes = G_batch.ptr[1:] - G_batch.ptr[:-1] # number of nodes of each graph
        t_batch = torch.repeat_interleave(t, n_nodes, dim=0)

        return self.net(G_batch.x, G_batch.edge_index, t_batch)

    def diffuse(self, G_batch, t0=None, t1=None, return_intermediate=False):
        """
        Diffuses noise to data, iteratively through time steps t0 to t1 (excluded).
        
        Parameters
        ----------
        G_batch : `torch_geometric.data.batch.DataBatch`
            batch of graphs, with:
            
            - `G_batch.x` : tensor of size `(G_batch.num_nodes, G_batch.num_node_features)`, \
            attribute onto which the operations are applied

        t0, t1 : ints, optional
            `t0 < t1`, starting and ending (excluded) time steps;
            by default: `t0 = 0` and `t1 = self.n_steps` are used
        
        return_intermediate : bool, default: `False`
            - if `True`: starting node features (`G_batch.x`), and node features obtained \
            after each time step (`t0`, ..., `t1-1`, forward) are returned in a list of \
            length `1+t0-t1`

        Returns
        -------
        G_batch : `torch_geometric.data.batch.DataBatch`
            final data after diffusion of noise 
            through all time steps from `t0` to `t1-1`)
        
        x_all : optional
            returned if `return_intermediate=True`, list of tensors (of same 
            size as `G_batch.x`) of length `1+t1-t0`, starting node features, and 
            node features after each time step (`t0`, ..., `t1-1`, forward)

        Notes 
        -----
        In place operations are done, `G_batch` is modified.
        """
        if t0 is None:
            t0 = 0
        if t1 is None:
            t1 = self.n_steps
        
        if return_intermediate:
            x_all = [G_batch.x]

        for t in range(t0, t1):
            eta = torch.randn_like(G_batch.x)
            G_batch.x = self.alphas[t].sqrt() * G_batch.x + self.betas[t].sqrt() * eta
            if return_intermediate:
                x_all.append(G_batch.x)

        if return_intermediate:
            return G_batch, x_all
        else:
            return G_batch

    def reconstruct(self, G_batch, t0=None, t1=None, sigmas=None, implicit=False, return_intermediate=False):
        """
        Reconstructs data, backward through time steps t0 to t1 (excluded).
        
        Parameters
        ----------
        G_batch : `torch_geometric.data.batch.DataBatch`
            batch of graphs, with:
            
                - `G_batch.x` : tensor of size `(G_batch.num_nodes, G_batch.num_node_features)`, \
                attribute onto which the operations are applied

        t0, t1 : ints, optional
            `t0 < t1`, starting and ending (excluded) time steps;
            by default: `t0 = 0` and `t1 = self.n_steps` are used
        
        sigmas : float or tensor of size (t1-t0,), optional
            standard deviation of noise added at each time step of the 
            reconstruction (backward + sampling process); 
            by default (`None`): default values given by the noise schedule
            (square roots of betas) are used
        
        implicit : bool, default: `False`
            - if `True`: no noise is added during the reconstruction, `sigmas` not used
            - if `False`: noise is added during the reconstruction, according to `sigmas`, \
            except at the last step (t0)
        
        return_intermediate : bool, default: `False`
            - if `True`: starting node features (`G_batch.x`), and node features obtained \
            after each time step (`t1-1, ..., t0`, backward) are returned in a list of \
            length `1+t0-t1`

        Returns
        -------
        G_batch : `torch_geometric.data.batch.DataBatch`
            final data after reconstruction 
            through all time steps from `t1-1` to `t0`)
        
        x_all : optional
            returned if `return_intermediate=True`, list of tensors (of same 
            size as `G_batch.x`) of length `1+t1-t0`, starting node features, and 
            node features after each time step (`t1-1, ..., t0`, backward)

        Notes 
        -----
        In place operations are done, `G_batch` is modified.
        """
        if t0 is None:
            t0 = 0
        if t1 is None:
            t1 = self.n_steps
        
        if sigmas is None:
            sigmas = self.betas[t0:t1].sqrt()
            # sigmas = (self.betas[t0:t1] * (1.0 - self.alpha_bars_prev[t0:t1]) / (1.0 - self.alpha_bars[t0:t1])).sqrt()
        else:
            sigmas = torch.as_tensor(sigmas).view(-1).to(self.device)            

        if return_intermediate:
            x_all = [G_batch.x]

        # with torch.no_grad():
        #     for i, t in enumerate(range(t1-1, t0-1, -1)):
        #         eta = self.backward(G_batch, torch.full((G_batch.num_graphs,), t).to(self.device))
        #         G_batch.x = 1.0 / torch.sqrt(self.alphas[t]) * (G_batch.x - self.betas[t] / torch.sqrt(1.0 - self.alpha_bars[t]) * eta)
        #         if t > t0 and not implicit:
        #             z = torch.randn_like(G_batch.x).to(self.device)
        #             G_batch.x = G_batch.x + sigmas[-i-1] * z
        #         if return_intermediate:
        #             x_all.append(G_batch.x)

        with torch.no_grad():
            for i, t in enumerate(range(t1-1, t0-1, -1)):
                if self.learn_noise:
                    # Getting estimation of noise (used from the original image)
                    eta = self.backward(G_batch, torch.full((G_batch.num_graphs,), t).to(self.device))
                    # Compute image representation at previous time step
                    G_batch.x = 1.0 / self.alphas[t].sqrt() * (G_batch.x - self.betas[t] / (1.0 - self.alpha_bars[t]).sqrt() * eta)
                else:
                    # Getting prediction of original image
                    x0 = self.backward(G_batch, torch.full((G_batch.num_graphs,), t).to(self.device))
                    # Compute image representation at previous time step
                    G_batch.x = 1.0 / (1.0 - self.alpha_bars[t]) * ((1.0 - self.alpha_bars_prev[t]) * self.alphas[t].sqrt() * G_batch.x + self.alpha_bars_prev[t].sqrt() * self.betas[t] * x0)
                if t > t0 and not implicit:
                # if t > 0 and not implicit:
                    z = torch.randn_like(G_batch.x).to(self.device)
                    G_batch.x = G_batch.x + sigmas[-i-1] * z
                if return_intermediate:
                    x_all.append(G_batch.x)

        # if self.learn_noise:
        #     with torch.no_grad():
        #         for i, t in enumerate(range(t1-1, t0-1, -1)):
        #             eta = self.backward(G_batch, torch.full((G_batch.num_graphs,), t).to(self.device))
        #             G_batch.x = 1.0 / torch.sqrt(self.alphas[t]) * (G_batch.x - self.betas[t] / torch.sqrt(1.0 - self.alpha_bars[t]) * eta)
        #             if t > t0 and not implicit:
        #                 z = torch.randn_like(G_batch.x).to(self.device)
        #                 G_batch.x = G_batch.x + sigmas[-i-1] * z
        #             if return_intermediate:
        #                 x_all.append(G_batch.x)
        # else:
        #     with torch.no_grad():
        #         for i, t in enumerate(range(t1-1, t0-1, -1)):
        #             G_batch.x = self.backward(G_batch, torch.full((G_batch.num_graphs,), t).to(self.device))
        #             if t > t0 and not implicit:
        #                 z = torch.randn_like(G_batch.x).to(self.device)
        #                 G_batch.x = G_batch.x + sigmas[-i-1] * z
        #             if return_intermediate:
        #                 x_all.append(G_batch.x)

        if return_intermediate:
            return G_batch, x_all
        else:
            return G_batch

    def generate(self, G_batch, generate_noise=True, sigmas=None, implicit=False, return_intermediate=False):
        """
        Generates data (from gaussian noise), according to DDPM (or DDIM, see `implicit` below).
        
        Parameters
        ----------
        G_batch : `torch_geometric.data.batch.DataBatch`
            batch of graphs, with:
            
            - `G_batch.x` : tensor of size `(G_batch.num_nodes, G_batch.num_node_features)`, \
            attribute onto which the operations are applied

            with starting noisy node features
            (`G_batch.x` ignored if `generate_noise=True`, see below)
        
        generate_noise : bool, default: `True`
            - if `True`: gaussian noise (in N(0, 1)) is generated for node features \
            (in `G_batch.x` )
            - if `False`: features in `G_batch.x` are used
        
        sigmas : float or tensor of size (n_steps, ), optional
            standard deviation of noise added at each time step of the 
            reconstruction (backward + sampling process); 
            by default (`None`): default values given by the noise schedule
            (square roots of betas) are used
        
        implicit : bool, default: `False`
            - if `True`: no noise is added during the reconstruction, `sigmas` not used
            - if `False`: noise is added during the reconstruction, according to `sigmas`, \
            except at the last step (t0)
        
        return_intermediate : bool, default: `False`
            - if `True`: initial node features (noise), and node features obtained \
            after each time step (`n_steps-1, ..., 0`) are returned in a list of length \
            `n_steps+1`

        Returns
        -------
        G_batch : `torch_geometric.data.batch.DataBatch`
            final data after reconstruction 
            through all time steps from `t1-1` to `t0`)
        
        x_all : optional
            returned if `return_intermediate=True`, list of tensors of length
            `n_steps+1`, initial node features and reconstructed node features after 
            each time step (`n_steps-1, ..., 0`)

        Notes 
        -----
        In place operations are done, `G_batch` is modified.
        """
        if generate_noise:
            G_batch.x = torch.randn_like(G_batch.x)
        
        return self.reconstruct(G_batch, sigmas=sigmas, implicit=implicit, return_intermediate=return_intermediate)

    def to_device(self, device):
        """
        Puts the model on device `device`.
        """
        self.net.to_device(device)
        # self.net.to(device)
        # self.net.time_embedding_tensor = self.net.time_embedding_tensor.to(device)

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.alpha_bars_prev = self.alpha_bars_prev.to(device)
        self.device = device
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def sinusoidal_embedding(n, d):
    """
    Computes the standard positional embedding.

    Parameters
    ----------
    n : int
        number of time steps in DDPM

    d : int
        number of dimension (for time embedding)
    
    Returns
    -------
    embedding : tensor of float of size `(n, d)`
        to embed an integer time step (from 0 to `n-1`), one hot encoding
        is used follows by the matrix multiplication with `embedding`
    """
    embedding = torch.zeros(n, d)
    t = torch.arange(n).view(n, 1)
    wk = (1 / 10_000 ** (2.0 * torch.arange(d)/d)).view(1, -1)
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class Graph_DDPM_net_model(torch.nn.Module):
    """
    Network to be used with DDPM.
    
    Network for prediction of noise used between original data and
    its representation at any time step t or for prediction
    of original data from its representation at any time step t
    (depend on `learn_noise` parameters of ddpm model).
    
    The design is inspired by "U-net": for an input set of features "x" on a 
    connected graph, the prediction of noise "eta" is done by applying 
    "downwards operations", "mid(dle) opoeration", "upwards operations" + 
    concatenation with corresponding level in downward part, and finaly a 
    "last operation" as illustrated below::

        x ---> down[0] - - - - - - - - - - - - - - - - - - - - - - - - - - - - > (cat)  up[nf_len-2] ---> up[nf_len-1] -----> last ---> eta
                 |                                                                       ^                             
                 |                                                                       |    
                 +---> down[1] - - - - - - - - - - - - - - - - -> (cat)  up[nf_len-3] ---+
                         .                                                ^
                         .                                                |
                         .                                                .
                         |                                                .   
                         |                                                .
                         +---> down[nf_len-2]  - - - - -> (cat)  up[0] ---+    
                                 |                                ^    
                                 |                                |    
                                 +---> down[nf_len-1] ---> mid ---+    

    Note that, contrary to U-net on images based on convolutional layer, the 
    support (graph) does not change during the process.
    
    Parameters
    ----------
    n_node_features : int
        number of node features in input / output of the net
    
    nf_list : list of ints
        number of node features at downwards/upwards steps (U-net style), 
        list of length "nf_len": 

            - operation "down[0]" has `n_node_features` input features, and\
            `nf_list[0]` output features
            - operation "down[i]" has `nf_list[i-1]` input features, and\
            `nf_list[i]` output features, for i = 1, ..., nf_len-1
            - operation "up[i]" has `nf_list[nf_len-1-i]` input features, and\
            `nf_list[nf_len-2-i]` output features, for i = 0, ..., nf_len-2
            - operation "up[`nf_len-1`]" has `nf_list[0]` input features, and\
            `nf_last` (see below) output features     

    nf_last : int, optional
        number of node features as input for "last operation";
        by default (`None`): set to `nf_list[0]`
    
    has_mid : bool, default: `False`
        if `True`, a "mid operation" (layer) is considered at the end of the 
        downwards steps
    
    nf_mid : int, optional
        number of node features within "mid" layer (used if `has_mid=True`);
        by default (`None`), set to last value in `nf_list`
    
    activation : torch activation module, default: torch.nn.SiLU()
        activation module used (except for time embedding and at output)
    
    te_activation : torch activation module, default: torch.nn.SiLU()
        activation module for time embedding
    
    normalize_down: bool, default : `True`
        if `True` a "normalization" operation (layer) is done before each
        "down" operation
    
    op_down1 : str, default : `'SAGEConv'`
        type for the first operation at each "down" operation
    
    op_down2 : str, optional
        type for the second operation at each "down" operation
    
    normalize_up: bool, default : `True`
        if `True` a "normalization" operation (layer) is done before each
        "up" operation
    
    op_up1 : str, default : `'GraphConv'`
        type for the first operation at each "up" operation
    
    op_up2 : str, optional
        type for the second operation at each "up" operation
    
    normalize_mid: bool, default : `True`
        if `True` a "normalization" operation (layer) is done before
        "mid" operation (used if `has_mid=True`)
    
    op_mid1 : str, default : `'GraphConv'`
        type for the first operation of "mid" operation (used if `has_mid=True`)
    
    op_mid2 : str, optional
        type for the second operation of "mid" operation (used if `has_mid=True`)
    
    normalize_last: bool, default : `True`
        if `True` a "normalization" operation (layer) is done before
        "last" operation
    
    op_last : str, default : `'GraphConv'`
        type for "last" operation
    
    n_steps : int, default: 1000
        number of time steps (for time embedding)
    
    time_embed_dim : int, default: 100
        dimension of time embedding
    
    Notes
    -----
    The string defining the type of operations must be a string <str> referring
    to a valid "torch_geometric.nn.<str>" operation. Moreoever, for last operation
    only, `op_last = 'Linear'` is accepted for `torch.nn.Linear` operation.
    """
    def __init__(self, 
                 n_node_features, 
                 nf_list, 
                 nf_last=None,
                 has_mid=False,
                 nf_mid=None,
                 activation=torch.nn.SiLU(),
                 te_activation=torch.nn.SiLU(),
                 normalize_down=True,
                 op_down1='SAGEConv',
                 op_down2=None,
                 normalize_up=True,
                 op_up1='GraphConv',
                 op_up2=None,
                 normalize_mid=True,
                 op_mid1='GraphConv',
                 op_mid2=None,
                 normalize_last=False,
                 op_last='GraphConv',
                 n_steps=1000, 
                 time_emb_dim=100):
        """Constructor method.
        """
        super().__init__()

        self.n_node_features = n_node_features
        self.nf_list = nf_list
        if nf_last is None:
            nf_last = nf_list[0]
        self.nf_last = nf_last
        self.nf_len = len(nf_list)

        self.has_mid = has_mid
        self.nf_mid = nf_mid

        self.n_steps = n_steps
        self.time_emb_dim = time_emb_dim

        # Positional embedding tensor
        self.time_embedding_tensor = sinusoidal_embedding(n_steps, time_emb_dim)
        #
        # self.time_embed = torch.nn.Embedding(n_steps, time_emb_dim)
        # self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        # self.time_embed.requires_grad_(False)
        # # Note: avoid using torch.nn.Embedding with fixed weights time_embed.weight.data (as in the 3 lines above)
        # # because these weights could be resetted when resetting / re-initializing weight of the module!

        # U-net style
        # - downwards
        nf_down = [self.n_node_features] + self.nf_list
        te_down = []
        laynorm_down = []
        layop_down1 = []
        layop_down2 = []
        for nf_a, nf_b in zip(nf_down[:-1], nf_down[1:]):
            te_down.append(self.time_embed_module(time_emb_dim, nf_a, activation=te_activation))
            if normalize_down:
                laynorm_down.append(torch_geometric.nn.norm.LayerNorm(nf_a))
            else:
                laynorm_down.append(None)
            layop_down1.append(eval(f'torch_geometric.nn.{op_down1}({nf_a}, {nf_b})'))
            if op_down2 is not None:
                layop_down2.append(eval(f'torch_geometric.nn.{op_down2}({nf_b}, {nf_b})'))
            else:
                layop_down2.append(None)

        self.te_down      = torch.nn.ModuleList(te_down)
        self.laynorm_down = torch.nn.ModuleList(laynorm_down)
        self.layop_down1  = torch.nn.ModuleList(layop_down1)
        self.layop_down2  = torch.nn.ModuleList(layop_down2)

        # - middle
        if self.has_mid:
            if self.nf_mid is None or op_mid2 is None:
                self.nf_mid = nf_down[-1]
            self.te_mid = self.time_embed_module(time_emb_dim, nf_down[-1], activation=te_activation)
            if normalize_mid:
               self.laynorm_mid = torch_geometric.nn.norm.LayerNorm(nf_down[-1])
            else:
                self.laynorm_mid = None
            self.layop_mid1 = eval(f'torch_geometric.nn.{op_mid1}({nf_down[-1]}, {self.nf_mid})')
            if op_mid2 is not None:
                self.layop_mid2 = eval(f'torch_geometric.nn.{op_mid2}({self.nf_mid}, {nf_down[-1]})')
            else:
                self.layop_mid2 = None

        # - upwards
        nf_up = self.nf_list[::-1] + [self.nf_last]
        te_up = []
        laynorm_up = []
        layop_up1 = []
        layop_up2 = []
        for i, (nf_a, nf_b) in enumerate(zip(nf_up[:-1], nf_up[1:])):
            if i > 0:
                nf_a2 = 2*nf_a
            else:
                nf_a2 = nf_a
            te_up.append(self.time_embed_module(time_emb_dim, nf_a2, activation=te_activation))
            if normalize_up:
                laynorm_up.append(torch_geometric.nn.norm.LayerNorm(nf_a2))
            else:
                laynorm_up.append(None)
            if op_up2 is not None:
                layop_up1.append(eval(f'torch_geometric.nn.{op_up1}({nf_a2}, {nf_a})'))
                layop_up2.append(eval(f'torch_geometric.nn.{op_up2}({nf_a}, {nf_b})'))
            else:
                layop_up1.append(eval(f'torch_geometric.nn.{op_up1}({nf_a2}, {nf_b})'))
                layop_up2.append(None)

        self.te_up      = torch.nn.ModuleList(te_up)
        self.laynorm_up = torch.nn.ModuleList(laynorm_up)
        self.layop_up1   = torch.nn.ModuleList(layop_up1)
        self.layop_up2   = torch.nn.ModuleList(layop_up2)

        # - last
        if normalize_last:
            self.laynorm_last = torch_geometric.nn.norm.LayerNorm(self.nf_last)
        else:
            self.laynorm_last = None
        self.layop_last = eval(f'torch_geometric.nn.{op_last}({self.nf_last}, {self.n_node_features})')

        if op_last == 'Linear':
            self.layop_last = eval(f'torch.nn.{op_last}({self.nf_last}, {self.n_node_features})')
        else:
            self.layop_last = eval(f'torch_geometric.nn.{op_last}({self.nf_last}, {self.n_node_features})')

        # activation
        self.activation = activation
        # self.te_activation = te_activation

        self.init_weights()

    def forward(self, x, edge_index, t):
        """
        Forward. 
        
        Parameters
        ----------
        x : tensor of floats
            attributes attached to the nodes of input graph(s)
        
        edge_index : tensor of Long
            edge index for the input graph(s) (given their topology)
        
        t : tensor of Long, of size `(G_batch.num_graphs, )`
            time step for each node of the input graph(s), nodes belonging to
            the same graph should have the same time step
 
        Returns
        -------
        eta : tensor
            predicted noise through DDPM, tensor of same size as `x`
        """
        # Positional embedding
        t = torch.matmul(torch.nn.functional.one_hot(t, num_classes=self.n_steps).to(torch.float), self.time_embedding_tensor)
        #
        # t = self.time_embed(t)
        # # Note: avoid a module to do that (see Note in the __init__)

        out = x
        # print('...start...', out[-1].size())

        out_down = []
        for i, (te, laynorm, layop1, layop2) in enumerate(zip(self.te_down, self.laynorm_down, self.layop_down1, self.layop_down2)):
            out = out + te(t)
            if laynorm is not None:
                out = laynorm(out)
            out = layop1(out, edge_index)
            out = self.activation(out)
            if layop2 is not None:
                out = layop2(out, edge_index)
                out = self.activation(out)
            if i < self.nf_len - 1:
                out_down.append(out)
                # i.e. do not store the last one
            # print('...down...', i, out.size())
            
        if self.has_mid:
            out = out + self.te_mid(t)
            if self.laynorm_mid is not None:
                out = self.laynorm_mid(out)
            ####out = self.lin_mid(out)
            out = self.layop_mid1(out, edge_index)
            out = self.activation(out)
            if self.layop_mid2 is not None:
                out = self.layop_mid2(out, edge_index)
                out = self.activation(out)
            # print('...mid...', out.size())

        for i, (te, laynorm, layop1, layop2) in enumerate(zip(self.te_up, self.laynorm_up, self.layop_up1, self.layop_up2)):
            out = out + te(t)
            if laynorm is not None:
                out = laynorm(out)
            out = layop1(out, edge_index)
            out = self.activation(out)
            if layop2 is not None:
                out = layop2(out, edge_index)
                out = self.activation(out)
            if i < self.nf_len - 1:
                out = torch.cat((out_down[-i-1], out), dim=1)
            # print('...up...', i, out.size())

        if self.laynorm_last is not None:
            out = self.laynorm_last(out)
        if isinstance(self.layop_last, torch.nn.Linear):
            out = self.layop_last(out)
        else:
            out = self.layop_last(out, edge_index)
        # print('...last...', i, out.size())
        
        return out

    def time_embed_module(self, dim_in, dim_out, activation=torch.nn.SiLU()):
        """
        Time embedding

        Parameters
        ----------
        dim_in: int
            input dimension of time embedding
        
        dim_out: int
            output dimension of time embedding
        
        activation : torch activation module, default: torch.nn.SiLU()
            activation module used
        """
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            activation, #torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out)
        )

    def init_weights(self, gain=1.0, seed=None):
        """Initializes weights of the network."""
        if seed is not None:
            torch.random.manual_seed(seed)

        for name, param in self.named_parameters():
            # print('PARAM...', name)
            if 'bias' in name:
                # print('...PARAM BIAS', name)
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                if param.ndim == 1:
                    # print('...PARAM WEIGHT1', name)    
                    # layerNorm
                    torch.nn.init.constant_(param, 1.0) 
                else:
                    # print('...PARAM WEIGHT2', name)    
                    torch.nn.init.xavier_uniform_(param, gain=gain)
                                                    # gain=nn.init.calculate_gain('sigmoid')
                                                    # gain=nn.init.calculate_gain('relu')
                                                    # ...

    def to_device(self, device):
        """
        Puts the model on device `device`.
        """
        self.to(device)
        self.time_embedding_tensor = self.time_embedding_tensor.to(device)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def train_graph_ddpm(
        train_data_loader,
        ddpm, 
        optimizer,
        loss_func,
        lr_scheduler=None,
        return_lr=False,
        return_loss=True,
        valid_data_loader=None,
        num_epochs=10,
        print_epoch=1,
        G_batch_fixed=None,
        save_gen_epoch=0,
        save_gen_file_fmt='./gen_{:04d}.pt',
        device=torch.device('cpu')):
    """
    Trains denoising diffusion probabilistic model (ddpm) for graph node features.
    
    Parameters
    ----------
    train_data_loader : `torch_geometric.loader.DataLoader`
        data loader yielding batch of graphs over training data set
    
    ddpm : model
        model (network) for the denoising diffusion probabilistic model (ddpm)
    
    loss_func : callable
        loss function
    
    updater : optimizer
        updater, torch optimizer for the model, 
        e.g. `torch.optim.SGD(ddpm.parameters(), weight_decay=0.001, lr=0.03)`
    
    lr_scheduler : scheduler, optional
        learning rate scheduler for the model, 
        e.g. `torch.optim.lr_scheduler.*`
    
    return_lr : bool, default: `False`
        - if `True`: return sequence of learning rates used
    
    return_loss : bool, default: `True`
        - if `True`: return the sequence of losses for training data set and \
        validation data set (if present)
    
    valid_data_loader : `torch_geometric.loader.DataLoader`, optional
        data loader yielding batch of graphs over validation data set
    
    num_epochs : int, default: 10
        number of epochs
    
    print_epoch : int, default: 1
        result of every `print_epoch` epoch is displayed in stdout, if 
        `print_epoch > 0`
    
    G_batch_fixed : `torch_geometric.data.batch.DataBatch`
        batch of graphs, with fixed random noise in N(0,1) used as starting node features
        (in G_batch_fixed.x), to generate data to save (see `save_gen_epoch`); 
        unused if `save_fake_epoch<=0`
    
    save_gen_epoch : int, default: 0
        if `save_gen_epoch > 0`, at each `save_gen_epoch` epoch, generate data using 
        the model and `G_batch_fixed` as input, and save them (write on the disk), 
        provided `G_batch_fixed` is specified (not `None`)
    
    save_gen_file_fmt : str, default: './gen_{:04d}.pt'
        string for filename (including path) for saving generated data (batch)
        (see `save_gen_epoch`), at epoch i the file will be 
        `save_gen_file_fmt.format(i)`
    
    device : torch device, default: torch.device('cpu')
        device on which the model is trained
    
    Returns
    -------
    train_loss : list, optional
        returned if `return_loss=True`, 
        loss at every epoch, list of floats of length `num_epochs`
    
    valid_loss : list, optional
        returned if `return_loss=True` and `valid_data_loader`, 
        loss at every epoch, list of floats of length `num_epochs`
    
    lr_used : list, optional
        returned if `return_lr=True`, learning rate used for conditional
        variational autoencoder at each epoch, list of floats of length 
        `num_epochs`,        
    """
    fname = 'train_ddpm'

    print('*** Training on', device, '***')

    # Copy model to device
    ddpm.to_device(device)

    # Initialize list for loss
    if return_loss:
        train_loss = []
        loss_epoch = 0.0 # reset loss of one epoch
        if valid_data_loader:
            valid_loss = []

    # Initialize list for lr
    if return_lr:
        lr_used = []

    # Check save_gen_epoch and z_fixed
    if save_gen_epoch:
        if G_batch_fixed is None:
            print(f'ERROR ({fname}): `G_batch_fixed` must be specified for saving generated data during training')
            return
        x_fixed = G_batch_fixed.x.clone()
        # -> G_batch_fixed, x_fixed on "cpu"

    # Train the network
    for epoch in range(num_epochs):
        ddpm.train() # set model in training mode
        # Train DDPM through every mini-batch
        train_len = 0  # reset train length
        for G_batch in train_data_loader:
            # one mini-batch
            G_batch = G_batch.to(device)

            # Picking some noise for node features of each graph in the batch, and a timestep
            eta = torch.randn_like(G_batch.x)
            t = torch.randint(0, ddpm.n_steps, (G_batch.num_graphs,), device=device)

            if ddpm.learn_noise:
                # Computing the noisy image based on X and the time-step (forward process, in place)
                G_batch = ddpm(G_batch, t, eta)
                # Getting estimation of noise based on the images and the time-step
                eta_hat = ddpm.backward(G_batch, t)

                # Define time step for all nodes
                n_nodes = G_batch.ptr[1:] - G_batch.ptr[:-1] # number of nodes of each graph
                t_batch = torch.repeat_interleave(t, n_nodes, dim=0)
                # Compute loss: the MSE between the noise plugged and the predicted noise
                loss = loss_func(eta_hat, eta)
            else:
                G_batch_x = G_batch.x # make a copy of G_batch.x
                # Computing the noisy image based on X and the time-step (forward process, in place)
                G_batch = ddpm(G_batch, t, eta)
                # Getting prediction of original image
                G_batch_x_hat = ddpm.backward(G_batch, t)
                # Compute loss: the MSE between the original image and the predicted image
                loss = loss_func(G_batch_x_hat, G_batch_x)
                
            # Optimizing the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if return_loss:
                with torch.no_grad():
                    loss_epoch += loss.item()*G_batch.num_graphs

            train_len += G_batch.num_graphs

        if return_lr:
            lr_used.append(optimizer.param_groups[0]['lr'])
        if lr_scheduler:
            # lr_used.append(lr_scheduler.get_last_lr()[0])
            lr_scheduler.step()

        if return_loss:
            train_loss.append(loss_epoch/train_len)
            loss_epoch = 0.0 # reset loss of one epoch

        if valid_data_loader:
            # Compute loss for the validation data set
            ddpm.eval() # set the variational autoencoder in evaluation mode
            valid_len = 0  # reset valid length
            with torch.no_grad():
                for G_batch in valid_data_loader:
                    # one mini-batch
                    G_batch = G_batch.to(device)

                    # Picking some noise for node features of each graph in the batch, and a timestep
                    eta = torch.randn_like(G_batch.x)
                    t = torch.randint(0, ddpm.n_steps, (G_batch.num_graphs,), device=device)

                    if ddpm.learn_noise:
                        # Computing the noisy image based on X and the time-step (forward process)
                        G_batch = ddpm(G_batch, t, eta)
                        # Getting model estimation of noise based on the images and the time-step
                        eta_hat = ddpm.backward(G_batch, t)
                        # Compute loss: the MSE between the noise plugged and the predicted noise
                        loss = loss_func(eta_hat, eta)
                    else:
                        G_batch_x = G_batch.x # make a copy of G_batch.x
                        # Computing the noisy image based on X and the time-step (forward process, in place)
                        G_batch = ddpm(G_batch, t, eta)
                        # Getting estimation of noisy image at previous time-step
                        G_batch_x_hat = ddpm.backward(G_batch, t)
                        # Compute loss: the MSE between the original image and the predicted image
                        loss = loss_func(G_batch_x_hat, G_batch_x)

                    if return_loss:
                        with torch.no_grad():
                            loss_epoch += loss.item()*G_batch.num_graphs

                    valid_len += G_batch.num_graphs

                if return_loss:
                    valid_loss.append(loss_epoch/valid_len)
                    loss_epoch = 0.0 # reset loss of one epoch

        if print_epoch > 0 and epoch % print_epoch == 0:
            # Print result of current epoch
            s = f'epoch {epoch+1} of {num_epochs}'
            if return_loss:
                s = s + f', loss : train: {train_loss[-1]:12.8f}'
                if valid_data_loader:
                    s = s + f', valid: {valid_loss[-1]:12.8f}'
            print(s)

        if save_gen_epoch > 0 and epoch%save_gen_epoch == 0:
            with torch.no_grad():
                G_batch_fixed.x = x_fixed.clone() 
                # -> G_batch_fixed on "cpu"
                G_batch_fixed = ddpm.generate(G_batch_fixed.to(device), generate_noise=False, sigmas=None, implicit=False, return_intermediate=False)
                # -> G_batch_fixed on "device"
                torch.save(G_batch_fixed.to('cpu'), save_gen_file_fmt.format(epoch))
                # -> G_batch_fixed on "cpu"

    # Set model on cpu
    ddpm.to_device(torch.device('cpu'))

    out = []
    if return_loss:
        out.append(train_loss)
        if valid_data_loader:
            out.append(valid_loss)
    if return_lr:
        out.append(lr_used)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    elif len(out) == 0:
        out = None
    return out
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def generate_graph_node_features(
        G, 
        ddpm, 
        attr='x',
        end_rescale=None,
        end_center=None,
        generate_noise=True, 
        sigmas=None, 
        implicit=False, 
        return_intermediate=False,
        device=torch.device('cpu')):
    """
    Generates node features on graph `G`, according to DDPM (or DDIM, see `implicit` below).
        
    Parameters
    ----------
    G : `networkx.Graph`
        graph
    
    ddpm : class :class:`Graph_DDPM`
        ddpm model
    
    attr : str, default: 'x'
        name of the feature to be generated
    
    end_rescale : sequence or tensor, optional
        scale factor applied at the end, sequence of same
        length as the length of `attr`, 
        if specified, `end_center` should also be specified
    
    end_center : sequence or tensor, optional
        center of features (shifted at the end), sequence of same
        length as the length of `attr`
        if specified, `end_rescale` should also be specified
    
    sigmas : float or tensor of size (ddpm.n_steps, ), optional
        standard deviation of noise added at each time step of the 
        reconstruction (backward + sampling process); 
        by default (`None`): default values given by the noise schedule
        (square roots of betas) are used
    
    implicit : bool, default: `False`
        - if `True`: no noise is added during the reconstruction, `sigmas` not used
        - if `False`: noise is added during the reconstruction, according to `sigmas`, \
        except at the last step (t0)
    
    return_intermediate : bool, default: `False`
        - if `True`: initial node features (noise), and node features obtained \
        after each time step (`n_steps-1, ..., 0`) are returned in a list of length \
        `n_steps+1`
    
    device : torch device, default: torch.device('cpu')
        device on which the network is trained

    Returns
    -------
    G : `networkx.Graph`
        graph with attribute `attr` containing the generated node features
    
    x_all : optional
        returned if `return_intermediate=True`, list of tensors of length
        `n_steps+1`, initial node features and reconstructed node features after 
        each time step (`n_steps-1, ..., 0`)

    Notes 
    -----
    In place operations are done, `G` is modified.
    """
    fname = 'generate_graph_node_features'
    if (end_rescale is None and end_center is not None) or (end_rescale is not None and end_center is None):
        raise ValueError(f'{fname}: `end_rescale` and `end_center` must be both specified')

    # Convert to torch_geometric (with zeros as node features)
    G_geom = torch_geometric.utils.from_networkx(G)
    if generate_noise:
        G_geom.x = torch.zeros((G_geom.num_nodes, ddpm.n_node_features))
        # else: G_geom.x must already contain starting noise

    # Convert to batch of one graph
    G_batch = torch_geometric.data.Batch.from_data_list([G_geom]) # torch_geometric.data.batch.DataBatch

    # Set model on specified device
    ddpm.to_device(device)
    G_batch.to(device)

    # Generation
    out = ddpm.generate(
            G_batch,
            generate_noise=generate_noise, 
            sigmas=sigmas, 
            implicit=implicit, 
            return_intermediate=return_intermediate)

    if return_intermediate:
        G_batch, x_all = out
        if x_all[0].device != torch.device('cpu'):
            x_all = [x.to('cpu') for x in x_all]
    else:
        G_batch = out

    if end_rescale is not None:
        # end_center is also not None
        if isinstance(end_rescale, list):
            end_rescale = np.asarray(end_rescale)
        if isinstance(end_rescale, np.ndarray):
            end_rescale = torch.from_numpy(end_rescale).to(torch.float)
        if isinstance(end_center, list):
            end_center = np.asarray(end_center)
        if isinstance(end_center, np.ndarray):
            end_center = torch.from_numpy(end_center).to(torch.float)
        # transformation:
        #   1. correction : substract mean of features on each batch
        #   2. multiply by end_scale
        #   3. shift by end_center
        G_batch.x = end_center + end_rescale * (G_batch.x.to('cpu') - torch.mean(G_batch.x.to('cpu'), dim=0))
        if return_intermediate:
            x_all = [end_center + end_rescale*(x - torch.mean(x, dim=0)) for x in x_all]
    
    node_features_dict = {i: xi.tolist() for i, xi in enumerate(G_batch.x.to('cpu').numpy())}
    networkx.set_node_attributes(G, node_features_dict, attr)

    ddpm.to_device('cpu')

    if return_intermediate:
        return G, x_all
    else:
        return G
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def generate_list_graph_node_features(
        G_list, 
        ddpm, 
        attr='x', 
        end_rescale=None,
        end_center=None,
        generate_noise=True, 
        sigmas=None, 
        implicit=False, 
        return_intermediate=False,
        device=torch.device('cpu')):
    """
    Generates node features on a list of graphs, according to DDPM (or DDIM, see `implicit` below).
        
    Parameters
    ----------
    G_list : list of `networkx.Graph`
        list of graphs
    
    ddpm : class :class:`Graph_DDPM`
        ddpm model
    
    attr : str, default: 'x'
        name of the feature(s) to be generated
    
    end_rescale : sequence or tensor, optional
        scale factor applied at the end, sequence of same
        length as the length of `attr`, 
        if specified, `end_center` should also be specified
    
    end_center : sequence or tensor, optional
        center of features (shifted at the end), sequence of same
        length as the length of `attr`
        if specified, `end_rescale` should also be specified

    generate_noise : bool, default: `True`
        - if `True`: gaussian noise (in N(0, 1)) is generated for "starting" \
        node features 'x' for the graphs in `G_list`
        - if `False`: it is assumed that the graphs in `G_list` have already\
        "starting" nodes features 'x'
    
    sigmas : float or tensor of size (ddpm.n_steps, ), optional
        standard deviation of noise added at each time step of the 
        reconstruction (backward + sampling process); 
        by default (`None`): default values given by the noise schedule
        (square roots of betas) are used
    
    implicit : bool, default: `False`
        - if `True`: no noise is added during the reconstruction, `sigmas` not used
        - if `False`: noise is added during the reconstruction, according to `sigmas`, \
        except at the last step (t0)
    
    return_intermediate : bool, default: `False`
        - if `True`: initial node features (noise), and node features obtained \
        after each time step (`n_steps-1, ..., 0`) are returned in a list of length \
        `n_steps+1`
    
    device : torch device, default: torch.device('cpu')
        device on which the network is trained

    Returns
    -------
    G_list : list of `networkx.Graph`
        list of graphs with attribute `attr` containing the generated node features
    
    x_all : optional
        returned if `return_intermediate=True`, list of tensors of length
        `n_steps+1`, initial node features and reconstructed node features after 
        each time step (`n_steps-1, ..., 0`)

    Notes 
    -----
    In place operations are done, `G` is modified.
    """
    fname = 'generate_list_graph_node_features'
    if (end_rescale is None and end_center is not None) or (end_rescale is not None and end_center is None):
        raise ValueError(f'{fname}: `end_rescale` and `end_center` must be both specified')
    
    # Convert to torch_geometric (with zeros as node features)
    G_geom_list = [torch_geometric.utils.from_networkx(G) for G in G_list]

    # Convert to batch of one graph
    G_batch = torch_geometric.data.Batch.from_data_list(G_geom_list) # torch_geometric.data.batch.DataBatch
    if generate_noise:
        G_batch.x = torch.zeros((G_batch.num_nodes, ddpm.n_node_features))
    # else: G_batch.x must already contain starting noise

    # Set model on specified device
    ddpm.to_device(device)
    G_batch.to(device)

    # Generation
    out = ddpm.generate(
            G_batch,
            generate_noise=generate_noise, 
            sigmas=sigmas, 
            implicit=implicit, 
            return_intermediate=return_intermediate)

    if return_intermediate:
        G_batch, x_all = out
        if x_all[0].device != torch.device('cpu'):
            x_all = [x.to('cpu') for x in x_all]
    else:
        G_batch = out

    if end_rescale is not None:
        # end_center is also not None
        if isinstance(end_rescale, list):
            end_rescale = np.asarray(end_rescale)
        if isinstance(end_rescale, np.ndarray):
            end_rescale = torch.from_numpy(end_rescale).to(torch.float)
        if isinstance(end_center, list):
            end_center = np.asarray(end_center)
        if isinstance(end_center, np.ndarray):
            end_center = torch.from_numpy(end_center).to(torch.float)
        # transformation:
        #   1. correction : substract mean of features on each batch
        #   2. multiply by end_scale
        #   3. shift by end_center
        G_batch.x = G_batch.x.to('cpu')
        for k in range(len(G_list)):
            G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]] = end_center + end_rescale*(G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]] - torch.mean(G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]], dim=0))
        if return_intermediate:
            for j in range(len(x_all)):
                for k in range(len(G_list)):
                    x_all[j][k] = end_center + end_rescale*(x_all[j][k] - torch.mean(x_all[j][k], dim=0))

    for k in range(len(G_list)):
        node_features_dict = {i: xi.tolist() for i, xi in enumerate(G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1], :].to('cpu').numpy())}
        networkx.set_node_attributes(G_list[k], node_features_dict, attr)

    ddpm.to_device('cpu')

    if return_intermediate:
        return G_list, x_all
    else:
        return G_list
# ------------------------------------------------------------------------------
