#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File (Python):  'ml_utils.py'
author:         Julien Straubhaar
date:           2024

General tools for machine learning (ml).
"""

import torch

# ==============================================================================
# Dealing with gpu / cpu
# ==============================================================================

# ------------------------------------------------------------------------------
def try_gpu(i=0):
    """Returns gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def try_all_gpus():
    """Returns all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
# ------------------------------------------------------------------------------

# ==============================================================================
# Dealing with network parameters
# ==============================================================================

# ------------------------------------------------------------------------------
def nb_net_params(net):
    """
    Returns the total number of (learnable) parameters in a net.
    """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
# ------------------------------------------------------------------------------

# ==============================================================================
# Size after conolutional layers
# ==============================================================================

# ------------------------------------------------------------------------------
def len_after_conv(input_len, kernel_size, stride=1, padding=0):
    """
    Computes size in output for a convolutional layer

    For a convolutional layer with kernel_size=k, stride=s, padding=p, 
    and n=input_dim, n'=output_dim: n' = 1 + (n - k + 2p)/s
    """
    return 1 + (input_len - kernel_size + 2*padding)/stride
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def len_after_convTranspose(input_len, kernel_size, stride=1, padding=0):
    """Computes size in output for a transposed convolutional layer.
    
    For a transposed convolutional layer with kernel_size=k, stride=s, padding=p, 
    and n'=input_dim, n=output_dim: n = (n'-1)*s + k - 2p
    """
    return (input_len - 1) * stride + kernel_size - 2*padding
# ------------------------------------------------------------------------------

# ==============================================================================
# Reset model parameters
# ==============================================================================
# From: 
#     https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch

# ------------------------------------------------------------------------------
def reset_all_parameters(model):
    """
    Resets all parameters of the given model.
    
    Using `torch.random.manual_seed(.)` beforehand allows for reproducibility.

    Parameters
    ----------
    model : torch.nn.Module

    References
    ----------
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def reset_param(m):
        # - check if the current module has reset_parameters & if it's callable called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=reset_param)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def reset_all_parameters_with_specific_layer_type(model, module_type):
    """
    Resets all parameters of modules of specidied type in the given model.
    
    Using `torch.random.manual_seed(.)` beforehand allows for reproducibility.

    Parameters
    ----------
    model : torch.nn.Module
    module_type : torch.nn.Module type
        module type, e.g. `torch.nn.Linear`
    Notes
    -----
    ref:
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def reset_param(m):
        if type(m) == module_type:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(reset_param)
# ------------------------------------------------------------------------------
