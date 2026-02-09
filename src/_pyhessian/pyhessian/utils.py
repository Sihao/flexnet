# *
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
# *

import torch


def get_group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def normalise_vector(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = get_group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, _ in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def get_active_params_and_gradient(model):
    """
    get active model parameters and corresponding gradients
    """
    params, gradients = [], []

    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        gradient = (
            torch.zeros_like(param).requires_grad_(True)
            if param.grad is None
            else param.grad
        )
        gradients.append(gradient)
    return params, gradients


def get_hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hessian_vector_product = torch.autograd.grad(
        gradsH,
        params,
        grad_outputs=v,
        only_inputs=True,
        retain_graph=True,
        allow_unused=True,
    )
    # [filter none (this should be mathematically correct)] by Don
    hessian_vector_product = [
        x if x is not None else torch.zeros_like(p)
        for x, p in zip(hessian_vector_product, params)
    ]
    return hessian_vector_product


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-get_group_product(w, v))
    return normalise_vector(w)
