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
import numpy as np

from src.utils.device import select_device

from src._pyhessian.pyhessian.utils import (
    get_group_product,
    group_add,
    normalise_vector,
    get_active_params_and_gradient,
    get_hessian_vector_product,
    orthnormal,
)


class hessian:
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data_batch=None, dataloader=None):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """
        assert not (data_batch and dataloader) and (
            data_batch or dataloader
        ), "Only one data source can be used"

        self.model = model.eval()
        self.criterion = criterion
        self.device = select_device()

        if data_batch != None:
            self.data = data_batch
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            self.inputs, self.targets = self.inputs.to(self.device), self.targets.to(
                self.device
            )

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model and corresponding gradients
        self.params_active, self.gradient_active = get_active_params_and_gradient(
            self.model
        )

    def dataloader_hv_product(self, v):
        num_data = 0  # count the number of datum points in the dataloader

        THv = [
            torch.zeros(p.size()).to(self.device) for p in self.params_active
        ]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(self.device))
            loss = self.criterion(outputs, targets.to(self.device))
            loss.backward(create_graph=True)
            params, gradsH = get_active_params_and_gradient(self.model)
            self.model.zero_grad()
            hessian_vector_product = torch.autograd.grad(
                gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False
            )
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.0
                for THv1, Hv1 in zip(THv, hessian_vector_product)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = get_group_product(THv, v).cpu().item()
        return eigenvalue, THv

    # -------- [ main functions ] --------
    def eigenvalues(self, max_iterations=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        max_iterations: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1, "top_n should be greater than or equal to 1"
        eigenvalues, eigenvectors, computed_dim = [], [], 0

        while computed_dim < top_n:
            eigenvalue = None
            guess_vector = [
                torch.randn(p.size()).to(self.device) for p in self.params_active
            ]  # generate random vector
            guess_vector = normalise_vector(guess_vector)  # normalize the vector

            for _ in range(max_iterations):
                guess_vector = orthnormal(
                    guess_vector, eigenvectors
                )  # find the orthogonal vector to eigenvectors
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, hessian_vector_product = self.dataloader_hv_product(
                        guess_vector
                    )
                else:
                    hessian_vector_product = get_hessian_vector_product(
                        self.gradient_active, self.params_active, guess_vector
                    )
                    tmp_eigenvalue = (
                        get_group_product(hessian_vector_product, guess_vector)
                        .cpu()
                        .item()
                    )

                guess_vector = normalise_vector(hessian_vector_product)

                # [what a weird way to write this!]
                # if eigenvalue == None:
                #     eigenvalue = tmp_eigenvalue
                # else:
                #     if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                #         break
                #     else:
                #         eigenvalue = tmp_eigenvalue

                # [modified] by Don
                if (
                    eigenvalue is None
                    or abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                    >= tol
                ):
                    eigenvalue = tmp_eigenvalue
                else:
                    break
                # -------- [end] --------

            eigenvalues.append(eigenvalue)
            eigenvectors.append(guess_vector)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, max_iterations=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        max_iterations: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        trace_estimates, trace = [], 0.0

        for _ in range(max_iterations):
            self.model.zero_grad()
            rademacher_random_variables = [
                torch.randint_like(p, high=2, device=self.device)
                for p in self.params_active
            ]

            for v_i in rademacher_random_variables:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, hessian_vector_product = self.dataloader_hv_product(
                    rademacher_random_variables
                )
            else:
                hessian_vector_product = get_hessian_vector_product(
                    self.gradient_active,
                    self.params_active,
                    rademacher_random_variables,
                )

            trace_estimates.append(
                get_group_product(hessian_vector_product, rademacher_random_variables)
                .cpu()
                .item()
            )

            if abs(np.mean(trace_estimates) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_estimates
            else:
                trace = np.mean(trace_estimates)

        return trace_estimates

    def density(self, iter=100, num_runs=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        num_runs: number of SLQ runs
        """

        eigen_list_full, weight_list_full = [], []

        for _ in range(num_runs):
            rademacher_random_variables = [
                torch.randint_like(p, high=2, device=self.device)
                for p in self.params_active
            ]

            for v_i in rademacher_random_variables:
                v_i[v_i == 0] = -1

            rademacher_random_variables = normalise_vector(rademacher_random_variables)

            # standard lanczos algorithm initlization
            v_list = [rademacher_random_variables]
            w_list, alpha_list, beta_list = [], [], []
            ############### Lanczos
            for i in range(iter):
                self.model.zero_grad()
                w_prime = [
                    torch.zeros(p.size()).to(self.device) for p in self.params_active
                ]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(
                            rademacher_random_variables
                        )
                    else:
                        w_prime = get_hessian_vector_product(
                            self.gradient_active,
                            self.params_active,
                            rademacher_random_variables,
                        )
                    alpha = get_group_product(w_prime, rademacher_random_variables)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, rademacher_random_variables, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(get_group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.0:
                        # We should re-orth it
                        rademacher_random_variables = orthnormal(w, v_list)
                        v_list.append(rademacher_random_variables)
                    else:
                        # generate a new vector
                        w = [
                            torch.randn(p.size()).to(self.device)
                            for p in self.params_active
                        ]
                        rademacher_random_variables = orthnormal(w, v_list)
                        v_list.append(rademacher_random_variables)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(
                            rademacher_random_variables
                        )
                    else:
                        w_prime = get_hessian_vector_product(
                            self.gradient_active,
                            self.params_active,
                            rademacher_random_variables,
                        )
                    alpha = get_group_product(w_prime, rademacher_random_variables)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(
                        w_prime, rademacher_random_variables, alpha=-alpha
                    )
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            tridiagonal_matrix = torch.zeros(iter, iter).to(self.device)
            for i in range(len(alpha_list)):
                tridiagonal_matrix[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    tridiagonal_matrix[i + 1, i] = beta_list[i]
                    tridiagonal_matrix[i, i + 1] = beta_list[i]
            eigenvalues, eigenvectors = torch.linalg.eig(tridiagonal_matrix)

            eigen_list = eigenvalues.real
            weight_list = torch.pow(eigenvectors[0, :], 2)
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full
