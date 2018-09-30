# -*- coding: utf-8 -*-
"""Error functions.

This module defines error functions, with the aim of model training being to
minimise the error function given a set of inputs and target outputs.

The error functions will typically measure some concept of distance between the
model outputs and target outputs, averaged over all data points in the data set
or batch.
"""

import numpy as np


class SumOfSquaredDiffsError(object):
    """Sum of squared differences (squared Euclidean distance) error."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar error function value.
        """
        batch_diff = outputs - targets
        batch_diff_sq = np.square(batch_diff)
        batch_err = 0.5*np.sum(batch_diff_sq, axis=1)

        sum_sq_diff = np.sum(batch_err)
        N = outputs.shape[0]
        err = (1/N) * sum_sq_diff
        return err

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs. This should be
            an array of shape (batch_size, output_dim).
        """
        diff = outputs - targets
        N = outputs.shape[0]
        err_grad = (1/N)*diff
        return err_grad

    def __repr__(self):
        return 'SumOfSquaredDiffsError'
