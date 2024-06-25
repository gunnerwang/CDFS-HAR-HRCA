import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

NUM_SAMPLES=1

def scm(context_features, context_labels, target_features):
    class_representations = OrderedDict() 
    class_precision_matrices = OrderedDict()
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(context_features, context_labels)
    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))
    number_of_classes = class_means.size(0)
    number_of_targets = target_features.size(0)
    repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes, repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1,0) * -1
    class_representations.clear()
    return split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.size(0)])

def build_class_reps_and_covariance_estimates(context_features, context_labels):
    class_representations = OrderedDict()
    class_precision_matrices = OrderedDict()
    task_covariance_estimate = estimate_cov(context_features)
    for c in torch.unique(context_labels):
        class_features = torch.index_select(context_features, 0, extract_class_indices(context_labels, c))
        class_rep = mean_pooling(class_features)
        class_representations[c.item()] = class_rep
        lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
        class_precision_matrices[c.item()] = torch.inverse((lambda_k_tau * estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                + torch.eye(class_features.size(1), class_features.size(1)).cuda(0))
    return class_representations, class_precision_matrices

def estimate_cov(examples, rowvar=False, inplace=False):
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()

def split_first_dim_linear(x, first_two_dims):
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class) 
    class_mask_indices = torch.nonzero(class_mask)
    return torch.reshape(class_mask_indices, (-1,))



