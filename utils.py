import torch
import torch.nn.functional as F
import torchmetrics
import os
import math
from enum import Enum
import sys


class TestAccuracies:
    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
        self.current_best_accuracy_dict = {}
        for dataset in self.datasets:
            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

    def is_better(self, accuracies_dict):
        is_better = False
        is_better_count = 0
        for i, dataset in enumerate(self.datasets):
            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
                is_better_count += 1
        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
            is_better = True
        return is_better

    def replace(self, accuracies_dict):
        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")
        print_and_log(logfile, "Test Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")

    def get_current_best_accuracy_dict(self):
        return self.current_best_accuracy_dict

def verify_checkpoint_dir(checkpoint_dir, resume, test_mode):
    if resume and not test_mode: 
        if not os.path.exists(checkpoint_dir):
            print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()

        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_file):
            print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
            sys.exit()
    elif test_mode:
       if not os.path.exists(checkpoint_dir):
           print("Can't test. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
           sys.exit()
    else:
        if os.path.exists(checkpoint_dir):
            print("Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
            print("If starting a new training run, specify a directory that does not already exist.", flush=True)
            print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
            sys.exit()


def print_and_log(log_file, message):
    print(message, flush=True)
    log_file.write(message + '\n')

def get_log_files(checkpoint_dir, resume, test_mode):
    verify_checkpoint_dir(checkpoint_dir, resume, test_mode)
    if not resume:
        os.makedirs(checkpoint_dir)
    checkpoint_path_validation = os.path.join(checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final

def stack_first_dim(x):
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)

def split_first_dim_linear(x, first_two_dims):
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)

def sample_normal(mean, var, num_samples):
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()

def loss(test_logits_sample, test_labels, device):
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)

def aggregate_accuracy(test_logits_sample, test_labels):
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())

def topk_accuracy(test_logits, test_labels, num_classes, device):
    top1 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device)
    top2 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=2).to(device)
    top3 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=3).to(device)
    top5 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
    return top1(test_logits, test_labels), top2(test_logits, test_labels), top3(test_logits, test_labels), top5(test_logits, test_labels)

def task_confusion(test_logits, test_labels, real_test_labels, batch_class_list):
    preds = torch.argmax(torch.logsumexp(test_logits, dim=0), dim=-1)
    real_preds = batch_class_list[preds]
    return real_preds

def linear_classifier(x, param_dict):
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])
