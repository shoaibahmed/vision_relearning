import time
from tqdm import tqdm
from typing import Iterable, Dict

import torch
import numpy as np


@torch.no_grad()
def add_gaussian_noise_to_parameters(model: torch.nn.Module, std: float):
    """
    Adds i.i.d. Gaussian noise (mean=0, std=std) to all parameters of the model.
    Modifies the parameters in place.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters will be perturbed.
        std (float): Standard deviation of the Gaussian noise.
    """
    counter = 0
    for param in model.parameters():
        noise = torch.randn_like(param) * std
        param.data.add_(noise)
        counter += 1
    print(f"!! Successfully added noise to {counter} parameters using standard deviation of {std}!")


@torch.no_grad()
def attenuate_model_parameters(model: torch.nn.Module, attenuation_factor: float):
    """
    Attentuates model parameters by weighting each parameter by a factor of 'attenuation_factor'.
    Modifies the parameters in place.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters will be perturbed.
        attenuation_factor (float): Attentuate the weights by multiplying by the attenuation factor.
    """
    counter = 0
    for param in model.parameters():
        param.data.multiply_(attenuation_factor)
        counter += 1
    print(f"!! Successfully attentuated {counter} parameters using an attenuation factor of {attenuation_factor}!")


@torch.no_grad()
def drop_model_parameters(model: torch.nn.Module, dropout_prob: float):
    """
    Drop model parameters with a probability specified by the dropout probability.
    Modifies the parameters in place.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters will be perturbed.
        dropout_prob (float): Weight dropout probability.
    """
    total_params = 0
    retained_params = 0
    for param in model.parameters():
        retain_mask = (torch.rand_like(param.data) > dropout_prob).to(param.data.dtype)
        param.data.multiply_(retain_mask)
        total_params += np.prod(retain_mask.shape)
        retained_params += int(retain_mask.sum())
    print(f"!! Successfully dropped parameters using a dropout prob of {dropout_prob} / total params: {total_params} / retained params: {retained_params}")


def compute_fisher_information_matrix_diagonal(model: torch.nn.Module, loss_fn: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                                               device: torch.device, amp_dtype: torch.dtype = None,
                                               grad_scaler: torch.cuda.amp.grad_scaler.GradScaler = None,
                                               average_over_batches: bool = False) -> Dict[str, torch.Tensor]:
    """
    Reference implementations:
        https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb
        https://github.com/kuc2477/pytorch-ewc/blob/master/model.py
        https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
    """
    # Iterate over all examples in the dataset to compute the gradient w.r.t. the loss function
    if torch.distributed.is_initialized():
        # TODO: add supprt for distributed models -- need to add appropriate reductions to the model
        raise NotImplementedError("Distributed support for fisher information matrix computation not implemented")

    # Assumes that gradients are already averaged within batch (default for most loss functions)
    print("!! Computing the fisher information matrix...")
    start_time = time.time()
    model.eval()  # set the model to eval mode

    # Initialize the Fisher values
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:  # only focus on parameters that require gradient
            fisher_dict[name] = torch.zeros_like(param.data)

    for (images, labels) in tqdm(dataloader):
        # Remove previous gradient information
        model.zero_grad()

        # Forward prop through the model and compute the loss (w/ AMP)
        with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype):
            logits = model(images.to(device))
            loss = loss_fn(logits, labels.to(device))

        # Backward pass: unscale if using grad_scaler
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            # Make sure to unscale before reading the raw grads
            grad_scaler.unscale_(torch.optim.SGD(model.parameters(), lr=1.0))  # Dummy optimizer
        else:
            loss.backward()

        with torch.no_grad():
            # Accumulate per-parameter squared gradient
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += (param.grad.data.clone().pow(2))

    if average_over_batches:  # convert sum to average over batches
        with torch.no_grad():
            num_batches = len(dataloader)
            for name in fisher_dict.keys():
                fisher_dict[name] /= num_batches

    time_elapsed_h = (time.time() - start_time) / (60 * 60)  # convert seconds into hours
    print(f"Fished information matrix computation finished. Total time elapsed: {time_elapsed_h:.2f}h")

    return fisher_dict


def filter_trainable(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if
            k.endswith(".weight") or k.endswith(".bias")}


def flatten_param_dict(params: Iterable[torch.Tensor] = None, param_dict: Dict[str, torch.Tensor] = None) -> torch.Tensor:
    assert (params is None) != (param_dict is None), "either the params or the param_dict should be specified"
    vec = []
    if param_dict is not None:
        for name, tensor in param_dict.items():
            vec.append(tensor.view(-1))
    else:
        assert params is not None
        for tensor in params:
            vec.append(tensor.view(-1))
    return torch.cat(vec)
