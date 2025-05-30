#!/bin/python

import os
import gc
import wget
import time
import json
import copy
import random
import tarfile
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
from itertools import cycle
from packaging import version
from collections import Counter
from typing import Tuple, List, Dict, Any

import wandb

import torch
import numpy as np

from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets import CIFAR10, CIFAR100
# from torchvision.models.resnet import resnet18
import torchvision
from torchvision import transforms

from vision_models import ResNet18, ResNet34
from dataset_utils import get_dataloader
from plot_utils import plot_histogram
from train_utils import get_num_model_params, get_optimizer, get_lr_scheduler
from dist_utils import is_main_proc, init_distributed_env, reduce_tensor, gather_tensor, convert_to_distributed, wait_for_other_procs
from model_utils import add_gaussian_noise_to_parameters, attenuate_model_parameters, drop_model_parameters, flatten_param_dict, \
    filter_trainable, compute_fisher_information_matrix_diagonal


class CrossEntropyLoss(torch.nn.Module):
    """Supports real-valued targets in comparison to the default PyTorch implementation"""
    def __init__(self, apply_log_softmax: bool = True):
        super().__init__()
        self.apply_log_softmax = apply_log_softmax

    def forward(self, pred_log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred_log_probs.shape == target.shape, f"{pred_log_probs.shape} != {target.shape}"
        if self.apply_log_softmax:
            pred_log_probs = torch.nn.functional.log_softmax(pred_log_probs, dim=-1)
        loss = - (target * pred_log_probs).sum(dim=-1).mean()  # B x C
        return loss


class MislabeledWrapper(torch.utils.data.Dataset):
    """Replaces the labels for the given dataset with random labels"""
    def __init__(self, dataset: torch.utils.data.Dataset, num_classes: int, seed: int, target_class: str = "any"):
        super().__init__()
        self.dataset = dataset
        if target_class == "any":
            self.random_labels = np.random.default_rng(seed).integers(0, num_classes, size=(len(dataset),))
        else:
            target_class = int(target_class)  # convert the class to int
            self.random_labels = [target_class for _ in range(len(dataset))]
        print("Assigned labels for mislabeled examples:", Counter(self.random_labels))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]
        label = self.random_labels[idx]  # assign a random label instead of the original label
        return img, label


class TensorDatasetWithTransform(torch.utils.data.Dataset):
    """Similar to the default torch.utils.data.TensorDataset, with the additional support for transforms"""
    def __init__(self, data: torch.Tensor, labels: List[int], transforms: Any):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.data[idx], self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label


class ClassMixDataset(torch.utils.data.Dataset):
    """
    Allows mixing instances of one dataset, but using all the instances of the specified class from another dataset
    """
    def __init__(self, primary_dataset: torch.utils.data.Dataset, mixing_dataset: torch.utils.data.Dataset, mixing_class: int):
        super().__init__()
        self.primary_dataset = primary_dataset
        self.mixing_dataset = mixing_dataset
        self.mixing_class = mixing_class
        self.mixed_indices = None  # each element is a tuple (source_idx, index_within_source)
        self._mix_datasets()

    def _mix_datasets(self):
        # Collect all labels
        primary_labels = np.array([batch[1] for batch in self.primary_dataset])
        mixing_ds_labels = np.array([batch[1] for batch in self.mixing_dataset])

        # Collect all the relevant indices
        primary_mask = primary_labels != self.mixing_class
        mixing_mask = mixing_ds_labels == self.mixing_class
        print(f"Primary ds: {len(primary_mask)}; selected: {primary_mask.sum()} / mixing ds: {len(mixing_mask)}; selected: {mixing_mask.sum()}")

        # Collect the final indices
        self.mixed_indices = [(0, i) for i in np.where(primary_mask)[0]]  # ds_idx is 0
        self.mixed_indices += [(1, i) for i in np.where(mixing_mask)[0]]  # ds_idx is 1

    def __len__(self) -> int:
        return len(self.mixed_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ds_idx, idx_within_ds = self.mixed_indices[idx]
        if ds_idx == 0:
            img, label = self.primary_dataset[idx_within_ds]
            assert label != self.mixing_class, f"{label} == {self.mixing_class}"
        else:
            assert ds_idx == 1, ds_idx
            img, label = self.mixing_dataset[idx_within_ds]
            assert label == self.mixing_class, f"{label} != {self.mixing_class}"
        return img, label


def train_direct(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.LRScheduler, train_steps: int,
                 eval_after_steps: int, gradient_accumulation_steps: int, device: torch.device, amp_dtype: torch.dtype,
                 grad_scaler: torch.amp.grad_scaler.GradScaler, clip_grad_norm: float, checkpoint_file: str, num_classes: int,
                 training_phase: str = "training", additional_loaders: Dict[str, torch.utils.data.DataLoader] = None):
    pbar = None
    if is_main_proc():
        pbar = tqdm(total=train_steps)

    epoch = 0
    iterator = 0  # counts the number of iterations for the loop
    train_step = 0  # counts the optimization steps
    last_eval_step = None
    training_completed = False
    start_time = time.time()

    model.train()
    optimizer.zero_grad()
    loss_fn = torch.nn.CrossEntropyLoss()

    while True:  # restart at the end of trainer
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            print(f"Setting sampler epoch: {epoch}")
            train_loader.sampler.set_epoch(epoch)

        for (images, labels) in train_loader:
            # Forward prop through the model and compute the loss (w/ AMP)
            with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype):
                logits = model(images.to(device))
                loss = loss_fn(logits, labels.to(device))

            # Accumulate gradients
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            current_lr = None
            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_last_lr()  # returns the last LR
                assert isinstance(current_lr, list)
                current_lr = current_lr[0]  # get the first element (assuming it should be the same)
                assert isinstance(current_lr, float)

            if iterator % gradient_accumulation_steps == gradient_accumulation_steps - 1:
                if grad_scaler is not None:
                    if clip_grad_norm is not None:
                        # https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
                        grad_scaler.unscale_(optimizer)  # get the gradients in the original scale
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    grad_scaler.step(optimizer)  # won't unscale if already unscaled
                    grad_scaler.update()
                else:
                    if clip_grad_norm is not None:  # clip the gradients before update -- applied on scaled gradients for AMP
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                train_step += 1

                if lr_scheduler is not None:
                    lr_scheduler.step()

            if pbar is not None:
                pbar.set_description(f"Loss: {float(loss):.4f}")
                pbar.update(1)
            if wandb.run is not None:
                output_dict = {f"{training_phase}/loss": float(loss)}
                if current_lr is not None:
                    output_dict[f"{training_phase}/lr"] = current_lr
                wandb.log(output_dict)
            if eval_after_steps is not None and train_step % eval_after_steps == eval_after_steps - 1 and last_eval_step != train_step:
                print("Evaluating model...")
                if test_loader is not None:
                    evaluate_model(model, test_loader, device, f"{training_phase}/test_set", num_classes)
                if additional_loaders is not None:
                    for key, loader in additional_loaders.items():
                        evaluate_model(model, loader, device, f"{training_phase}/{key}", num_classes)
                model.train()
                last_eval_step = train_step
            if train_step >= train_steps:
                print(f"Training completed for {train_steps} steps. Stopping trainer.")
                training_completed = True
                break
            iterator += 1
        if training_completed:
            break
        epoch += 1
    if pbar is not None:
        pbar.close()

    time_elapsed_h = (time.time() - start_time) / (60 * 60)  # convert seconds into hours
    epochs_completed = train_step / len(train_loader)
    print(f"Model training finished / time elapsed: {time_elapsed_h:.2f}h / epochs completed: {epochs_completed:.2f} (counter: {epoch})")

    # Save the final checkpoint
    if is_main_proc() and checkpoint_file is not None:
        torch.save(model.state_dict(), checkpoint_file)
        print("Model state dict saved:", checkpoint_file)


def compute_retain_loss(model: torch.nn.Module, orig_model: torch.nn.Module, device: torch.device, sec_device: torch.device,
                        c_r: float, mse: torch.nn.Module, kl_div: torch.nn.Module, cross_entropy: torch.nn.Module,
                        unlearning_method: str, unlearning_alpha: float, unlearning_gamma: float, unlearning_layer_idxes: List[int],
                        retain_batch: Tuple[torch.tensor], amp_dtype: torch.dtype) -> Tuple[torch.tensor]:
    with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype):
        retain_images = retain_batch[0].to(device)
        retain_labels = retain_batch[1].to(device)
        if unlearning_method in ["catastrophic_forgetting", "gradient_ascent", "alternating_gradient_ascent", "random_relabeling",
                                 "alternating_random_relabeling", "weight_distortion", "weight_attenuation", "ssd", "l1_sparse",
                                 "mode_connectivity", "weight_dropout", "weight_dist_reg"]:  # regular CE loss
            retain_logits = model(retain_images)
            prediction_loss = cross_entropy(retain_logits, retain_labels)
            retain_loss_holder = float(prediction_loss)
            if "random_relabeling" in unlearning_method:
                loss = unlearning_alpha * prediction_loss
            else:
                assert unlearning_method in ["catastrophic_forgetting", "weight_distortion", "weight_attenuation", "weight_dropout", "ssd", "l1_sparse",
                                             "mode_connectivity", "weight_dist_reg"] or "gradient_ascent" in unlearning_method, unlearning_method
                loss = prediction_loss
        elif unlearning_method in ["circuit_breakers", "alternating_circuit_breakers"]:
            with torch.no_grad():
                retain_orig_rep = orig_model(retain_images.to(sec_device), output_hidden_states=True)
            retain_rep = model(retain_images, output_hidden_states=True)
            retain_loss = 0.
            for idx in unlearning_layer_idxes:
                retain_loss = retain_loss + mse(retain_rep[idx].to(device), retain_orig_rep[idx].detach().to(device))
            retain_loss = retain_loss / len(unlearning_layer_idxes)
            retain_loss_holder = float(retain_loss)
            loss = c_r * retain_loss
        elif unlearning_method == "tar":
            with torch.no_grad():
                retain_orig_rep = orig_model(retain_images.to(sec_device), output_hidden_states=True)
            retain_rep = model(retain_images, output_hidden_states=True)
            rep_loss = 0.
            for idx in unlearning_layer_idxes:
                rep_loss = rep_loss + mse(retain_rep[idx].to(device), retain_orig_rep[idx].detach().to(device))
            rep_loss = rep_loss / len(unlearning_layer_idxes)
            retain_logits = model(retain_images)
            prediction_loss = cross_entropy(retain_logits, retain_labels)
            loss = unlearning_alpha * (rep_loss + prediction_loss)
            retain_loss_holder = {"total": float(loss), "rep": float(rep_loss), "pred": float(prediction_loss)}
        elif unlearning_method in ["scrub", "alternating_scrub", "uniform_scrub"]:
            with torch.no_grad():
                retain_orig_probs = torch.nn.functional.softmax(orig_model(retain_images.to(sec_device)), dim=-1)
            retain_logits = model(retain_images)
            kl_div_loss = kl_div(torch.nn.functional.log_softmax(retain_logits, dim=-1), retain_orig_probs.detach().to(device))
            prediction_loss = cross_entropy(retain_logits, retain_labels)
            loss = unlearning_alpha * kl_div_loss + unlearning_gamma * prediction_loss
            retain_loss_holder = {"total": float(loss), "kl_div": float(kl_div_loss), "pred": float(prediction_loss)}
        else:
            raise RuntimeError(f"Unknown unlearning method: {unlearning_method}")
    return loss, retain_loss_holder


def compute_forget_loss(model: torch.nn.Module, orig_model: torch.nn.Module, device: torch.device, sec_device: torch.device,
                        c_u: float, relu: torch.nn.Module, kl_div: torch.nn.Module, cosine_sim: torch.nn.Module,
                        cross_entropy: torch.nn.Module, unlearning_method: str, unlearning_alpha: float, unlearning_gamma: float,
                        unlearning_layer_idxes: List[int], forget_batch: Tuple[torch.tensor], amp_dtype: torch.dtype,
                        num_classes: int, pretrained_model: torch.nn.Module = None) -> Tuple[torch.tensor]:
    with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype):
        forget_images = forget_batch[0].to(device)
        forget_labels = forget_batch[1].to(device)
        if unlearning_method in ["gradient_ascent", "alternating_gradient_ascent", "random_relabeling", "alternating_random_relabeling"]:
            forget_logits = model(forget_images)
            if "gradient_ascent" in unlearning_method:  # negative of the gradient
                prediction_loss = - cross_entropy(forget_logits, forget_labels)
            else:
                assert "random_relabeling" in unlearning_method, unlearning_method
                offsets = torch.randint(1, num_classes, size=forget_labels.shape, device=forget_labels.device)
                random_labels = (forget_labels + offsets) % num_classes
                prediction_loss = cross_entropy(forget_logits, random_labels)
            forget_loss_holder = float(prediction_loss)
            loss = prediction_loss
        elif unlearning_method in ["circuit_breakers", "alternating_circuit_breakers"]:
            with torch.no_grad():
                forget_orig_rep = orig_model(forget_images.to(sec_device), output_hidden_states=True)
            forget_rep = model(forget_images, output_hidden_states=True)
            forget_loss = 0.
            for idx in unlearning_layer_idxes:
                forget_loss = forget_loss + relu(cosine_sim(forget_rep[idx].to(device).flatten(start_dim=1),
                                                            forget_orig_rep[idx].detach().to(device).flatten(start_dim=1))).mean()
            forget_loss = forget_loss / len(unlearning_layer_idxes)
            forget_loss_holder = float(forget_loss)
            loss = c_u * forget_loss
        elif unlearning_method == "tar":
            forget_logits = model(forget_images)
            log_probs = torch.nn.functional.log_softmax(forget_logits, dim=-1)
            probs = torch.nn.functional.softmax(forget_logits, dim=-1)
            entropy = - (probs * log_probs).sum(dim=-1).mean()  # - \sum_{i=1}^{C} p_i \log p_i
            tamper_resistance_loss = -entropy # Minimize negative entropy i.e., maximize entropy
            forget_loss_holder = float(tamper_resistance_loss)
            loss = tamper_resistance_loss
        elif unlearning_method == "mode_connectivity":
            """
            Unlearning alpha defines the weight on the mode connectivity loss
            Unlearning batch in this case contains both the retain set examples as well as the forget set examples
            """
            assert pretrained_model is not None
            t = 0.5  # mixing coefficient (0: pretrained model, 1: unlearned model)

            # Interpolate model parameters
            cur_params = dict(model.named_parameters())
            with torch.no_grad():
                pre_params = dict(pretrained_model.named_parameters())
            mixed_params = {
                name: (1. - t) * pre_params[name].detach() + t * cur_params[name]
                for name in pre_params.keys()
            }

            # Interpolate buffers (required for batch-norm layers)
            cur_buffers = dict(model.named_buffers())
            with torch.no_grad():
                pre_buffers = dict(pretrained_model.named_buffers())

            mixed_buffers = {}
            for name, buf_pre in pre_buffers.items():
                buf_cur = cur_buffers[name]
                if torch.is_floating_point(buf_pre):   # Only interpolate floating point buffers
                    mixed_buffers[name] = (1. - t) * buf_pre.detach() + t * buf_cur
                else:  # would mainly comprise of integer buffers such as batch counts
                    mixed_buffers[name] = buf_cur  # keep the current value

            # Forward prop the inputs through the model using the merged state dict
            mixed_state = {**mixed_params, **mixed_buffers}  # merge params and buffers into a single state dict
            forget_logits = torch.func.functional_call(model, mixed_state, (forget_images,))
            mode_conn_loss = - cross_entropy(forget_logits, forget_labels)  # maximize CE on the midpoint

            # Use the specified unlearning gamma as an upper bound for loss maximization
            unlearning_gamma = torch.tensor(unlearning_gamma, dtype=mode_conn_loss.dtype, device=mode_conn_loss.device)
            mode_conn_loss = torch.max(mode_conn_loss, -unlearning_gamma)  # don't maximize beyond the specified unlearning gamma
            forget_loss_holder = float(mode_conn_loss)
            loss = unlearning_alpha * mode_conn_loss
        elif unlearning_method == "weight_dist_reg":
            assert pretrained_model is not None
            eps = 1e-8
            param_diff = flatten_param_dict(params=pretrained_model.parameters()).detach() - flatten_param_dict(params=model.parameters())
            # weight_dist_loss = - torch.linalg.norm(param_diff)  # aim is to maximize the distance
            weight_dist_loss = - torch.sqrt(torch.mean(torch.square(param_diff)) + eps)  # uses mean instead of sum to get a normalized metric
            forget_loss_holder = float(weight_dist_loss)
            loss = unlearning_alpha * weight_dist_loss
        elif unlearning_method == "l1_sparse":
            sparsity_loss = torch.mean(torch.abs(flatten_param_dict(params=model.parameters())))  # mean abs value
            forget_loss_holder = float(sparsity_loss)
            loss = unlearning_alpha * sparsity_loss
        elif unlearning_method in ["scrub", "alternating_scrub", "uniform_scrub"]:
            forget_logits = model(forget_images)
            if unlearning_method == "uniform_scrub":  # use the uniform distribution as the unlearning target instead of just maximizing the distance
                forget_orig_probs = torch.ones_like(forget_logits)
                forget_orig_probs /= forget_orig_probs.shape[-1]  # normalize w.r.t. the number of classes
                loss = kl_div(torch.nn.functional.log_softmax(forget_logits, dim=-1), forget_orig_probs)
            else:
                with torch.no_grad():
                    forget_orig_probs = torch.nn.functional.softmax(orig_model(forget_images.to(sec_device)), dim=-1)
                loss = - kl_div(torch.nn.functional.log_softmax(forget_logits, dim=-1), forget_orig_probs.detach().to(device))
            forget_loss_holder = float(loss)
        else:
            raise RuntimeError(f"Unknown unlearning method: {unlearning_method}")
    return loss, forget_loss_holder


def train(model: torch.nn.Module, orig_model: torch.nn.Module, forget_loader: torch.utils.data.DataLoader,
          retain_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, train_steps: int, eval_after_steps: int, unlearning_method: str,
          unlearning_alpha: float, unlearning_gamma: float, unlearning_layer_idxes: List[int],
          gradient_accumulation_steps: int, device: torch.device, sec_device: torch.device,
          amp_dtype: torch.dtype, grad_scaler: torch.amp.grad_scaler.GradScaler, clip_grad_norm: float,
          checkpoint_file: str, num_classes: int, training_phase: str = "unlearning",
          pretrained_model: torch.nn.Module = None, args: Namespace = None) -> None:
    pbar = None
    if is_main_proc():
        pbar = tqdm(total=train_steps)

    forget_epoch = 0
    retain_epoch = 0
    iterator = 0  # counts the number of iterations for the loop
    train_step = 0  # counts the optimization steps
    last_eval_step = None
    current_cycle = "retrain" if "alternating_" in unlearning_method else None  # switched to unlearn before the first step
    start_time = time.time()

    model.train()
    orig_model.train()  # keep in train mode as the representations are misaligned at the start otherwise due to BN

    mse = torch.nn.MSELoss()
    relu = torch.nn.ReLU()
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    cross_entropy = torch.nn.CrossEntropyLoss()
    kl_div = torch.nn.KLDivLoss()

    if unlearning_method == "weight_distortion":
        # Distort weights of model via additive Gaussian noise with an standard deviation of alpha
        add_gaussian_noise_to_parameters(model, unlearning_alpha)  # modifies the model in-place
    elif unlearning_method == "weight_attenuation":
        # Distort weights of model via weight attenuation with an attenuation factor of alpha
        attenuate_model_parameters(model, unlearning_alpha)  # modifies the model in-place
    elif unlearning_method == "weight_dropout":
        # Distort weights of model via weight dropout with a dropout prob of alpha
        drop_model_parameters(model, unlearning_alpha)  # modifies the model in-place
    elif unlearning_method == "ssd":
        # Compute the diagonal of the Fisher information matrix
        print("!! Computing Fisher information matrix diagonal on the retain set and the forget set to identify parameter importance...")
        retain_fisher_diagonal_dict = compute_fisher_information_matrix_diagonal(model, cross_entropy, retain_loader, device, amp_dtype, grad_scaler)
        forget_fisher_diagonal_dict = compute_fisher_information_matrix_diagonal(model, cross_entropy, forget_loader, device, amp_dtype, grad_scaler)

        # Plot the histogram of fisher values
        xlabel = "Log Fisher diagonal magnitude"
        ylabel = "Log frequency"
        plot_kwargs = dict(xlabel=xlabel, ylabel=xlabel, xscale='log', yscale='log')
        plot_histogram(flatten_param_dict(param_dict=retain_fisher_diagonal_dict), f"{unlearning_method}_retain_fisher_hist.png", **plot_kwargs)
        plot_histogram(flatten_param_dict(param_dict=forget_fisher_diagonal_dict), f"{unlearning_method}_forget_fisher_hist.png", **plot_kwargs)

        with torch.no_grad():
            # Compute the parameter importance as the ratio of the retain and the forget set fisher info
            counter = 0
            beta_vec = []
            for param_name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # no fisher information available

                retain_fisher_diagonal = retain_fisher_diagonal_dict[param_name]
                forget_fisher_diagonal = forget_fisher_diagonal_dict[param_name]

                # β = min( λ[]D,i / []Df,i , 1) -> lambda is assumed to be 1
                eps = 1e-6
                beta = torch.clamp(retain_fisher_diagonal / (forget_fisher_diagonal + eps), min=0., max=1.)  # clamp between 0 and 1
                beta_vec.append(beta.clone().view(-1))

                mask_retain_imp_weights = False
                if mask_retain_imp_weights:
                    # Dampen parameters based on the computed beta value
                    retain_mask = forget_fisher_diagonal <= unlearning_alpha * retain_fisher_diagonal  # []Df,i should be > alpha * []Dr,i
                    beta[retain_mask] = 1.  # no dampening of parameters that are important for the retain set

                # Apply dampening
                param.data.multiply_(beta)  # attenuate by a factor of beta
                if mask_retain_imp_weights:
                    num_dampened_parameters = np.prod(retain_mask.shape) - int(torch.sum(retain_mask))
                else:
                    num_dampened_parameters = torch.sum(1. - beta)
                counter += num_dampened_parameters
            print(f"!! Successfully dampened {counter} parameters for SSD!")

            # Plot the beta distribution
            plot_histogram(torch.cat(beta_vec), f"{unlearning_method}_beta_hist.png", "Log beta val", ylabel, xscale='log', yscale='log')

            del retain_fisher_diagonal_dict
            del forget_fisher_diagonal_dict
            del beta_vec
            torch.cuda.empty_cache()
            gc.collect()

    # Initialize the optimizer here after resetting the model weights
    optimizer = get_optimizer(model, lr=args.unlearning_learning_rate, wd=args.unlearning_weight_decay, optimizer_name=args.optimizer_name)
    optimizer.zero_grad()

    forget_iter = cycle(forget_loader)
    forget_iter_size = len(forget_loader)
    retain_iter = cycle(retain_loader)
    retain_iter_size = len(retain_loader)
    forget_iterations_enabled = unlearning_method not in ["catastrophic_forgetting", "weight_distortion", "weight_attenuation",
                                                          "weight_dropout", "ssd"]

    while True:
        # Set the dataloader and iterators correctly
        if current_cycle is None:  # regular training
            if iterator % forget_iter_size == 0:  # first step
                if hasattr(forget_loader, "sampler") and isinstance(forget_loader.sampler, DistributedSampler):
                    print(f"Setting sampler epoch for the forget loader: {forget_epoch}")
                    forget_loader.sampler.set_epoch(forget_epoch)
                    forget_epoch += 1
            if iterator % retain_iter_size == 0:  # first step
                if hasattr(retain_loader, "sampler") and isinstance(retain_loader.sampler, DistributedSampler):
                    print(f"Setting sampler epoch for the retain loader: {retain_epoch}")
                    retain_loader.sampler.set_epoch(retain_epoch)
                    retain_epoch += 1
        else:
            assert current_cycle in ["unlearn", "retrain"], current_cycle
            if current_cycle == "retrain" and iterator % retain_iter_size == 0:  # first step
                if hasattr(retain_loader, "sampler") and isinstance(retain_loader.sampler, DistributedSampler):
                    print(f"Setting sampler epoch for the retain loader: {retain_epoch}")
                    retain_loader.sampler.set_epoch(retain_epoch)
                    retain_epoch += 1

                # reset iterator and gradient
                iterator = 0
                optimizer.zero_grad()
                current_cycle = "unlearn"
                print(f"Cycle switched to: {current_cycle}")

            elif current_cycle == "unlearn" and iterator % forget_iter_size == 0:  # first step
                if hasattr(forget_loader, "sampler") and isinstance(forget_loader.sampler, DistributedSampler):
                    print(f"Setting sampler epoch for the forget loader: {forget_epoch}")
                    forget_loader.sampler.set_epoch(forget_epoch)
                    forget_epoch += 1

                # reset iterator and gradient
                iterator = 0
                optimizer.zero_grad()
                current_cycle = "retrain"
                print(f"Cycle switched to: {current_cycle}")

        # Define weighting factors for circuit breakers
        c_u_schedule = "complete"  # circuit breakers paper used half schedule for LLMs with a high alpha
        assert c_u_schedule in ["half", "complete"]
        normalizer = (2 * train_steps) if c_u_schedule == "half" else train_steps
        c_r = unlearning_alpha * (train_step + 1) / normalizer
        c_u = unlearning_alpha * (1. - (train_step + 1) / normalizer)
        if "alternating_" in unlearning_method:
            c_u = unlearning_alpha  # directly define as alpha
            c_r = 1.0

        # Define default loss values for logging
        total_loss = 0.
        if unlearning_method in ["scrub", "alternating_scrub", "uniform_scrub"]:
            retain_loss_holder = {"total": -1., "kl_div": -1., "pred": -1.}
        else:
            retain_loss_holder = -1.
        forget_loss_holder = -1.

        if unlearning_method == "tar":
            """
            Things to consider for TAR:
            (i) How many attackers and attack steps? Paper used 4 steps
            (ii) Should we use a fresh batch of data during the attack? That's what we are doing right now
            (iii) Should we use a fresh batch of data when computing the temper resistant loss? That's what we are doing right now
            (iv) What's the correct gamma which weights the temper resistence gradients? Paper used 3/4
            """
            num_adversaries = 4  # adjustable hyperparameter (K from Tar paper)
            attacker_steps_per_adversary = 16  # how many fine-tuning steps each adversary does
            inner_tamper_gradients = None  # will accumulate the total tamper-resistance gradient over all adversaries

            # Taken from the official implementation: https://github.com/rishub-tamirisa/tamper-resistance/blob/main/tar.py
            use_weighting_schedule = False  # schedule on the gradient accummulation for temper resistance loss
            schedule_lambda = 0.5
            sample_optim_params = False
            batch_sampling_mechanism = "adv"
            assert batch_sampling_mechanism in ["adv", "step", "always"]  # 'always' refers to sampling of at every loss computation step

            for _ in range(num_adversaries):
                # Define adversary HPs
                adversary_lr = random.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3]) if sample_optim_params else 1e-3
                adversary_wd = random.choice([0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]) if sample_optim_params else 0.

                if batch_sampling_mechanism == "adv":
                    forget_batch = next(forget_iter)  # sample a new batch for optimization

                # Clone model for adversarial updates
                adversary_model = copy.deepcopy(model)
                adversary_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=adversary_lr, weight_decay=adversary_wd)

                for step in range(attacker_steps_per_adversary):
                    if batch_sampling_mechanism in ["step", "always"]:
                        forget_batch = next(forget_iter)  # sample a new batch for optimization

                    # Attacker attacks by minimizing the CE loss on the forget labels -- retain loss with CF does gradient descent
                    adv_loss, _ = compute_retain_loss(adversary_model, orig_model, device, sec_device, c_r, mse, kl_div, cross_entropy,
                                                      "catastrophic_forgetting", unlearning_alpha, unlearning_gamma, unlearning_layer_idxes,
                                                      forget_batch, amp_dtype)

                    # Update the adversary model
                    adversary_optimizer.zero_grad()
                    adv_loss.backward()
                    adversary_optimizer.step()

                    # Now compute the TAR tamper-resistance loss on adversarially fine-tuned model (defender maximizes entropy on the forget set)
                    if batch_sampling_mechanism == "always":
                        forget_batch = next(forget_iter)  # sample a new batch for optimization
                    tamper_loss, forget_loss_holder = compute_forget_loss(adversary_model, orig_model, device, sec_device, c_u, relu, kl_div,
                                                                          cosine_sim, cross_entropy, "tar", unlearning_alpha, unlearning_gamma,
                                                                          unlearning_layer_idxes, forget_batch, amp_dtype, num_classes)
                    adversary_optimizer.zero_grad()
                    tamper_loss.backward()

                    # Taken from the official implementation: https://github.com/rishub-tamirisa/tamper-resistance/blob/main/tar.py
                    scheduled_weighting = (float(torch.exp(schedule_lambda * (torch.tensor(step) - (attacker_steps_per_adversary - 1))))
                                           if use_weighting_schedule else 1 / attacker_steps_per_adversary)

                    # Accumulate gradients correctly by normalizing by the number of adversaries and the number of optimization steps
                    grads_current_step = [((scheduled_weighting / num_adversaries) * p.grad.clone()) if p.grad is not None
                                          else torch.zeros_like(p) for p in adversary_model.parameters()]
                    if inner_tamper_gradients is None:
                        inner_tamper_gradients = grads_current_step
                    else:
                        for acc_grad, this_grad in zip(inner_tamper_gradients, grads_current_step):
                            acc_grad.add_(this_grad)

            # Reset gradients and compute retain loss
            optimizer.zero_grad()
            retain_loss, retain_loss_holder = compute_retain_loss(model, orig_model, device, sec_device, c_r, mse, kl_div, cross_entropy,
                                                                  unlearning_method, unlearning_alpha, unlearning_gamma,
                                                                  unlearning_layer_idxes, next(retain_iter), amp_dtype)
            retain_loss.backward()

            # Add scaled tamper-resistance gradients (retain loss is already scaled by unlearning_alpha)
            for param, tamper_grad in zip(model.parameters(), inner_tamper_gradients):
                if param.grad is None:
                    param.grad = unlearning_gamma * tamper_grad
                else:
                    param.grad.add_(unlearning_gamma * tamper_grad)

            # Compute total loss for logging
            total_loss = float(retain_loss) + float(tamper_loss)

        else:
            retain_batch = None
            if current_cycle is None or current_cycle == "retrain":
                # Iteration on the retain batch
                retain_batch = next(retain_iter)
                loss, retain_loss_holder = compute_retain_loss(model, orig_model, device, sec_device, c_r, mse, kl_div, cross_entropy,
                                                               unlearning_method, unlearning_alpha, unlearning_gamma, unlearning_layer_idxes,
                                                               retain_batch, amp_dtype)
                total_loss += float(loss)  # include retain loss

                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

            if forget_iterations_enabled and (current_cycle is None or current_cycle == "unlearn"):
                # Iteration on the forget batch
                forget_batch = next(forget_iter)
                if unlearning_method == "mode_connectivity":  # concatenate both the retain and the forget batches for loss maximization at midpoint
                    assert retain_batch is not None
                    assert len(retain_batch) == 2, len(retain_batch)
                    forget_batch = (torch.cat([retain_batch[0], forget_batch[0]], dim=0), torch.cat([retain_batch[1], forget_batch[1]], dim=0))
                loss, forget_loss_holder = compute_forget_loss(model, orig_model, device, sec_device, c_u, relu, kl_div, cosine_sim, cross_entropy,
                                                               unlearning_method, unlearning_alpha, unlearning_gamma, unlearning_layer_idxes,
                                                               forget_batch, amp_dtype, num_classes, pretrained_model=pretrained_model)
                total_loss += float(loss)  # add forget loss

                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

        if iterator % gradient_accumulation_steps == gradient_accumulation_steps - 1:
            if grad_scaler is not None:
                if clip_grad_norm is not None:
                    # https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
                    grad_scaler.unscale_(optimizer)  # get the gradients in the original scale
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                grad_scaler.step(optimizer)  # won't unscale if already unscaled
                grad_scaler.update()
            else:
                if clip_grad_norm is not None:  # clip the gradients before update -- applied on scaled gradients for AMP
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            train_step += 1

        if pbar is not None:
            if unlearning_method in ["catastrophic_forgetting", "random_relabeling", "alternating_random_relabeling", "gradient_ascent",
                                     "alternating_gradient_ascent", "circuit_breakers", "alternating_circuit_breakers", "weight_distortion",
                                     "weight_attenuation", "weight_dropout", "ssd", "l1_sparse", "mode_connectivity", "weight_dist_reg"]:
                retain_str = f"{retain_loss_holder:.4f}"
            elif unlearning_method in ["scrub", "alternating_scrub", "uniform_scrub", "tar"]:
                retain_str = "{" + ', '.join([f"{x}: {y:.4f}" for x, y in retain_loss_holder.items()]) + "}"
            else:
                raise RuntimeError(f"Unknown unlearning method: {unlearning_method}")
            pbar.set_description(f"Loss: {total_loss:.4f} / Retain loss: {retain_str} / Forget loss: {forget_loss_holder:.4f}")
            pbar.update(1)
        if wandb.run is not None:
            output_dict = {"loss": total_loss, "retain_loss": retain_loss_holder, "forget_loss": forget_loss_holder}
            if "circuit_breakers" in unlearning_method:
                output_dict.update({"c_u": c_u, "c_r": c_r})
            wandb.log({f"{training_phase}/{k}": v for k, v in output_dict.items()})
        if eval_after_steps is not None and train_step % eval_after_steps == eval_after_steps - 1 and last_eval_step != train_step:
            print("Evaluating model...")
            _, retain_acc = evaluate_model(model, retain_loader, device, f"{training_phase}/retain_set", num_classes)
            _, forget_acc = evaluate_model(model, forget_loader, device, f"{training_phase}/forget_set", num_classes)
            _, test_acc = evaluate_model(model, test_loader, device, f"{training_phase}/test_set", num_classes)
            tow = (retain_acc) * (1. - forget_acc) * test_acc
            if wandb.run is not None:
                wandb.log({f"{training_phase}/tow": tow})
            model.train()
            last_eval_step = train_step
        if train_step >= train_steps:
            print(f"Training completed for {train_steps} steps. Stopping trainer.")
            break
        iterator += 1
    if pbar is not None:
        pbar.close()

    time_elapsed_h = (time.time() - start_time) / (60 * 60)  # convert seconds into hours
    print(f"Model training finished / time elapsed: {time_elapsed_h:.2f}h / retain epochs: {retain_epoch-1} / forget epochs: {forget_epoch-1}")

    # Save the final checkpoint
    if is_main_proc() and checkpoint_file is not None:  # Save the final model
        torch.save(model.state_dict(), checkpoint_file)
        print("Model state dict saved:", checkpoint_file)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, device: torch.device, split_name: str,
                   num_classes: int = None) -> Tuple[float, float]:
    model.eval()
    avg_loss = 0.
    num_ex = 0
    all_correct = 0
    if num_classes is not None:
        class_correct = {k: 0 for k in range(num_classes)}
        class_total = {k: 0 for k in range(num_classes)}
    loss_fn = torch.nn.CrossEntropyLoss()

    for (images, labels) in tqdm(eval_loader):
        labels = labels.to(device)
        logits = model(images.to(device))
        loss = loss_fn(logits, labels)
        pred = torch.argmax(logits, dim=-1)
        assert pred.shape == labels.shape, f"{pred.shape} != {labels.shape}"
        correct = pred == labels

        avg_loss += float(loss)
        num_ex += len(images)
        all_correct += int((correct).sum())
        if num_classes is not None:
            for cls in range(num_classes):
                label_mask = labels == cls
                class_correct[cls] += int(correct[label_mask].sum())
                class_total[cls] += int(label_mask.int().sum())

    # Collect the stats from all processes
    avg_loss = float(reduce_tensor(torch.tensor(avg_loss).to(device)))
    all_correct = int(reduce_tensor(torch.tensor(all_correct).to(device)))
    num_ex = int(reduce_tensor(torch.tensor(num_ex).to(device)))
    if num_classes is not None:
        for k in range(num_classes):
            class_correct[k] = int(reduce_tensor(torch.tensor(class_correct[k]).to(device)))
            class_total[k] = int(reduce_tensor(torch.tensor(class_total[k]).to(device)))

    avg_loss = avg_loss / num_ex
    acc = all_correct / num_ex
    output_dict = {"split": split_name, "num_ex": num_ex, "correct": all_correct, "avg_loss": avg_loss, "accuracy": acc}

    acc_table = None
    if num_classes is not None:
        class_acc = {k: (float(class_correct[k]) / class_total[k]) if class_total[k] > 0 else -1. for k in range(num_classes)}
        output_dict["class_accuracy"] = class_acc

        columns = ["Class", "Correct", "Total", "Accuracy"]
        acc_table = wandb.Table(columns=columns)
        for k in range(num_classes):
            acc_table.add_data(str(k), class_correct[k], class_total[k], class_acc[k])
        acc_table.add_data("mean", all_correct, num_ex, acc)
        cls_total = sum([class_total[k] for k in range(num_classes)])
        assert num_ex == cls_total, f"{num_ex} != {cls_total}"

    print(json.dumps(output_dict))
    if split_name is not None and wandb.run is not None:
        wandb.log({f"{split_name}": {"num_ex": num_ex, "correct": all_correct, "avg_loss": avg_loss, "accuracy": acc}})
        if num_classes is not None:
            assert acc_table is not None
            wandb.log({f"{split_name}_acc_table": acc_table})
    return avg_loss, acc


def mix_state_dicts(state_dict_a: Dict[str, torch.Tensor], state_dict_b: Dict[str, torch.Tensor], mixing_coeff: float) \
    -> Dict[str, torch.Tensor]:
    assert 0. <= mixing_coeff <= 1., f"Invalid mixing coeff: {mixing_coeff}"
    output_dict = {}
    for k in state_dict_a.keys():
        output_dict[k] = ((1. - mixing_coeff) * state_dict_a[k]) + (mixing_coeff * state_dict_b[k])
    return output_dict


def compress_file_name(file_name: str) -> str:
    file_name = file_name.replace("_unlearning_method_", "_um_")
    file_name = file_name.replace("_mode_connectivity_", "_mode_conn_")
    file_name = file_name.replace("_circuit_breakers_", "_circuit_b_")
    file_name = file_name.replace("_4_16_adamw_firord_adv_", "_def_")
    file_name = file_name.replace("_random_", "_rand_")
    file_name = file_name.replace("_new_cu_sched_", "_")
    return file_name


def terminate_process():
    if wandb.run is not None:
        wandb.finish()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    exit()


def main(args: Namespace):
    init_distributed_env(args)

    generator = None
    if args.seed is not None:  # Set process seed to reduce stochasticity
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
        print("Setting process seed:", args.seed)

        # Generator to seed dataloaders
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    base_name = f"{args.dataset_name}_{args.model_name}"
    checkpoint_dir = "checkpoints"
    if is_main_proc() and not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        print("Checkpoint directory created:", checkpoint_dir)

    suffix = f"steps_{args.train_steps}" if args.train_steps is not None else f"ep_{args.train_epochs}"
    suffix += f"_bs_{args.batch_size}" + (f"_eff_{args.effective_batch_size}" if args.batch_size != args.effective_batch_size else "")
    suffix += f"_lr_{args.learning_rate}"
    suffix += f"_sched_{args.lr_scheduler}" if args.lr_scheduler is not None else ""
    suffix += f"_wd_{args.weight_decay}" if args.weight_decay > 0 else ""
    suffix += f"{('_amp_' + args.amp_dtype + ('_grad_sc_' if args.use_grad_scaler else ''))}" if args.amp_dtype is not None else ""
    suffix += f"_clip_{args.clip_grad_norm}" if args.clip_grad_norm is not None else ""
    suffix += f"_canaries_{args.fraction_of_canaries}" if (args.fraction_of_canaries is not None and args.fraction_of_canaries > 0.) else ""
    suffix_wo_unlearning = f"{suffix}"

    unlearning_method_args = f"{args.unlearning_method.replace('alternating_', 'alt_')}_lr_{args.unlearning_learning_rate}"
    unlearning_method_args += f"_wd_{args.unlearning_weight_decay}" if args.unlearning_weight_decay > 0 else ""
    if "circuit_breakers" in args.unlearning_method:
        sched = '_new_cu_sched' if 'alternating_' not in args.unlearning_method else ''
        unlearning_method_args += f"_a_{args.unlearning_alpha}_l_{args.unlearning_layer_idx}{sched}"
    elif "scrub" in args.unlearning_method:
        unlearning_method_args += f"_a_{args.unlearning_alpha}_g_{args.unlearning_gamma}"
    elif "random_relabeling" in args.unlearning_method:
        unlearning_method_args += f"_a_{args.unlearning_alpha}"
    elif args.unlearning_method == "weight_distortion":
        unlearning_method_args += f"_std_{args.unlearning_alpha}"
    elif args.unlearning_method == "weight_attenuation":
        unlearning_method_args += f"_atten_fac_{args.unlearning_alpha}"
    elif args.unlearning_method == "weight_dropout":
        unlearning_method_args += f"_prob_{args.unlearning_alpha}"  # dropout prob
    elif args.unlearning_method == "ssd":
        assert args.unlearning_alpha >= 1., f"Alpha that defines the comparison should be >= 1. Found: {args.unlearning_alpha}"
        unlearning_method_args += f"_a_{args.unlearning_alpha}"
    elif args.unlearning_method == "l1_sparse":
        unlearning_method_args += f"_a_{args.unlearning_alpha}"  # weight on the sparsity term
    elif args.unlearning_method == "tar":
        assert args.initial_safeguard_args is not None
        unlearning_method_args += f"_a_{args.unlearning_alpha}_g_{args.unlearning_gamma}"  # retain loss weight + gradient scaling factor
        unlearning_method_args += f"_l_{args.unlearning_layer_idx}"  # layers to apply the retain loss
        unlearning_method_args += f"_init_sg_{args.initial_safeguard_args}"
    elif args.unlearning_method == "mode_connectivity":
        unlearning_method_args += f"_state_r+f_a_{args.unlearning_alpha}_g_{args.unlearning_gamma}"  # mode connectivity maximization weight / loss upper bound
        if args.initial_safeguard_args is not None:  # init sg is optional
            unlearning_method_args += f"_init_sg_{args.initial_safeguard_args}"
    elif args.unlearning_method == "weight_dist_reg":
        unlearning_method_args += f"_a_{args.unlearning_alpha}"  # weight on the distance regularization term
        if args.initial_safeguard_args is not None:  # init sg is optional
            unlearning_method_args += f"_init_sg_{args.initial_safeguard_args}"
    elif args.unlearning_method in ["catastrophic_forgetting", "gradient_ascent", "alternating_gradient_ascent"]:
        pass  # no args
    else:
        raise RuntimeError(f"Unknown unlearning method: {args.unlearning_method}")
    suffix += f"_unlearning_method_{unlearning_method_args}"  # integrate the method args

    suffix += f"_frac_{args.fraction_to_unlearn}_tgt_{args.unlearning_target_class}_crit_{args.unlearning_example_selection_criterion}"
    suffix += f"_steps_{args.unlearning_steps}" if args.unlearning_steps is not None else f"_ep_{args.unlearning_epochs}"
    suffix += f"_bs_{args.unlearning_batch_size}" if args.unlearning_batch_size != args.batch_size else ""
    suffix_wo_relearning = f"{suffix}"
    relearn_ex_type_str = f"_{args.relearn_example_type}" if args.relearn_example_type != "retain" else ""
    suffix += f"_relearn{relearn_ex_type_str}"
    suffix += f"_frac_{args.fraction_to_relearn}_crit_{args.relearning_example_selection_criterion}_lr_{args.relearning_learning_rate}"
    suffix += f"_wd_{args.relearning_weight_decay}" if args.relearning_weight_decay > 0 else ""
    suffix += f"_steps_{args.relearning_steps}" if args.relearning_steps is not None else f"_ep_{args.relearning_epochs}"
    args.unlearning_checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_{suffix_wo_relearning}.pth")
    args.relearning_checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_{suffix}.pth")
    args.base_checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_{suffix_wo_unlearning}.pth")
    args.random_init_checkpoint_file = os.path.join(checkpoint_dir, f"random_init_{base_name}_{suffix_wo_unlearning}.pth")
    args.retrain_from_scratch_checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_{suffix_wo_unlearning}_retrain_from_scratch_unlearn_frac_"
                                                             f"{args.fraction_to_unlearn}_crit_{args.unlearning_example_selection_criterion}.pth")
    args.relearn_from_scratch_checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_{suffix_wo_unlearning}_relearn_from_scratch_unlearn_frac_"
                                                             f"{args.fraction_to_unlearn}_crit_{args.unlearning_example_selection_criterion}_relearn"
                                                             f"{relearn_ex_type_str}_frac_{args.fraction_to_relearn}"
                                                             f"_crit_{args.relearning_example_selection_criterion}.pth")
    args.learn_from_scratch_checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_{suffix_wo_unlearning}_learn_from_scratch_unlearn_frac_"
                                                           f"{args.fraction_to_unlearn}_crit_{args.unlearning_example_selection_criterion}_relearn_frac_"
                                                           f"{args.fraction_to_relearn}_crit_{args.relearning_example_selection_criterion}.pth")
    if args.generate_relearning_grid:
        suffix += "_grid"
        suffix += "_all_remaining_eval" if args.use_all_examples_for_eval else ""
        suffix += f"_model_{args.relearn_grid_eval_model}" if args.relearn_grid_eval_model != "both" else ""
        suffix += f"_ex_{args.relearn_grid_eval_ex}" if args.relearn_grid_eval_ex != "limited" else ""

    # Validate that the file name is within limits
    file_name_len_limit = 255
    if len(args.unlearning_checkpoint_file) > file_name_len_limit:
        print(f"Unlearning checkpoint file length exceeded / len: {len(args.unlearning_checkpoint_file)} / {args.unlearning_checkpoint_file}")
        args.unlearning_checkpoint_file = compress_file_name(args.unlearning_checkpoint_file)
        print(f"Compressed unlearning checkpoint file / len: {len(args.unlearning_checkpoint_file)} / {args.unlearning_checkpoint_file}")
    assert len(args.unlearning_checkpoint_file) <= file_name_len_limit, f"{len(args.unlearning_checkpoint_file)} > {file_name_len_limit}"

    if len(args.relearning_checkpoint_file) > file_name_len_limit:
        print(f"Relearning checkpoint file length exceeded / len: {len(args.relearning_checkpoint_file)} / {args.relearning_checkpoint_file}")
        args.relearning_checkpoint_file = compress_file_name(args.relearning_checkpoint_file)
        print(f"Compressed relearning checkpoint file / len: {len(args.relearning_checkpoint_file)} / {args.relearning_checkpoint_file}")
    assert len(args.relearning_checkpoint_file) <= file_name_len_limit, f"{len(args.relearning_checkpoint_file)} > {file_name_len_limit}"

    args.initial_safeguarded_model = None
    if args.unlearning_method in ["tar", "mode_connectivity", "weight_dist_reg"] and args.initial_safeguard_args is not None:  # optional except TAR
        # FIXME: update this if the old models are not trained for a 100 epochs
        initial_safeguarded_model_suffix = suffix_wo_relearning.replace(unlearning_method_args, args.initial_safeguard_args)
        initial_safeguarded_model_suffix = initial_safeguarded_model_suffix.replace(f'_ep_{args.unlearning_epochs}', '_ep_100')  # old models
        args.initial_safeguarded_model = os.path.join(checkpoint_dir, f"{base_name}_{initial_safeguarded_model_suffix}.pth")
        print("Imputed initial safeguards model file:", args.initial_safeguarded_model)
        assert os.path.exists(args.initial_safeguarded_model), f"Unable to locate initial safeguarded model file: {args.initial_safeguarded_model}"

    if args.wandb_project is not None and is_main_proc():
        print("Initialization w&b...")
        args.wandb_run_name = f"{base_name}_{suffix}"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args, resume=False)

    # Load the dataset
    data_dir = f"./Datasets/{args.dataset_name}/"
    num_classes = 10 if args.dataset_name == "cifar10" else 100 if args.dataset_name == "cifar100" else None
    assert num_classes is not None

    scores_file = None
    use_cscores = True  # memorization scores not computed for CIFAR-10
    train_set_wo_transform = None
    if args.dataset_name in ["cifar10", "cifar100"]:
        cifar_train_transform = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.ToTensor()]
        cifar_test_transform = [transforms.ToTensor()]
        DatasetCls = CIFAR10 if args.dataset_name == "cifar10" else CIFAR100 if args.dataset_name == "cifar100" else None
        assert DatasetCls is not None
        train_set = DatasetCls(data_dir, download=True, train=True, transform=transforms.Compose(cifar_train_transform))
        test_set = DatasetCls(data_dir, download=True, train=False, transform=transforms.Compose(cifar_test_transform))
        if args.evaluate_linear_mode_connectivity:
            train_set_wo_transform = DatasetCls(data_dir, download=True, train=True, transform=transforms.Compose(cifar_test_transform))

        # Download the consistency scores
        scores_file = f"{args.dataset_name}-cscores-orig-order.npz"
        url = f"https://pluskid.github.io/structural-regularity/cscores/{scores_file}"
        if not os.path.exists(scores_file):
            scores_file = wget.download(url)
            print("CScores file downloaded:", scores_file)

        # Load the corrupted dataset
        dataset_name = "CIFAR-10" if args.dataset_name == "cifar10" else "CIFAR-100"
        corrupted_ds_path = f"{dataset_name}-C"
        if not os.path.exists(corrupted_ds_path):
            # https://zenodo.org/record/2535967#.YSaPSXUzaEA
            # https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
            # https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
            print(f"Downloading {dataset_name}-C...")
            corrupted_tar_file = f"{dataset_name}-C.tar"
            url_code = "2535967" if args.dataset_name == "cifar10" else "3555552"
            url = f"https://zenodo.org/records/{url_code}/files/{corrupted_tar_file}"
            corrupted_tar_file = wget.download(url)
            print("Corrupted dataset file downloaded:", corrupted_tar_file)

            # Extract tar to get the dataset
            print("Extracting tar file...")
            tar = tarfile.open(corrupted_tar_file)
            tar.extractall()
            tar.close()
            os.remove(corrupted_tar_file)  # remove the downloaded tar file

        noise_models = os.listdir(corrupted_ds_path)
        noise_models = [x for x in noise_models if x != "labels.npy"]
        print(f"Corrupted dataset path: {corrupted_ds_path} / {len(noise_models)} / {noise_models}")

        # Load the correct noised version of the dataset
        selected_noise_type = "jpeg_compression.npy"
        selected_corruption_level = 5
        corruption_levels = range(1, 6)
        assert selected_corruption_level in corruption_levels
        data_labels = np.load(os.path.join(corrupted_ds_path, 'labels.npy'))
        num_images = len(test_set)
        for i in corruption_levels:  # label validation
            assert all([x == y for x, y in zip(test_set.targets, data_labels[i*num_images:(i+1)*num_images])])
        data = np.load(os.path.join(corrupted_ds_path, selected_noise_type))
        start_idx = (selected_corruption_level - 1) * num_images
        end_idx = selected_corruption_level * num_images
        corrupted_test_set = TensorDatasetWithTransform(data[start_idx:end_idx], data_labels[start_idx:end_idx].tolist(),
                                                        transforms=transforms.Compose(cifar_test_transform))
        assert (np.array(corrupted_test_set.labels) == np.array(test_set.targets)).all()

    print(f"Loaded dataset: {args.dataset_name} / # train: {len(train_set)} / # test: {len(test_set)} / corrupted: {len(corrupted_test_set)}")

    if args.dataset_name == "cifar10":
        if use_cscores:
            cscore_data = np.load(scores_file, allow_pickle=True)
            memorization_labels = cscore_data['labels']
            memorization_values = 1. - cscore_data['scores']
        else:
            raise NotImplementedError
    elif args.dataset_name == "cifar100":
        if use_cscores:
            cscore_data = np.load(scores_file, allow_pickle=True)
            memorization_labels = cscore_data['labels']
            memorization_values = 1. - cscore_data['scores']
        else:
            with np.load(scores_file) as data:
                print(list(data.keys()))
                memorization_values = data["tr_mem"]
    print("Loaded memorization values shape:", memorization_values.shape)

    # Load the model
    if args.model_name == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif args.model_name == "resnet34":
        model = ResNet34(num_classes=num_classes)
    else:
        raise RuntimeError(f"Unknown model: {args.model_name}")
    num_model_params = get_num_model_params(model)
    num_model_layers = model.get_num_model_layers()
    print(f"# model params: {num_model_params/1_000_000:.2f}M / # layers: {num_model_layers}")

    if not os.path.exists(args.random_init_checkpoint_file):  # save random init for retrain from scratch
        torch.save(model.state_dict(), args.random_init_checkpoint_file)
        print("Random init checkpoint saved:", args.random_init_checkpoint_file)

    args.unlearning_layer_idx = [int(x) for x in args.unlearning_layer_idx.split(',')]
    print(f"Unlearning config / layer idx: {args.unlearning_layer_idx} / alpha value: {args.unlearning_alpha}")
    assert all([x < num_model_layers for x in args.unlearning_layer_idx]), \
        f"Unlearning layer idx {args.unlearning_layer_idx} should be less than total number of layers {num_model_layers}"

    # Convert to DDP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # move to device
    model = convert_to_distributed(model, args.local_rank, use_ddp=True)  # cast into DDP

    if args.compile_model:
        if version.parse(torch.__version__) > version.parse("2.0"):
            print("Compiling model...")
            model.compile()  # Use the torch compile method`
        else:
            print("[WARNING] Can't compile model for PyTorch version < 2.")

    # Select unlearning examples
    possible_class_targets = ["all"] + [str(x) for x in range(num_classes)]
    assert args.unlearning_target_class in possible_class_targets, f"{args.unlearning_target_class} not in {possible_class_targets}"
    canaries_idx = []
    if args.unlearning_target_class == "all":
        examples_to_unlearn = int(args.fraction_to_unlearn * len(train_set))  # examples at the start are selected
        if args.unlearning_example_selection_criterion == "random":
            permuted_idx = np.random.default_rng(args.seed).permutation(np.arange(len(train_set)))
        elif args.unlearning_example_selection_criterion in ["low_mem", "high_mem"]:
            permuted_idx = np.argsort(memorization_values)  # sort in ascending order
            if args.unlearning_example_selection_criterion == "high_mem":
                permuted_idx = permuted_idx[::-1]  # invert the indices
        else:
            raise RuntimeError(f"Unknown unlearning example selection criterion: {args.unlearning_example_selection_criterion}")
        unlearning_idx = permuted_idx[:examples_to_unlearn]
        if args.fraction_of_canaries is not None:
            # TODO: make it independent of the unlearning selection i.e., by randomly picking instances from the retain set?
            examples_as_canaries = int(args.fraction_of_canaries * len(train_set))
            canaries_idx = permuted_idx[examples_to_unlearn:examples_to_unlearn+examples_as_canaries].tolist()
            examples_to_unlearn += examples_as_canaries  # start the retain set after counting the canaries
        retain_idx = permuted_idx[examples_to_unlearn:]
    else:
        target_class = int(args.unlearning_target_class)
        if args.dataset_name in ["cifar10", "cifar100"]:
            dataset_targets = train_set.targets  # works for CIFAR-10/CIFAR-100
        else:
            raise NotImplementedError(f"targets should be defined for the dataset {args.dataset_name}")
        dataset_targets = np.array(dataset_targets)  # assume it to be a python array for now
        assert len(dataset_targets.shape) == 1, f"{dataset_targets.shape} was assumed to be a vector"
        target_class_idx = np.where(dataset_targets == target_class)[0]
        if args.unlearning_example_selection_criterion == "random":
            permuted_idx = np.random.default_rng(args.seed).permutation(target_class_idx)
        elif args.unlearning_example_selection_criterion in ["low_mem", "high_mem"]:
            permuted_idx = np.argsort(memorization_values)  # sort in ascending order
            if args.unlearning_example_selection_criterion == "high_mem":
                permuted_idx = permuted_idx[::-1]  # invert the indices
            permuted_idx = np.array([i for i in permuted_idx if i in target_class_idx])  # sorted target class indices w.r.t. memorization scores
        else:
            raise RuntimeError(f"Unknown unlearning example selection criterion: {args.unlearning_example_selection_criterion}")
        examples_to_unlearn = int(args.fraction_to_unlearn * len(target_class_idx))
        unlearning_idx = permuted_idx[:examples_to_unlearn].tolist()
        if args.fraction_of_canaries is not None:
            # TODO: make it independent of the unlearning selection i.e., by randomly picking instances from the retain set?
            examples_as_canaries = int(args.fraction_of_canaries * len(target_class_idx))
            canaries_idx = permuted_idx[examples_to_unlearn:examples_to_unlearn+examples_as_canaries].tolist()
            examples_to_unlearn += examples_as_canaries  # start the retain set after counting the canaries
        retain_idx = permuted_idx[examples_to_unlearn:].tolist()

        # Also add instances from all other classes to the retain set
        other_class_instances = np.where(dataset_targets != target_class)[0]
        retain_idx = retain_idx + other_class_instances.tolist()
    print(f"Unlearning selections / fraction: {args.fraction_to_unlearn} / # unlearn: {len(unlearning_idx)}"
          f" / canaries: {len(canaries_idx)} / # retain: {len(retain_idx)} / total: {len(train_set)}")
    print("Sample unlearning idxs:", unlearning_idx[:25])
    assert len(retain_idx) + len(unlearning_idx) + len(canaries_idx) == len(train_set), \
        f"{len(retain_idx)} + {len(unlearning_idx)} + {len(canaries_idx)} != {len(train_set)}"
    assert len(set(canaries_idx).intersection(set(unlearning_idx))) == 0, \
        f"Overlap found between canaries and unlearning idx: {len(set(canaries_idx).intersection(set(unlearning_idx)))}"
    assert len(set(retain_idx).intersection(set(unlearning_idx).union(set(canaries_idx)))) == 0, \
        f"Overlap found between retain and unlearning+canaries idx: {len(set(retain_idx).intersection(set(unlearning_idx).union(set(canaries_idx))))}"

    retain_set = torch.utils.data.Subset(train_set, retain_idx)
    forget_set = torch.utils.data.Subset(train_set, unlearning_idx)
    canaries_set = torch.utils.data.Subset(train_set, canaries_idx)  # can be empty
    if len(canaries_set) > 0:
        wrapper = MislabeledWrapper(canaries_set, num_classes, args.seed, target_class="any")  # add incorrect labels for the canaries set
        if args.dataset_name not in ["cifar10", "cifar100"]:
            raise NotImplementedError(f"Label replacement for dataset {args.dataset_name} not implemented")
        for random_idx, replace_idx in enumerate(canaries_idx):
            train_set.targets[replace_idx] = wrapper.random_labels[random_idx]
        # train_set.targets[canaries_idx] = wrapper.random_labels  # replace the targets in the original dataset
        # canaries_set = MislabeledWrapper(canaries_set, num_classes, args.seed, target_class="any")  # add incorrect labels for the canaries set
    forget_canaries_set = torch.utils.data.ConcatDataset([forget_set, canaries_set])
    print(f"# examples in the forget set: {len(forget_set)} / retain set: {len(retain_set)} / canaries set: {len(canaries_set)}")

    train_loader = get_dataloader(train_set, args.batch_size, args.num_workers, generator=generator)
    train_wo_transform_loader = None
    if train_set_wo_transform is not None:
        train_wo_transform_loader = get_dataloader(train_set_wo_transform, args.batch_size, args.num_workers, generator=generator)
    test_loader = get_dataloader(test_set, args.test_batch_size, args.num_workers, generator=generator)
    forget_loader = get_dataloader(forget_set, args.unlearning_batch_size, args.num_workers, generator=generator)
    retain_loader = get_dataloader(retain_set, args.batch_size, args.num_workers, generator=generator)
    canaries_loader = None
    if len(canaries_set) > 0:
        canaries_loader = get_dataloader(canaries_set, args.batch_size, args.num_workers, generator=generator)
    forget_canaries_loader = get_dataloader(forget_canaries_set, args.unlearning_batch_size, args.num_workers, generator=generator)

    # Define the relearn set which is a fraction of the forget set
    examples_to_relearn = int(args.fraction_to_relearn * len(forget_set))
    if args.relearning_example_selection_criterion == "random":
        forget_set_permuted_idx = np.random.default_rng(args.seed).permutation(np.arange(len(forget_set)))
    elif args.relearning_example_selection_criterion in ["low_mem", "high_mem"]:
        selected_memorization_values = [float(memorization_values[i]) for i in unlearning_idx]  # grab the memorization values of the examples in the forget set
        forget_set_permuted_idx = np.argsort(selected_memorization_values)  # sort in ascending order
        if args.relearning_example_selection_criterion == "high_mem":
            forget_set_permuted_idx = forget_set_permuted_idx[::-1]  # invert the indices
    else:
        raise RuntimeError(f"Unknown relearning example selection criterion: {args.relearning_example_selection_criterion}")
    relearning_idx = forget_set_permuted_idx[:examples_to_relearn]
    remaining_unlearned_idx = forget_set_permuted_idx[examples_to_relearn:]
    relearn_forget_set = torch.utils.data.Subset(forget_set, relearning_idx)
    remaining_forget_set = torch.utils.data.Subset(forget_set, remaining_unlearned_idx)
    print(f"Relearning selections / fraction: {args.fraction_to_relearn} / # forget: {len(forget_set)}"
          f" / # relearn: {len(relearn_forget_set)} / # remaining unlearned: {len(remaining_forget_set)}")
    print("Sample relearning idxs:", relearning_idx[:25])

    # Plot examples from the retain set, forget set, and the relearn set
    num_examples_to_plot = 64
    retain_selected_ex = np.random.default_rng(args.seed).choice(np.arange(len(retain_set)), size=num_examples_to_plot, replace=False)
    forget_selected_ex = np.arange(len(forget_set))
    canaries_selected_ex = np.arange(len(canaries_set))
    relearn_selected_ex = np.arange(len(relearn_forget_set))
    if num_examples_to_plot < len(forget_selected_ex):
        forget_selected_ex = np.random.default_rng(args.seed).choice(forget_selected_ex, size=num_examples_to_plot, replace=False)
    if num_examples_to_plot < len(canaries_selected_ex):
        canaries_selected_ex = np.random.default_rng(args.seed).choice(canaries_selected_ex, size=num_examples_to_plot, replace=False)
    if num_examples_to_plot < len(relearn_selected_ex):
        relearn_selected_ex = np.random.default_rng(args.seed).choice(relearn_selected_ex, size=num_examples_to_plot, replace=False)
    retain_grid = torchvision.utils.make_grid([retain_set[i][0] for i in retain_selected_ex], nrow=8)
    forget_grid = torchvision.utils.make_grid([forget_set[i][0] for i in forget_selected_ex], nrow=8)
    canaries_grid = None
    if len(canaries_selected_ex) > 0:
        canaries_grid = torchvision.utils.make_grid([canaries_set[i][0] for i in canaries_selected_ex], nrow=8)
    relearn_grid = torchvision.utils.make_grid([relearn_forget_set[i][0] for i in relearn_selected_ex], nrow=8)
    if wandb.run is not None and not args.evaluate_linear_mode_connectivity and not args.generate_relearning_grid \
        and not args.evaluate_parameter_diff:
        output_dict = {"retain_imgs": wandb.Image(transforms.ToPILImage()(retain_grid)),
                       "forget_imgs": wandb.Image(transforms.ToPILImage()(forget_grid)),
                       "relearn_forget_imgs": wandb.Image(transforms.ToPILImage()(relearn_grid))}
        if len(canaries_selected_ex) > 0:
            assert canaries_grid is not None
            output_dict["canaries_img"] = wandb.Image(transforms.ToPILImage()(canaries_grid))
        wandb.log(output_dict)

    relearn_forget_loader = get_dataloader(relearn_forget_set, args.batch_size, args.num_workers, generator=generator)
    remaining_forget_loader = get_dataloader(remaining_forget_set, args.batch_size, args.num_workers, generator=generator)
    additional_relearn_loaders = {"relearn_forget_set": relearn_forget_loader, "remaining_forget_set": remaining_forget_loader,
                                  "forget_set": forget_loader, "retain_set": retain_loader}
    if canaries_loader is not None:
        additional_relearn_loaders["canaries_set"] = canaries_loader

    relearn_complete_set = torch.utils.data.ConcatDataset([retain_set, relearn_forget_set])
    relearn_complete_test_set = torch.utils.data.ConcatDataset([test_set, relearn_forget_set])
    relearn_complete_corrupted_test_set = torch.utils.data.ConcatDataset([corrupted_test_set, relearn_forget_set])
    relearn_complete_loader = get_dataloader(relearn_complete_set, args.batch_size, args.num_workers, generator=generator)
    relearn_complete_test_loader = get_dataloader(relearn_complete_test_set, args.batch_size, args.num_workers, generator=generator)
    relearn_complete_corrupted_test_loader = get_dataloader(relearn_complete_corrupted_test_set, args.batch_size, args.num_workers,
                                                            generator=generator)

    mixed_dataset = None
    if "+retain_cls" in args.relearn_example_type:
        assert args.unlearning_target_class != "all", "Retain class instances for target unlearning class assumes sub-class unlearning"
        assert args.unlearning_target_class.isnumeric(), args.unlearning_target_class
        if args.relearn_example_type == "test+retain_cls":
            print("Using mixed dataset as a combination of the test set and the retain set...")
            mixed_dataset = ClassMixDataset(test_set, retain_set, mixing_class=int(args.unlearning_target_class))
        else:
            print("Using mixed dataset as a combination of the corrupted test set and the retain set...")
            assert args.relearn_example_type == "corrupted_test+retain_cls", args.relearn_example_type
            mixed_dataset = ClassMixDataset(corrupted_test_set, retain_set, mixing_class=int(args.unlearning_target_class))

    if args.relearn_example_type == "retain":
        print(f"Using both the retain set and a small subset of the forget set for relearning...")
        relearn_set, relearn_loader = relearn_complete_set, relearn_complete_loader
    elif args.relearn_example_type == "only_forget":
        print(f"Using only a small subset of the forget set for relearning...")
        relearn_set, relearn_loader = relearn_forget_set, relearn_forget_loader
    elif args.relearn_example_type == "test":
        print(f"Using both the test set and a small subset of the forget set for relearning...")
        relearn_set, relearn_loader = relearn_complete_test_set, relearn_complete_test_loader
    elif args.relearn_example_type == "corrupted_test":
        print(f"Using the corrupted test set and a small subset of the forget set for relearning...")
        relearn_set, relearn_loader = relearn_complete_corrupted_test_set, relearn_complete_corrupted_test_loader
    elif "+retain_cls" in args.relearn_example_type:
        print(f"Using the corrupted test set but with retain set unlearned class data and a small subset of the forget set for relearning...")
        assert mixed_dataset is not None
        relearn_complete_mixed_set = torch.utils.data.ConcatDataset([mixed_dataset, relearn_forget_set])
        mixed_ds_dataloader = get_dataloader(relearn_complete_mixed_set, args.batch_size, args.num_workers, generator=generator)
        relearn_set, relearn_loader = relearn_complete_mixed_set, mixed_ds_dataloader
    else:
        raise NotImplementedError(f"Relearn example type not implemented: {args.relearn_example_type}")
    print(f"{args.relearn_example_type} / # examples / relearn set: {len(relearn_set)} / relearn loader: {len(relearn_loader)}")

    if args.train_steps is None:
        args.train_steps = int((len(train_loader) * args.train_epochs) / args.gradient_accumulation_steps)
        args.train_steps = min(gather_tensor(args.train_steps))
        print(f"Train epochs {args.train_epochs} converted to train steps: {args.train_steps}")
    print("Total number of training steps:", args.train_steps)

    if args.unlearning_steps is None:
        args.unlearning_steps = int((len(train_loader) * args.unlearning_epochs) / args.gradient_accumulation_steps)
        args.unlearning_steps = min(gather_tensor(args.unlearning_steps))
        print(f"Unlearning train epochs {args.unlearning_epochs} converted to train steps: {args.unlearning_steps}")
    print("Total number of unlearning optimization steps:", args.unlearning_steps)

    if args.relearning_steps is None:
        args.relearning_steps = int((len(train_loader) * args.relearning_epochs) / args.gradient_accumulation_steps)
        args.relearning_steps = min(gather_tensor(args.relearning_steps))
        print(f"Relearning train epochs {args.relearning_epochs} converted to train steps: {args.relearning_steps}")
    print("Total number of relearning training steps:", args.relearning_steps)

    if args.amp_dtype is not None:  # convert the amp_dtype to torch dtype
        if args.amp_dtype == "fp16":
            args.amp_dtype = torch.float16
            if not args.use_grad_scaler:
                print("[WARNING] float16 AMP is being used without GradScaler")
        else:
            assert args.amp_dtype == "bfp16", args.amp_dtype
            args.amp_dtype = torch.bfloat16
        print("Using AMP dtype:", args.amp_dtype)

    if args.evaluate_linear_mode_connectivity or args.evaluate_parameter_diff:
        if args.evaluate_linear_mode_connectivity:
            print("Evaluating linear mode connectivity between different models...")
        else:
            assert args.evaluate_parameter_diff
            print("Evaluating parameter diff between different models...")

        if not os.path.exists(args.base_checkpoint_file):
            raise RuntimeError(f"Linear mode connectivity eval assumes that pretrained checkpoint already exists...")
        if not os.path.exists(args.unlearning_checkpoint_file):
            raise RuntimeError(f"Linear mode connectivity eval assumes that unlearned checkpoint already exists...")
        if not os.path.exists(args.retrain_from_scratch_checkpoint_file):
            raise RuntimeError(f"Linear mode connectivity eval assumes that retrain from scratch checkpoint already exists...")

        # Load all the model weights
        pretrained_model_state = torch.load(args.base_checkpoint_file, map_location="cpu")
        unlearned_model_state = torch.load(args.unlearning_checkpoint_file, map_location="cpu")
        retrain_from_scratch_model_state = torch.load(args.retrain_from_scratch_checkpoint_file, map_location="cpu")

        if args.evaluate_parameter_diff:
            # Convert the state dict into a parameter vector
            filter_trainable_params = True
            if filter_trainable_params:
                filter_lambda = lambda state_dict: filter_trainable(state_dict)
            else:
                filter_lambda = lambda state_dict: state_dict
            pretrained_vec = flatten_param_dict(param_dict=filter_lambda(pretrained_model_state))
            unlearned_vec = flatten_param_dict(param_dict=filter_lambda(unlearned_model_state))
            retrain_from_scratch_vec = flatten_param_dict(param_dict=filter_lambda(retrain_from_scratch_model_state))

            # Compute the difference norms between vectors
            pretrained_unlearned_diff = pretrained_vec - unlearned_vec
            pretrained_unlearned_diff = float(torch.norm(pretrained_unlearned_diff, p="fro"))

            pretrained_retrained_from_scratch_diff = pretrained_vec - retrain_from_scratch_vec
            pretrained_retrained_from_scratch_diff = float(torch.norm(pretrained_retrained_from_scratch_diff, p="fro"))

            # Log the results
            output_dict = {"pretrained_unlearned_diff": pretrained_unlearned_diff,
                           "pretrained_retrained_from_scratch_diff": pretrained_retrained_from_scratch_diff}
            print(json.dumps(output_dict))
            if wandb.run is not None:
                wandb.log(output_dict)

            terminate_process()

        config_list = [(pretrained_model_state, unlearned_model_state, "pretrained_vs_unlearned"),
                       (pretrained_model_state, retrain_from_scratch_model_state, "pretrained_vs_retrain_from_scratch")]
        for model_a, model_b, config_name in config_list:
            print("="*25)
            print(f">> Selected linear mode connectivity config: {config_name}")
            num_evals = 1. / args.mode_connectivity_stride
            assert num_evals.is_integer(), f"Expected the stride to vary from 0 to 1, but found: {args.mode_connectivity_stride}"
            decimal_places = len(str(args.mode_connectivity_stride).split("."))
            print(f"Stride: {args.mode_connectivity_stride} / # evaluations: {int(num_evals)} / decimal places: {decimal_places}")

            mixing_coeff = 0.
            for _ in range(int(num_evals)+1):
                print(f"!! Selected mixing coefficient: {mixing_coeff}")

                # Mix state dicts and load into the model
                mixed_state_dict = mix_state_dicts(model_a, model_b, mixing_coeff)
                model.load_state_dict(mixed_state_dict)

                # Evaluate the mixed model and log the results
                print("Evaluating mixed model...")
                log_name = f"{config_name}/coeff_{mixing_coeff}"
                evaluate_model(model, retain_loader, device, f"{log_name}/retain_set", num_classes)
                evaluate_model(model, forget_loader, device, f"{log_name}/forget_set", num_classes)
                evaluate_model(model, test_loader, device, f"{log_name}/test_set", num_classes)

                mixing_coeff = round(mixing_coeff + args.mode_connectivity_stride, decimal_places)  # update the coeff for the next iteration

        terminate_process()

    elif args.generate_relearning_grid:
        print("Generating relearning grid results...")
        if not os.path.exists(args.unlearning_checkpoint_file):
            raise RuntimeError(f"Relearning grid mode assumes that unlearned checkpoint already exists...")
        if not os.path.exists(args.retrain_from_scratch_checkpoint_file):
            raise RuntimeError(f"Relearning grid mode assumes that retrain from scratch checkpoint already exists...")
        args.relearning_checkpoint_file = None  # override to avoid writing any checkpoint
        max_forget_examples = len(forget_set_permuted_idx)
        print("# max forget examples:", max_forget_examples)

        # Identify the right models to evaluate
        model_type_list = [("unlearned", args.unlearning_checkpoint_file),
                           ("retrained_from_scratch", args.retrain_from_scratch_checkpoint_file)]
        if args.relearn_grid_eval_model == "unlearned":
            model_type_list = model_type_list[0:1]
        elif args.relearn_grid_eval_model == "retrain_from_scratch":
            model_type_list = model_type_list[1:2]
        print(f"Grid eval model: {args.relearn_grid_eval_model} / model type list: {model_type_list}")

        for model_type, checkpoint_file in model_type_list:
            model.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))
            _, retain_acc = evaluate_model(model, retain_loader, device, f"{model_type}_model/retain_set", num_classes)
            if canaries_loader is not None:
                evaluate_model(model, canaries_loader, device, f"{model_type}_model/canaries_set", num_classes)
            _, test_acc = evaluate_model(model, test_loader, device, f"{model_type}_model/test_set", num_classes)

        eval_examples_start = None
        end_limit = max_forget_examples - 1
        if not args.use_all_examples_for_eval:
            training_examples_fraction = 0.2
            eval_examples_start = int(max_forget_examples * training_examples_fraction)
            max_forget_examples = eval_examples_start
            end_limit = eval_examples_start + 1  # to make sure we can count the end limit within the sequence
            print(f"Using fixed examples for eval / full forget set: {len(forget_set_permuted_idx)} / training fraction: {training_examples_fraction}"
                  f" / eval start: {eval_examples_start} / max forget examples: {max_forget_examples} / end limit: {end_limit}")

        if args.relearn_grid_eval_ex == "limited":
            example_range_list = ([] if args.relearn_example_type == "only_forget" else [0]) + [1, 5, 10, 50, 100]  # limited eval
            example_range_list = [x for x in example_range_list if x < end_limit]  # remove all elements that exceed the limit
        elif args.relearn_grid_eval_ex == "dense":
            example_range_list = list(range(1 if args.relearn_example_type == "only_forget" else 0, 10)) + list(range(10, min(100, end_limit), 10))
            example_range_list += list(range(100, min(1000, end_limit), 100)) + list(range(1000, end_limit, 1000))
        else:
            assert args.relearn_grid_eval_ex == "zero", args.relearn_grid_eval_ex
            assert args.relearn_example_type != "only_forget", "0 examples is not possible with only forget relearning"
            example_range_list = [0]

        if not args.use_all_examples_for_eval:
            assert eval_examples_start >= max(example_range_list), f"{eval_examples_start} should be >= {max(example_range_list)}"
        print(f"Grid eval ex: {args.relearn_grid_eval_ex} / example range: {example_range_list}")

        # Iterate over the number of relearning examples
        for num_relearn_examples in example_range_list:
            print("~"*10)
            print("Selected number of relearning examples:", num_relearn_examples)
            print("~"*10)

            relearning_idx = forget_set_permuted_idx[:num_relearn_examples]
            if args.use_all_examples_for_eval:
                print("Using all remaining forget set examples for eval...")
                remaining_unlearned_idx = forget_set_permuted_idx[num_relearn_examples:]
            else:
                print("Using the same fixed number of examples for eval...")
                assert eval_examples_start >= num_relearn_examples, f"{eval_examples_start} should be >= {num_relearn_examples}"
                remaining_unlearned_idx = forget_set_permuted_idx[eval_examples_start:]
            relearn_forget_set = torch.utils.data.Subset(forget_set, relearning_idx)
            remaining_forget_set = torch.utils.data.Subset(forget_set, remaining_unlearned_idx)
            print(f"Relearning selections / # examples to relearn: {num_relearn_examples} / # forget: {len(forget_set)}"
                  f" / # relearn: {len(relearn_forget_set)} / # remaining unlearned: {len(remaining_forget_set)}")

            relearn_forget_loader = None
            if len(relearn_forget_set) > 0:
                relearn_forget_loader = get_dataloader(relearn_forget_set, args.batch_size, args.num_workers, generator=generator)
            remaining_forget_loader = get_dataloader(remaining_forget_set, args.batch_size, args.num_workers, generator=generator)
            additional_relearn_loaders = {"relearn_forget_set": relearn_forget_loader, "remaining_forget_set": remaining_forget_loader}
            if relearn_forget_loader is None:  # empty relearning set
                del additional_relearn_loaders["relearn_forget_set"]
            if canaries_loader is not None:
                additional_relearn_loaders["canaries_set"] = canaries_loader
            relearn_complete_set = torch.utils.data.ConcatDataset([retain_set, relearn_forget_set])
            relearn_complete_test_set = torch.utils.data.ConcatDataset([test_set, relearn_forget_set])
            relearn_complete_corrupted_test_set = torch.utils.data.ConcatDataset([corrupted_test_set, relearn_forget_set])
            relearn_complete_loader = get_dataloader(relearn_complete_set, args.batch_size, args.num_workers, generator=generator)
            relearn_complete_test_loader = get_dataloader(relearn_complete_test_set, args.batch_size, args.num_workers, generator=generator)
            relearn_complete_corrupted_test_loader = get_dataloader(relearn_complete_corrupted_test_set, args.batch_size, args.num_workers,
                                                                    generator=generator)

            if args.relearn_example_type == "retain":
                print(f"Using both the retain set and a small subset of the forget set for relearning...")
                relearn_set, relearn_loader = relearn_complete_set, relearn_complete_loader
            elif args.relearn_example_type == "only_forget":
                assert len(relearn_forget_set) > 0, "Cannot do relearning only on the forget set with a size of 0"
                print(f"Using only a small subset of the forget set for relearning...")
                relearn_set, relearn_loader = relearn_forget_set, relearn_forget_loader
            elif args.relearn_example_type == "test":
                print(f"Using both the test set and a small subset of the forget set for relearning...")
                relearn_set, relearn_loader = relearn_complete_test_set, relearn_complete_test_loader
            elif args.relearn_example_type == "corrupted_test":
                print(f"Using the corrupted test set and a small subset of the forget set for relearning...")
                relearn_set, relearn_loader = relearn_complete_corrupted_test_set, relearn_complete_corrupted_test_loader
            elif "+retain_cls" in args.relearn_example_type:
                print(f"Using the corrupted test set but with retain set unlearned class data and a small subset of the forget set for relearning...")
                assert mixed_dataset is not None
                relearn_complete_mixed_set = torch.utils.data.ConcatDataset([mixed_dataset, relearn_forget_set])
                mixed_ds_dataloader = get_dataloader(relearn_complete_mixed_set, args.batch_size, args.num_workers, generator=generator)
                relearn_set, relearn_loader = relearn_complete_mixed_set, mixed_ds_dataloader
            else:
                raise NotImplementedError(f"Relearn example type not implemented: {args.relearn_example_type}")
            print(f"{args.relearn_example_type} / # examples / relearn set: {len(relearn_set)} / relearn loader: {len(relearn_loader)}")

            for model_type, checkpoint_file in model_type_list:
                print(f"Model type: {model_type} / checkpoint file: {checkpoint_file}")
                model.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))

                # Setup optimizer
                print(f"Optimizer name: {args.optimizer_name} / lr: {args.relearning_learning_rate} / wd: {args.relearning_weight_decay}")
                optimizer = get_optimizer(model, lr=args.relearning_learning_rate, wd=args.relearning_weight_decay, optimizer_name=args.optimizer_name)
                lr_scheduler = None
                if args.lr_scheduler is not None:
                    min_lr = args.relearning_learning_rate * 0.1
                    lr_scheduler = get_lr_scheduler(optimizer, args.relearning_steps, max_lr=None, min_lr=min_lr, warmup_steps=None,
                                                    scheduler_name="cosine")
                    print(f"!! Using LR scheduler: {args.lr_scheduler}")

                grad_scaler = None
                if args.amp_dtype is not None and args.use_grad_scaler:
                    print("Using gradient scaler...")
                    grad_scaler = torch.amp.GradScaler('cuda')

                phase = f"{model_type}_model/relearning_ex_{num_relearn_examples}"
                if len(relearn_forget_set) > 0:  # skip when using empty relearn set
                    evaluate_model(model, relearn_forget_loader, device, f"{phase}/relearn_forget_set", num_classes)
                evaluate_model(model, remaining_forget_loader, device, f"{phase}/remaining_forget_set", num_classes)
                if canaries_loader is not None:
                    evaluate_model(model, canaries_loader, device, f"{phase}/canaries_set", num_classes)

                # Train the model on the relearn set
                train_direct(model, relearn_loader, test_loader, optimizer, lr_scheduler, args.relearning_steps,
                             args.eval_after_steps, args.gradient_accumulation_steps, device, args.amp_dtype, grad_scaler,
                             args.clip_grad_norm, args.relearning_checkpoint_file, num_classes, training_phase=phase,
                             additional_loaders=additional_relearn_loaders)

        terminate_process()

    print("Evaluating random init model...")
    model.load_state_dict(torch.load(args.random_init_checkpoint_file, map_location="cpu"))
    evaluate_model(model, test_loader, device, "random_init/test_set", num_classes)

    if not os.path.exists(args.base_checkpoint_file):
        print(f"Checkpoint file not found: {args.base_checkpoint_file}")
        print(">> Initiating base model training...")

        # Reload the randomly initialized model
        model.load_state_dict(torch.load(args.random_init_checkpoint_file, map_location="cpu"))

        # Setup optimizer
        print(f"Optimizer name: {args.optimizer_name} / lr: {args.learning_rate} / wd: {args.weight_decay}")
        optimizer = get_optimizer(model, lr=args.learning_rate, wd=args.weight_decay, optimizer_name=args.optimizer_name)
        lr_scheduler = None
        if args.lr_scheduler is not None:
            min_lr = args.learning_rate * 0.1
            lr_scheduler = get_lr_scheduler(optimizer, args.train_steps, max_lr=None, min_lr=min_lr, warmup_steps=None,
                                            scheduler_name="cosine")
            print(f"!! Using LR scheduler: {args.lr_scheduler}")

        grad_scaler = None
        if args.amp_dtype is not None and args.use_grad_scaler:
            print("Using gradient scaler...")
            grad_scaler = torch.amp.GradScaler('cuda')

        # Train the model
        train_direct(model, train_loader, test_loader, optimizer, lr_scheduler, args.train_steps, args.eval_after_steps,
                     args.gradient_accumulation_steps, device, args.amp_dtype, grad_scaler, args.clip_grad_norm,
                     args.base_checkpoint_file, num_classes, training_phase="pretraining")

        wait_for_other_procs()
        print("!! Base model training finished...")
        del optimizer
        del grad_scaler

    print("Evaluating base model...")
    model.load_state_dict(torch.load(args.base_checkpoint_file, map_location="cpu"))
    _, retain_acc = evaluate_model(model, retain_loader, device, "pretrained_model/retain_set", num_classes)
    _, forget_acc = evaluate_model(model, forget_loader, device, "pretrained_model/forget_set", num_classes)
    if canaries_loader is not None:
        evaluate_model(model, canaries_loader, device, "pretrained_model/canaries_set", num_classes)
    _, test_acc = evaluate_model(model, test_loader, device, "pretrained_model/test_set", num_classes)
    tow = (retain_acc) * (1. - forget_acc) * test_acc
    if wandb.run is not None:
        wandb.log({"pretrained_model/tow": tow})

    if not os.path.exists(args.retrain_from_scratch_checkpoint_file):
        print(f"Checkpoint file not found: {args.retrain_from_scratch_checkpoint_file}")
        print(">> Initiating retrain from scratch model training...")

        # Reload the randomly initialized model
        model.load_state_dict(torch.load(args.random_init_checkpoint_file, map_location="cpu"))

        # Setup optimizer
        print(f"Optimizer name: {args.optimizer_name} / lr: {args.learning_rate} / wd: {args.weight_decay}")
        optimizer = get_optimizer(model, lr=args.learning_rate, wd=args.weight_decay, optimizer_name=args.optimizer_name)
        lr_scheduler = None
        if args.lr_scheduler is not None:
            min_lr = args.learning_rate * 0.1
            lr_scheduler = get_lr_scheduler(optimizer, args.train_steps, max_lr=None, min_lr=min_lr, warmup_steps=None,
                                            scheduler_name="cosine")
            print(f"!! Using LR scheduler: {args.lr_scheduler}")

        grad_scaler = None
        if args.amp_dtype is not None and args.use_grad_scaler:
            print("Using gradient scaler...")
            grad_scaler = torch.amp.GradScaler('cuda')

        # Train the model
        train_direct(model, retain_loader, test_loader, optimizer, lr_scheduler, args.train_steps, args.eval_after_steps,
                     args.gradient_accumulation_steps, device, args.amp_dtype, grad_scaler, args.clip_grad_norm,
                     args.retrain_from_scratch_checkpoint_file, num_classes, training_phase="retraining_from_scratch")

        wait_for_other_procs()
        print("!! Retrain from scratch model training finished...")
        del optimizer
        del grad_scaler

    print("Evaluating retrain from scratch model...")
    model.load_state_dict(torch.load(args.retrain_from_scratch_checkpoint_file, map_location="cpu"))
    _, retain_acc = evaluate_model(model, retain_loader, device, "retrain_from_scratch/retain_set", num_classes)
    _, forget_acc = evaluate_model(model, forget_loader, device, "retrain_from_scratch/forget_set", num_classes)
    if canaries_loader is not None:
        evaluate_model(model, canaries_loader, device, "retrain_from_scratch/canaries_set", num_classes)
    _, test_acc = evaluate_model(model, test_loader, device, "retrain_from_scratch/test_set", num_classes)
    tow = (retain_acc) * (1. - forget_acc) * test_acc
    if wandb.run is not None:
        wandb.log({"retrain_from_scratch/tow": tow})

    if not os.path.exists(args.unlearning_checkpoint_file):
        print(f"Unlearned checkpoint file not found: {args.unlearning_checkpoint_file}")
        print(">> Initiating model unlearning...")

        if args.initial_safeguarded_model is not None:
            # Load the trained safeguarded model for TAR and other methods which support init_sg
            model.load_state_dict(torch.load(args.initial_safeguarded_model, map_location="cpu"))
            print("Initial safeguarded model loaded from checkpoint:", args.initial_safeguarded_model)

            print("Evaluating initial safeguarded model...")
            _, retain_acc = evaluate_model(model, retain_loader, device, "unlearning_initial_safeguard/retain_set", num_classes)
            _, forget_acc = evaluate_model(model, forget_loader, device, "unlearning_initial_safeguard/forget_set", num_classes)
            if canaries_loader is not None:
                evaluate_model(model, canaries_loader, device, "unlearning_initial_safeguard/canaries_set", num_classes)
            _, test_acc = evaluate_model(model, test_loader, device, "unlearning_initial_safeguard/test_set", num_classes)
            tow = (retain_acc) * (1. - forget_acc) * test_acc
            if wandb.run is not None:
                wandb.log({"unlearning_initial_safeguard/tow": tow})

        else:
            # Load the trained base model
            model.load_state_dict(torch.load(args.base_checkpoint_file, map_location="cpu"))
            print("Base model loaded from checkpoint:", args.base_checkpoint_file)
        base_module = model.module if hasattr(model, 'module') else model
        orig_model = copy.deepcopy(base_module).to(sec_device)  # create a new copy of the base model checkpoint

        pretrained_model = None
        if args.unlearning_method in ["mode_connectivity", "weight_dist_reg"]:  # also keep track of the pretrained model
            print(f"Creating a copy of the pretrained model for {args.unlearning_method.replace('_', ' ')}...")
            pretrained_model = copy.deepcopy(base_module).to(sec_device)  # create a new copy of the base model checkpoint
            pretrained_model.load_state_dict(torch.load(args.base_checkpoint_file, map_location="cpu"))
            pretrained_model.eval()

        # Setup optimizer
        print(f"Optimizer name: {args.optimizer_name} / lr: {args.unlearning_learning_rate} / wd: {args.unlearning_weight_decay}")
        optimizer = None  # initialized inside the train method after jittering model weights

        grad_scaler = None
        if args.amp_dtype is not None and args.use_grad_scaler:
            print("Using gradient scaler...")
            grad_scaler = torch.amp.GradScaler('cuda')

        # Train the model (use the forget_canaries set instead of just the forget set)
        train(model, orig_model, forget_canaries_loader, retain_loader, test_loader, optimizer, args.unlearning_steps,
              args.eval_after_steps, args.unlearning_method, args.unlearning_alpha, args.unlearning_gamma,
              args.unlearning_layer_idx, args.gradient_accumulation_steps, device, sec_device, args.amp_dtype,
              grad_scaler, args.clip_grad_norm, args.unlearning_checkpoint_file, num_classes, pretrained_model=pretrained_model,
              args=args)

        wait_for_other_procs()
        print("!! Model training finished...")
        del optimizer
        del grad_scaler
        del orig_model

    print("Performing final evaluation for unlearned model...")
    model.load_state_dict(torch.load(args.unlearning_checkpoint_file, map_location="cpu"))
    _, retain_acc = evaluate_model(model, retain_loader, device, "unlearned_model/retain_set", num_classes)
    _, forget_acc = evaluate_model(model, forget_loader, device, "unlearned_model/forget_set", num_classes)
    if canaries_loader is not None:
        evaluate_model(model, canaries_loader, device, "unlearned_model/canaries_set", num_classes)
    _, test_acc = evaluate_model(model, test_loader, device, "unlearned_model/test_set", num_classes)
    tow = (retain_acc) * (1. - forget_acc) * test_acc
    if wandb.run is not None:
        wandb.log({"unlearned_model/tow": tow})

    if not os.path.exists(args.relearning_checkpoint_file):
        print(f"Relearned checkpoint file not found: {args.relearning_checkpoint_file}")
        print(">> Initiating model relearning...")

        # Setup optimizer
        print(f"Optimizer name: {args.optimizer_name} / lr: {args.relearning_learning_rate} / wd: {args.relearning_weight_decay}")
        optimizer = get_optimizer(model, lr=args.relearning_learning_rate, wd=args.relearning_weight_decay, optimizer_name=args.optimizer_name)
        lr_scheduler = None
        if args.lr_scheduler is not None:
            min_lr = args.relearning_learning_rate * 0.1
            lr_scheduler = get_lr_scheduler(optimizer, args.relearning_steps, max_lr=None, min_lr=min_lr, warmup_steps=None,
                                            scheduler_name="cosine")
            print(f"!! Using LR scheduler: {args.lr_scheduler}")

        grad_scaler = None
        if args.amp_dtype is not None and args.use_grad_scaler:
            print("Using gradient scaler...")
            grad_scaler = torch.amp.GradScaler('cuda')

        # Train the model on the relearn set
        train_direct(model, relearn_loader, test_loader, optimizer, lr_scheduler, args.relearning_steps,
                     args.eval_after_steps, args.gradient_accumulation_steps, device, args.amp_dtype, grad_scaler,
                     args.clip_grad_norm, args.relearning_checkpoint_file, num_classes, training_phase="relearning",
                     additional_loaders=additional_relearn_loaders)

    print("Performing final evaluation for relearned model...")
    model.load_state_dict(torch.load(args.relearning_checkpoint_file, map_location="cpu"))
    _, retain_acc = evaluate_model(model, retain_loader, device, "relearned_model/retain_set", num_classes)
    _, forget_acc = evaluate_model(model, forget_loader, device, "relearned_model/forget_set", num_classes)
    evaluate_model(model, additional_relearn_loaders["relearn_forget_set"], device,
                   "relearned_model/relearn_forget_set", num_classes)
    evaluate_model(model, additional_relearn_loaders["remaining_forget_set"], device,
                   "relearned_model/remaining_forget_set", num_classes)
    if canaries_loader is not None:
        evaluate_model(model, canaries_loader, device, "relearned_model/canaries_set", num_classes)
    _, test_acc = evaluate_model(model, test_loader, device, "relearned_model/test_set", num_classes)
    tow = (retain_acc) * (1. - forget_acc) * test_acc
    if wandb.run is not None:
        wandb.log({"relearned_model/tow": tow})

    if not os.path.exists(args.relearn_from_scratch_checkpoint_file):
        print(f"Checkpoint file not found: {args.relearn_from_scratch_checkpoint_file}")
        print(">> Initiating relearn from scratch model training...")

        # Reload the retrained from scratch model
        model.load_state_dict(torch.load(args.retrain_from_scratch_checkpoint_file, map_location="cpu"))

        # Setup optimizer
        print(f"Optimizer name: {args.optimizer_name} / lr: {args.learning_rate} / wd: {args.weight_decay}")
        optimizer = get_optimizer(model, lr=args.learning_rate, wd=args.weight_decay, optimizer_name=args.optimizer_name)
        lr_scheduler = None
        if args.lr_scheduler is not None:
            min_lr = args.learning_rate * 0.1
            lr_scheduler = get_lr_scheduler(optimizer, args.train_steps, max_lr=None, min_lr=min_lr, warmup_steps=None,
                                            scheduler_name="cosine")
            print(f"!! Using LR scheduler: {args.lr_scheduler}")

        grad_scaler = None
        if args.amp_dtype is not None and args.use_grad_scaler:
            print("Using gradient scaler...")
            grad_scaler = torch.amp.GradScaler('cuda')

        # Relearn from scratch
        train_direct(model, relearn_loader, test_loader, optimizer, lr_scheduler, args.relearning_steps,
                     args.eval_after_steps, args.gradient_accumulation_steps, device, args.amp_dtype, grad_scaler,
                     args.clip_grad_norm, args.relearn_from_scratch_checkpoint_file, num_classes,
                     training_phase="relearning_from_scratch", additional_loaders=additional_relearn_loaders)

        wait_for_other_procs()
        print("!! Relearn from scratch model training finished...")
        del optimizer
        del grad_scaler

    print("Evaluating relearn from scratch model...")
    model.load_state_dict(torch.load(args.relearn_from_scratch_checkpoint_file, map_location="cpu"))
    _, retain_acc = evaluate_model(model, retain_loader, device, "relearn_from_scratch/retain_set", num_classes)
    _, forget_acc = evaluate_model(model, forget_loader, device, "relearn_from_scratch/forget_set", num_classes)
    evaluate_model(model, additional_relearn_loaders["relearn_forget_set"], device,
                   "relearn_from_scratch/relearn_forget_set", num_classes)
    evaluate_model(model, additional_relearn_loaders["remaining_forget_set"], device,
                   "relearn_from_scratch/remaining_forget_set", num_classes)
    if canaries_loader is not None:
        evaluate_model(model, canaries_loader, device, "relearn_from_scratch/canaries_set", num_classes)
    _, test_acc = evaluate_model(model, test_loader, device, "relearn_from_scratch/test_set", num_classes)
    tow = (retain_acc) * (1. - forget_acc) * test_acc
    if wandb.run is not None:
        wandb.log({"relearn_from_scratch/tow": tow})

    if not os.path.exists(args.learn_from_scratch_checkpoint_file):
        print(f"Checkpoint file not found: {args.learn_from_scratch_checkpoint_file}")
        print(">> Initiating learn from scratch model training...")

        # Reload the randomly initialized model
        model.load_state_dict(torch.load(args.random_init_checkpoint_file, map_location="cpu"))

        # Setup optimizer
        print(f"Optimizer name: {args.optimizer_name} / lr: {args.learning_rate} / wd: {args.weight_decay}")
        optimizer = get_optimizer(model, lr=args.learning_rate, wd=args.weight_decay, optimizer_name=args.optimizer_name)
        lr_scheduler = None
        if args.lr_scheduler is not None:
            min_lr = args.learning_rate * 0.1
            lr_scheduler = get_lr_scheduler(optimizer, args.train_steps, max_lr=None, min_lr=min_lr, warmup_steps=None,
                                            scheduler_name="cosine")
            print(f"!! Using LR scheduler: {args.lr_scheduler}")

        grad_scaler = None
        if args.amp_dtype is not None and args.use_grad_scaler:
            print("Using gradient scaler...")
            grad_scaler = torch.amp.GradScaler('cuda')

        # Learn from scratch
        train_direct(model, relearn_complete_loader, test_loader, optimizer, lr_scheduler, args.train_steps,
                     args.eval_after_steps, args.gradient_accumulation_steps, device, args.amp_dtype,
                     grad_scaler, args.clip_grad_norm, args.learn_from_scratch_checkpoint_file, num_classes,
                     training_phase="learning_from_scratch", additional_loaders=additional_relearn_loaders)

        wait_for_other_procs()
        print("!! Learn from scratch model training finished...")
        del optimizer
        del grad_scaler

    print("Evaluating learn from scratch model...")
    model.load_state_dict(torch.load(args.learn_from_scratch_checkpoint_file, map_location="cpu"))
    _, retain_acc = evaluate_model(model, retain_loader, device, "learn_from_scratch/retain_set", num_classes)
    _, forget_acc = evaluate_model(model, forget_loader, device, "learn_from_scratch/forget_set", num_classes)
    evaluate_model(model, additional_relearn_loaders["relearn_forget_set"], device,
                   "learn_from_scratch/relearn_forget_set", num_classes)
    evaluate_model(model, additional_relearn_loaders["remaining_forget_set"], device,
                   "learn_from_scratch/remaining_forget_set", num_classes)
    if canaries_loader is not None:
        evaluate_model(model, canaries_loader, device, "learn_from_scratch/canaries_set", num_classes)
    _, test_acc = evaluate_model(model, test_loader, device, "learn_from_scratch/test_set", num_classes)
    tow = (retain_acc) * (1. - forget_acc) * test_acc
    if wandb.run is not None:
        wandb.log({"learn_from_scratch/tow": tow})

    terminate_process()


if __name__ == "__main__":
    dataset_choices = ['cifar10', 'cifar100']
    model_choices = ['resnet18', 'resnet34']
    unlearning_method_choices = ['catastrophic_forgetting', 'random_relabeling', 'alternating_random_relabeling', 'gradient_ascent',
                                 'alternating_gradient_ascent', 'circuit_breakers', 'alternating_circuit_breakers', 'scrub',
                                 'alternating_scrub', 'uniform_scrub', 'weight_distortion', 'weight_attenuation', 'weight_dropout',
                                 'ssd', 'l1_sparse', 'tar', 'mode_connectivity', 'weight_dist_reg']
    unlearning_target_class_choices = ["all"] + [str(x) for x in range(10)]
    example_selection_criterions = ["random", "low_mem", "high_mem"]
    relearn_example_types = ["retain", "only_forget", "test", "test+retain_cls", "corrupted_test", "corrupted_test+retain_cls"]
    relearn_grid_eval_models = ["both", "unlearned", "retrain_from_scratch"]
    relearn_grid_eval_examples = ["limited", "zero", "dense"]

    # Create ArgumentParser object
    parser = ArgumentParser(description='Argument parser for vision unlearner')

    # Add arguments
    parser.add_argument('-d', '--dataset-name', default=dataset_choices[0], choices=dataset_choices,
                        help=f'Dataset name (default: {dataset_choices[0]})')
    parser.add_argument('-m', '--model-name', default=model_choices[0], choices=model_choices,
                        help=f'Model name (default: {model_choices[0]})')
    parser.add_argument('--compile-model', action='store_true', default=False,
                        help='Compile model (only applicable for PyTorch > 2.0)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size per process (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        help='Batch size per process for testing (default: equal to --batch-size)')
    parser.add_argument('--train-steps', type=int, default=None,
                        help='Number of training steps (default: None)')
    parser.add_argument('--train-epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of gradient steps to accumulate before calling optimizer.step()')
    parser.add_argument('--amp-dtype', default='None', choices=['None', 'fp16', 'bfp16'],
                        help='AMP dtype for model training (defaults to None i.e., no AMP)')
    parser.add_argument('--use-grad-scaler', action='store_true', default=False,
                        help='Use gradient scaler for training (useful when using AMP)')
    parser.add_argument('--eval-after-steps', type=int, default=None,
                        help='Evaluate the model after the specified number of optimization steps (default: None)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='fine-tuning learning rate for optimization (default: 1e-4)')
    parser.add_argument('--lr-scheduler', default="cosine", choices=["none", "cosine", "linear"],
                        help='LR scheduler to be used for training (default: cosine)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay for optimization (default: 1e-4)')
    parser.add_argument('--clip-grad-norm', type=float, default=None,
                        help='gradient clipping magnitude (default: None)')
    parser.add_argument('--optimizer-name', default='adam', choices=['adamw', 'adam', 'sgd'],
                        help='optimizer name (default: adam)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for the dataloader (default: 8)')
    parser.add_argument('--seed', type=int, default=43,
                        help='seed value (default: 43)')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='W&B project name (none indicates no W&B initialization)')
    parser.add_argument('--fraction-of-canaries', type=float, default=None,
                        help='fraction of examples to add as canaries i.e., mislabeled examples (default: None)')
    parser.add_argument('-u', '--unlearning-method', default=unlearning_method_choices[0], choices=unlearning_method_choices,
                        help=f'Unlearning method (default: {unlearning_method_choices[0]})')
    parser.add_argument('--unlearning-layer-idx', type=str, default="4,7",
                        help='Unlearning layer idx (default: 4,7)')
    parser.add_argument('--unlearning-alpha', type=float, default=1,
                        help='Unlearning alpha (default: 1)')
    parser.add_argument('--unlearning-gamma', type=float, default=1,
                        help='Unlearning gamma for SCRUB (default: 1)')
    parser.add_argument('--fraction-to-unlearn', type=float, default=0.1,
                        help='fraction of class/dataset to unlearn (default: 0.1)')
    parser.add_argument('--unlearning-target-class', default='all', choices=unlearning_target_class_choices,
                        help='Classes to target for unlearning (default: all)')
    parser.add_argument('--unlearning-example-selection-criterion', default='random', choices=example_selection_criterions,
                        help='example selection criterion (default: random)')
    parser.add_argument('--unlearning-learning-rate', type=float, default=1e-5,
                        help='unlearning learning rate (default: 1e-5)')
    parser.add_argument('--unlearning-weight-decay', type=float, default=0.0,
                        help='unlearnign weight decay (default: 0.0)')
    parser.add_argument('--unlearning-steps', type=int, default=None,
                        help='Number of unlearning optimization steps (default: None)')
    parser.add_argument('--unlearning-epochs', type=int, default=100,
                        help='Number of unlearning epochs (default: 100)')
    parser.add_argument('--unlearning-batch-size', type=int, default=None,
                        help='Batch size for the unlearning dataset per process (default: 32)')
    parser.add_argument('--fraction-to-relearn', type=float, default=0.1,
                        help='fraction of the forget set to relearn (default: 0.1)')
    parser.add_argument('--relearning-example-selection-criterion', default='random', choices=example_selection_criterions,
                        help='example selection criterion for relearning (default: random)')
    parser.add_argument('--relearning-learning-rate', type=float, default=1e-5,
                        help='relearning learning rate (default: 1e-5)')
    parser.add_argument('--relearning-weight-decay', type=float, default=0.0,
                        help='relearnign weight decay (default: 0.0)')
    parser.add_argument('--relearning-steps', type=int, default=None,
                        help='Number of relearning optimization steps (default: None)')
    parser.add_argument('--relearning-epochs', type=int, default=100,
                        help='Number of relearning training epochs (default: 100)')
    parser.add_argument('--relearn-example-type', default=relearn_example_types[0], choices=relearn_example_types,
                        help=f'Relearn example type (default: {relearn_example_types[0]})')
    parser.add_argument('--generate-relearning-grid', action='store_true', default=False,
                        help='Generate relearning grid results (only applicable once the unlearned model exists)')
    parser.add_argument('--relearn-grid-eval-model', default=relearn_grid_eval_models[0], choices=relearn_grid_eval_models,
                        help=f'Relearning grid evaluation model (default: {relearn_grid_eval_models[0]})')
    parser.add_argument('--relearn-grid-eval-ex', default=relearn_grid_eval_examples[0], choices=relearn_grid_eval_examples,
                        help=f'Relearning grid evaluation examples (default: {relearn_grid_eval_examples[0]})')
    parser.add_argument('--use-all-examples-for-eval', action='store_true', default=False,
                        help='Use all examples for eval when doing relearning grid instead of a fixed eval set')
    parser.add_argument('--initial-safeguard-args', type=str, default=None,
                        help='Unlearning method arguments to load the initially safeguarded model when using TAR or other methods that support init_sg')
    parser.add_argument('--evaluate-linear-mode-connectivity', action='store_true', default=False,
                        help='Evaluate linear mode connectivity between the three main models')
    parser.add_argument('--evaluate-parameter-diff', action='store_true', default=False,
                        help='Evaluate parameter difference between the pretrained model and the unlearned model')
    parser.add_argument('--mode-connectivity-stride', type=float, default=0.05,
                        help='Stride used to define the mixing weights for the two networks')

    # Parse the arguments
    args = parser.parse_args()

    # Set default vals
    if args.amp_dtype == "None":
        args.amp_dtype = None
    if args.lr_scheduler == "none":
        args.lr_scheduler = None
    if args.eval_after_steps is not None and args.eval_after_steps <= 0:
        args.eval_after_steps = None
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
        print("Setting test batch size to be equal to batch size:", args.test_batch_size)
    if args.unlearning_batch_size is None or args.unlearning_batch_size < 0:
        args.unlearning_batch_size = args.batch_size
        print("Setting unlearning batch size to be equal to batch size:", args.unlearning_batch_size)
    if args.fraction_of_canaries <= 0.:
        args.fraction_of_canaries = None
    if args.initial_safeguard_args == "none":
        args.initial_safeguard_args = None

    # Basic option validation
    assert (args.train_epochs is None) != (args.train_steps is None), "Either --train-epochs or --train-steps should be specified"
    assert (args.unlearning_epochs is None) != (args.unlearning_steps is None), \
        "Either --unlearning-epochs or --unlearning-steps should be specified"
    assert (args.relearning_epochs is None) != (args.relearning_steps is None), \
        "Either --relearning-epochs or --relearning-steps should be specified"
    assert args.fraction_of_canaries is None or (isinstance(args.fraction_of_canaries, float) and
                                                 args.fraction_of_canaries >= 0.), args.fraction_of_canaries
    assert args.unlearning_method not in ["tar"] or args.initial_safeguard_args is not None, \
        "Initial safeguard args are essential for TAR"
    assert not (args.generate_relearning_grid and (args.evaluate_linear_mode_connectivity or args.evaluate_parameter_diff)), \
        "Cannot select both relearning grid generation and linear mode connectivity / parameter diff option at the same time"
    assert 0. < args.mode_connectivity_stride <= 0.5 and (1/args.mode_connectivity_stride).is_integer(), \
        f"{args.mode_connectivity_stride} is not a valid mixing coefficient"

    main(args)
