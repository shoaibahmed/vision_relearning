from typing import List
import matplotlib.pyplot as plt

import torch


def plot_histogram(data: torch.tensor, output_file: str, xlabel: str, ylabel: str, bins: int = 100,
                   xscale: str = 'linear', yscale: str = 'linear', figsize: List[int] = (6, 4)):
    assert xscale in ["linear", "log"], xscale
    assert yscale in ["linear", "log"], yscale

    plt.figure(figsize=figsize)
    plt.hist(data.detach().cpu().view(-1).numpy(), bins=100)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
