# Tamper-Resistant Unlearning Through Weight-Space Regularization

Official implementation to reproduce results from the paper; ***From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space Regularization***.

## Usage

The main experiments can be launched via a single script i.e., `scripts/launcher.sh`.
The script internally calls `scripts/train.sh`, which sets up the correct args for training, and then calls `vision_unlearner.py`.

Note that launcher script iterates over all methods and train/evaluates for the specified dataset and model combination.
The important variable in the script is ***`generate_relearning_grid`***.
Setting it to `true` evaluates the model, while setting it to `false` trains the model.
Evaluation assumes that model has already been trained.
`perform_lmc` variable (mutually exclusive with `generate_relearning_grid`) triggers linear-mode connectivity evaluation (which again assumes that the model is already trained).

Calling the `scripts/launcher.sh` script with `generate_relearning_grid` to be `false` should train and save all checkpoints for analysis.
Method-specific hyperparameters are defined in `scripts/train.sh`.

Evaluation of different relearn set type can be launched using `scripts/relearn_set_type_launcher.sh`, which iterates over all combinations.

You can you check the full list of supported arugments in the vision_unlearner.py script:
```bash
python vision_unlearner.py --help
```

### Plotting results from W&B

You can plot the results saved to your w&b account via the script located in `wandb_utils/get_wandb_relearn_grid.py`.
This assumes you already have the W&B API key saved as an environment variable.

### Dependencies

The code has minimal dependencies, mainly relying on PyTorch. See `requirements.txt` for reference.


## Citation

```
@article{siddiqui2022metadataarchaeology,
  title={From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space Regularization},
  author={Siddiqui, Shoaib Ahmed and Weller, Adrian and Krueger, David and Dziugaite, Gintare Karolina and Mozer, Michael Curtis and Triantafillou, Eleni},
  journal={arXiv preprint},
  year={2025},
  url={https://arxiv.org/abs/2505.22310}
}
```

## Issues/Feedback

In case of any issues, feel free to drop me an email or open an issue on the repository.

Email: **msas3@cam.ac.uk**

## License

MIT
