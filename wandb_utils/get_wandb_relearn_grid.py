#!/bin/bash

import os
import sys
import time
import wandb
import pickle
from typing import List, Tuple

import natsort

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def extract_unlearning_method_name(name_parts: List[str]) -> str:
    unlearning_method = []
    last_part = None
    extracting_start = False
    prev_match = False
    for part in name_parts:
        if part == "unlearning":
            prev_match = True
        elif part == "method":
            assert prev_match, "unlearning match not found"
            assert last_part == "unlearning"
            extracting_start = True
            prev_match = False
        elif extracting_start:
            if part == "lr":
                extracting_start = False
                break
            else:
                unlearning_method.append(part)
        last_part = part
    unlearning_method = " ".join(unlearning_method)
    return unlearning_method


def convert_new_name_to_old_name(model_name: str) -> str:
    model_name = model_name.replace("_sched_", "_scheduler_")
    model_name = model_name.replace("_grad_sc_", "_grad_scaler_")
    model_name = model_name.replace("_alt_", "_alternating_")
    model_name = model_name.replace("_a_", "_alpha_")
    model_name = model_name.replace("_g_", "_gamma_")
    model_name = model_name.replace("_l_", "_layer_")
    model_name = model_name.replace("_frac_", "_fraction_")
    model_name = model_name.replace("_crit_", "_criterion_")
    model_name = model_name.replace("_tgt_", "_target_")
    return model_name


def extract_relearning_example_type(name_parts: List[str]) -> str:
    relearn_example_types = ["retain", "only_forget", "test", "test+retain_cls", "corrupted_test", "corrupted_test+retain_cls"]
    relearn_example_type = []
    extracting_start = False
    for part in name_parts:
        if extracting_start:
            if part == "fraction":
                extracting_start = False
                break
            else:
                relearn_example_type.append(part)
        elif part == "relearn":
            extracting_start = True
    if len(relearn_example_type) == 0:
        relearn_example_type = "retain"
    else:
        relearn_example_type = "_".join(relearn_example_type)
    assert relearn_example_type in relearn_example_types, f"Unrecognized relearning type: {relearn_example_type} from parts: {name_parts}"
    return relearn_example_type


def get_criterions(full_name: str) -> Tuple[str, str]:
    criterion_parts = full_name.split("_criterion_")
    assert len(criterion_parts) == 3, criterion_parts
    unlearning_criterion_parts = criterion_parts[1].split("_")
    unlearning_criterion = ""
    iterator = 0
    while unlearning_criterion_parts[iterator] != "ep":
        unlearning_criterion += "_" + unlearning_criterion_parts[iterator]
        iterator += 1
    unlearning_criterion = unlearning_criterion[1:]  # remove the first '_'

    relearning_criterion_parts = criterion_parts[-1].split("_")
    relearning_criterion = ""
    iterator = 0
    while relearning_criterion_parts[iterator] != "lr":
        relearning_criterion += "_" + relearning_criterion_parts[iterator]
        iterator += 1
    relearning_criterion = relearning_criterion[1:]  # remove the first '_'

    return unlearning_criterion, relearning_criterion


def normalize_method_name(method: str) -> str:
    method = method.lower().strip()
    for prefix in ("alt ", "alternating "):
        if method.startswith(prefix):
            method = method[len(prefix):]
            break
    return method


# Set plot args
skip_init_sg_runs = sys.argv[1] == "true" if len(sys.argv) > 1 else True
skip_non_init_sg_runs = not skip_init_sg_runs
assert not (skip_init_sg_runs and skip_non_init_sg_runs)
plot_fig_1 = False
relearn_ex_type = False
dataset = sys.argv[2] if len(sys.argv) > 2 else "cifar10"
model = sys.argv[3] if len(sys.argv) > 3 else "resnet18"
target_cls = sys.argv[4] if len(sys.argv) > 4 else "0"

assert dataset in ["cifar10", "cifar100"], dataset
assert model in ["resnet18", "resnet34"], model
assert target_cls in ["0", "all"], target_cls
print(f"Skip init sg: {skip_init_sg_runs} / dataset: {dataset} / model: {model} / target cls: {target_cls}")

timestamp = "zero_02_05_25" if plot_fig_1 else "c100_08_05_25" if dataset == "cifar100" else "r34_08_05_25" if model == "resnet34" \
    else "all_cls_08_05_25" if target_cls == "all" else "relearn_ex_15_05_25" if relearn_ex_type else "latest_28_04_25"
project_name = "vision-unlearning-grid-relearning-zero" if plot_fig_1 else "vision-unlearning-c100-grid-relearning" if dataset == "cifar100" \
    else "vision-unlearning-r34-grid-relearning" if model == "resnet34" else "vision-unlearning-all-cls-grid-relearning" if target_cls == "all" \
        else "vision-unlearning-grid-relearning-ex-type-latest" if relearn_ex_type else "vision-unlearning-grid-relearning-latest"
print("Time stamp:", timestamp)
print("Selected project:", project_name)
time.sleep(3)  # wait for 3 seconds

plots_output_dir = f"plots_{timestamp}/"
pickle_output_file = f"relearning_grid_data_{timestamp}.pkl"
fetch_canaries_stats = "canaries" in project_name
fetch_test_set_stats = True
filter_all_class_results = "all-class" in project_name
compare_relearning_example_type = "example-type" in project_name  or "ex-type" in project_name
assert not relearn_ex_type or compare_relearning_example_type
print(f"Fetch canaries: {fetch_canaries_stats} / test set: {fetch_test_set_stats} / all class results: {filter_all_class_results} / "
      f"compare relearning example type: {compare_relearning_example_type}")

model_type_list = ["unlearned", "retrained_from_scratch"]
fetch_latest_runs = True
discard_removed_runs = True  # discard previous runs that are now removed
verbose = False

if not os.path.exists(pickle_output_file) or fetch_latest_runs:
    loaded_name_list = []
    loaded_output_dict = None
    if os.path.exists(pickle_output_file):
        print("Loading existing data from file:", pickle_output_file)
        with open(pickle_output_file, "rb") as f:
            loaded_output_dict = pickle.load(f)
        loaded_name_list = loaded_output_dict["name_list"]

    api = wandb.Api(timeout=300)
    # Fetch all runs in the specified project
    runs = api.runs(path=project_name)
    print("Total runs in the project:", len(runs))

    # Iterate through each run and retrieve the table
    summary_list, config_list, name_list, loaded_model_type_list = [], [], [], []
    relearning_acc_list, test_set_acc_list, canaries_acc_list = [], [], []
    matched_loaded_idx = []
    for run in runs:
        if not run.name.startswith("cifar"):
            print(f"!! Skipping model: {run.name}")
            continue  # skip these models
        run_name = run.name
        if run_name in loaded_name_list:
            print(f"Run already exists in file. Skipping run: {run_name}")
            match_idx = [i for i in range(len(loaded_name_list)) if run_name == loaded_name_list[i]]
            assert len(match_idx) == 1
            matched_loaded_idx.append(match_idx[0])
            continue
        name_list.append(run_name)
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        rows = []
        for row in run.scan_history():
            rows.append(row)
        history_df = pd.DataFrame(rows)
        print("Columns for run:", run_name, "\n", history_df.columns)

        eval_type = "limited"
        if run_name.endswith("_ex_zero"):
            eval_type = "zero"
            run_name = run_name.replace("_ex_zero", "")
        elif run_name.endswith("_ex_dense"):
            eval_type = "dense"
            run_name = run_name.replace("_ex_dense", "")

        assert run_name.endswith("_grid") or run_name.endswith("_grid_model_retrain_from_scratch") or \
            run_name.endswith("_grid_model_unlearned"), run_name
        current_model_type_list = model_type_list
        if run_name.endswith("_grid_model_retrain_from_scratch"):
            current_model_type_list = ["retrained_from_scratch"]
        elif run_name.endswith("_grid_model_unlearned"):
            current_model_type_list = ["unlearned"]
        print("Current run model type list:", current_model_type_list)
        loaded_model_type_list.append(current_model_type_list)

        if eval_type == "limited":
            example_range_list = [0, 1, 5, 10, 50, 100]
            if dataset == "cifar100":
                example_range_list = example_range_list[:-2]  # discard the last two configs
        elif eval_type == "zero":
            example_range_list = [0]
        else:
            assert eval_type == "dense", eval_type
            example_range_list = list(range(0, 10)) + list(range(10, 101, 10))
        print(f"eval type: {eval_type} / example range: {example_range_list}")

        data_dict = {}
        test_set_dict = {}
        canaries_dict = {}
        for model_type in current_model_type_list:
            data_dict[model_type] = {}
            test_set_dict[model_type] = {}
            canaries_dict[model_type] = {}
            for num_relearn_examples in example_range_list:
                if "only_forget" in name_list[-1] and num_relearn_examples == 0:
                    continue
                metric_key = f"{model_type}_model/relearning_ex_{num_relearn_examples}/remaining_forget_set.accuracy"
                print("Retrieving key:", metric_key)
                assert metric_key in history_df.columns, f"{metric_key} not in {history_df.columns}"
                time_series = history_df[[metric_key]].dropna()
                print("TS shape:", time_series.shape)
                data_dict[model_type][num_relearn_examples] = time_series

                if fetch_test_set_stats:
                    test_set_metric_key = f"{model_type}_model/relearning_ex_{num_relearn_examples}/test_set.accuracy"
                    assert test_set_metric_key in history_df.columns, f"{test_set_metric_key} not in {history_df.columns}"
                    test_set_time_series = history_df[[test_set_metric_key]].dropna()
                    print("Test set TS shape:", test_set_time_series.shape)
                    test_set_dict[model_type][num_relearn_examples] = test_set_time_series

                if fetch_canaries_stats:
                    canaries_metric_key = f"{model_type}_model/relearning_ex_{num_relearn_examples}/canaries_set.accuracy"
                    assert canaries_metric_key in history_df.columns, f"{canaries_metric_key} not in {history_df.columns}"
                    canaries_time_series = history_df[[canaries_metric_key]].dropna()
                    print("Canaries TS shape:", canaries_time_series.shape)
                    canaries_dict[model_type][num_relearn_examples] = canaries_time_series
        relearning_acc_list.append(data_dict)
        if fetch_test_set_stats:
            test_set_acc_list.append(test_set_dict)
        if fetch_canaries_stats:
            canaries_acc_list.append(canaries_dict)

    output_dict = dict(summary_list=summary_list, config_list=config_list, name_list=name_list, loaded_model_type_list=loaded_model_type_list,
                       relearning_acc_list=relearning_acc_list)
    if fetch_test_set_stats:
        output_dict["test_set_acc_list"] = test_set_acc_list
    if fetch_canaries_stats:
        output_dict["canaries_acc_list"] = canaries_acc_list
    if loaded_output_dict is not None:
        discarded_run_idx = [i for i in range(len(loaded_name_list)) if i not in matched_loaded_idx]
        print("Number of unmatched runs in the existing log:", len(discarded_run_idx))
        for k in output_dict:
            assert k in loaded_output_dict, f"Loaded output dict should contain key: {k}"
            existing_list = loaded_output_dict[k]
            if discard_removed_runs:
                existing_list = [existing_list[i] for i in matched_loaded_idx]
            output_dict[k] = existing_list +  output_dict[k]  # append the new records to list
    with open(pickle_output_file, "wb") as f:
        pickle.dump(output_dict, f)

# Reload the file
print("Loading data from file:", pickle_output_file)
with open(pickle_output_file, "rb") as f:
    output_dict = pickle.load(f)
print("Loaded keys:", output_dict.keys())
name_list = output_dict["name_list"]
relearning_acc_list = output_dict["relearning_acc_list"]  # run idx -> model type -> num relearn examples -> time series
summary_list = output_dict["summary_list"]
loaded_model_type_list = output_dict["loaded_model_type_list"]
canaries_acc_list = None
if "canaries_acc_list" in output_dict:
    canaries_acc_list = output_dict["canaries_acc_list"]
test_set_acc_list = None
if "test_set_acc_list" in output_dict:
    test_set_acc_list = output_dict["test_set_acc_list"]
keys = list(relearning_acc_list[0][loaded_model_type_list[0][0]].keys())
name_list = [convert_new_name_to_old_name(x) for x in name_list]  # convert name to old style for compatibility purposes
print("Name list:", name_list)
print(f"# relearning acc list:", len(relearning_acc_list))
if test_set_acc_list:
    print(f"# test acc list:", len(test_set_acc_list))
if canaries_acc_list:
    print(f"# canaries acc list:", len(canaries_acc_list))
print("Keys:", keys)

sorted_name_idx_list = natsort.index_natsorted(name_list)  # get the sorted idx
seq_len = len(relearning_acc_list[0][loaded_model_type_list[0][0]][keys[0]])
x_axis = np.arange(seq_len)
print("x-axis shape:", x_axis.shape)
if verbose: print(relearning_acc_list)

output_file_format = "pdf"
if not os.path.exists(plots_output_dir):
    os.mkdir(plots_output_dir)
    print("Plots output directory created:", plots_output_dir)

sorted_keys = sorted(keys)
# sorted_keys = [1] + list(range(10, 51, 10))  # 1, 5, 10, 15, ...
print("Sorted keys:", sorted_keys)
norm = plt.Normalize(min(sorted_keys), max(sorted_keys))
cmap = plt.cm.cool

# Generate the plot
generate_grid_plots = False
if generate_grid_plots:
    for model_type in model_type_list:
        for idx in sorted_name_idx_list:
            name = name_list[idx]
            print("Name:", name)
            name_parts = name.split("_")
            base_name = "_".join(name_parts[:2])
            unlearning_method = extract_unlearning_method_name(name_parts).replace(" ", "_")
            fig, ax = plt.subplots(figsize=(6, 4))
            for i, k in enumerate(sorted_keys):
                values = relearning_acc_list[idx][model_type][k]
                ax.plot(x_axis, values, color=cmap(norm(k)), alpha=0.7)

            ax.set_xlabel("Training step")
            ax.set_ylabel("Accuracy on the held-out forget set")

            # Create a ScalarMappable and add a color bar to show the range of # examples
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # necessary for colorbar to work properly
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Number of relearning examples")

            fig.tight_layout()
            plt.savefig(os.path.join(plots_output_dir, f"grid_{model_type}_{base_name}_unlearning_method_{unlearning_method}.{output_file_format}"), dpi=300, bbox_inches="tight")
            plt.close()

# Generate the plot per model
num_rows, num_cols = 1, 1 if plot_fig_1 else 3 if skip_non_init_sg_runs else 4
total_plots = num_rows * num_cols
fontsize = 22
# color_list = [k for k, v in mcolors.TABLEAU_COLORS.items()]
line_type_list = ["solid", "dotted"]
if num_rows == 1:
    sorted_keys = ["unlearned", 0, 10, 100]
    if dataset == "cifar100" and skip_init_sg_runs:
        sorted_keys = sorted_keys[:-1]  # discard 100 examples eval
        num_cols -= 1  # remove the last col
elif len(sorted_keys) > total_plots:
    assert num_cols == 3, num_cols
    sorted_keys = ["unlearned", 0, 5, 10, 50, 100]
    if dataset == "cifar100":
        sorted_keys = sorted_keys[:-2]  # discard the last two configs
        num_cols -= 1  # remove the last col which would remove two plots
    # sorted_keys = sorted_keys[:total_plots-1] + [sorted_keys[-1]]  # use the max number of relearning examples at the last place
total_plots = num_rows * num_cols  # recompute total plots after adjustment
selected_method_list = None
if plot_fig_1:
    sorted_keys = ["unlearned", 0]
    # selected_method_list = ["scrub", "gradient ascent", "catastrophic forgetting", "circuit breakers", "weight distortion"]
    selected_method_list = ["scrub", "gradient ascent", "circuit breakers"]
elif skip_non_init_sg_runs:
    sorted_keys = ["sc", "cb", "wd"]  # should be 3 plots
print("Sorted keys:", sorted_keys)

# Define the full list of methods that we want to process (implicitly considers `retain from scratch``)
all_method_list = ["retrain from scratch", "random relabeling", "scrub", "gradient ascent", "circuit breakers", "l1 sparse", "catastrophic forgetting",
                   "ssd", "weight attenuation", "weight dropout", "tar", "weight distortion", "mode connectivity", "weight dist reg"]

# Sort the methods w.r.t. the given list of all methods above
indices_by_method = {m: [] for m in all_method_list}
for idx in sorted_name_idx_list:
    assert len(loaded_model_type_list[idx]) == 1, loaded_model_type_list[idx]
    model_type = loaded_model_type_list[idx][0]
    if model_type == "unlearned":
        parts = name_list[idx].split("_")
        raw_method = extract_unlearning_method_name(parts)
        norm_method = normalize_method_name(raw_method)
    else:
        assert model_type == "retrained_from_scratch", model_type
        norm_method = "retrain from scratch"

    assert norm_method in indices_by_method, f"{norm_method} not in {indices_by_method.keys()}"
    indices_by_method[norm_method].append(idx)

# Convert the method specific list to a linearized order
sorted_name_idx_list = []
for method in all_method_list:
    sorted_name_idx_list.extend(indices_by_method[method])

model_list = [(f"{dataset}_{model}", "low_mem", "random"), (f"{dataset}_{model}", "random", "random"), (f"{dataset}_{model}", "high_mem", "random")]
for model_config, selected_unlearning_criterion, selected_relearning_criterion in model_list:
    plot_list = [("held_out_acc", relearning_acc_list)]
    if canaries_acc_list is not None:
        plot_list += [("canaries_acc", canaries_acc_list)]

    ignore_init_sg_marker = True
    print(f"Model config: {model_config} / unlearning criterion: {selected_unlearning_criterion} / relearning criterion: {selected_relearning_criterion}")
    required_pastel_colors = (1 + (0 if ignore_init_sg_marker else 3)) if not skip_init_sg_runs else len(all_method_list) - 5  # exclude retrain from scratch + 3 other methods
    viridis_cmap = plt.get_cmap('tab10' if compare_relearning_example_type else 'viridis', required_pastel_colors)
    color_list = viridis_cmap(range(required_pastel_colors))
    color_list[:, 3] = 0.7 if compare_relearning_example_type else 0.4  # reduce color alpha for the pastel colors
    new_colors = ([] if skip_non_init_sg_runs else [[0., 128/255., 0., 0.8]]) + [[0., 191/255., 1., 0.8], [1., 0., 0., 0.8]]  # two separate colors for mode conn and weight dist reg
    color_list = np.concatenate([color_list, np.array(new_colors)], axis=0)

    num_points_to_avg = 50
    for container_type, container in plot_list:
        for plot_type in ["scatter", "line"]:
            # Set the plot args
            current_rows, current_cols = num_rows, num_cols
            if plot_fig_1:
                plot_width = 6
                plot_height = 4
                current_rows, current_cols = 1, 1  # override the number of rows and cols
            elif plot_type == "line":
                plot_width = 4
                plot_height = 3
                if skip_init_sg_runs:
                    current_cols -= 1  # no unlearned col
            else:
                assert plot_type == "scatter"
                plot_width = 3.5
                plot_height = plot_width
                if compare_relearning_example_type:
                    current_cols -= 1  # no unlearned col

            # Create the figure
            fig, ax = plt.subplots(current_rows, current_cols, figsize=(plot_width * current_cols, plot_height * current_rows), sharex=True, sharey=True)
            if plot_fig_1:
                if plot_type != "scatter":
                    continue  # not needed for fig 1 plots
                ax = np.array([ax])  # convert to np array for .ravel()

            handles_dict = {}
            unlearned_point = {}
            color_map = {}  # Maps each legend_label to a color
            color_counter = 0  # Tracks the next color to assign

            ax_list = ax.ravel()
            init_sg_list = ["alternating_scrub", "circuit_breakers", "weight_distortion"]
            init_sg_label_map = {"circuit_breakers": "CB", "alternating_scrub": "SC", "weight_distortion": "WD"}
            for i, k in enumerate(sorted_keys):
                if plot_type == "line":
                    if k == "unlearned":
                        continue  # skip this
                    i -= 1  # ignore the first sorted key
                elif compare_relearning_example_type:
                    if k == "unlearned":
                        continue  # ignore 'unlearned' config as doesn't require comparison
                    i -= 1  # ignore the first sorted key

                selected_init_sg = None
                if skip_non_init_sg_runs:
                    selected_init_sg = init_sg_list[i]
                    init_sg_label_map = {"circuit_breakers": "Circuit Breakers", "alternating_scrub": "SCRUB", "weight_distortion": "Weight Distortion"}
                    title = init_sg_label_map[selected_init_sg]
                else:
                    title = "unlearned model" if k == "unlearned" else f"{k} relearning examples"
                if plot_fig_1:
                    current_ax = ax_list[0]  # only one element in the list
                else:
                    current_ax = ax_list[i]
                    current_ax.set_title(title, fontsize=fontsize-5)
                rss_plotted = False

                processed_models = 0
                current_unlearning_method_list = []
                for idx in sorted_name_idx_list:
                    full_name = name_list[idx]

                    if filter_all_class_results:  # filter based on classes
                        search_str = "_fraction_0.01_target_all"
                        if search_str not in full_name:
                            print("Filtering model:", full_name)
                            continue

                    # Filter for k = 0 when using only_forget relearning
                    if "only_forget" in full_name and k == 0:
                        continue  # nothing to plot for 0 examples when considering only forget set for relearning

                    # Filter based on the config base name
                    name_parts = full_name.split("_")
                    base_name = "_".join(name_parts[:2])
                    if base_name != model_config:  # only plot one config at a time
                        continue

                    # Filter based on the unlearning + relearning criterions of the config
                    unlearning_criterion, relearning_criterion = get_criterions(full_name)
                    if verbose: print(f"Unlearning criterion: {unlearning_criterion} / relearning criterion: {relearning_criterion}")
                    if unlearning_criterion != selected_unlearning_criterion or relearning_criterion != selected_relearning_criterion:
                        continue

                    # Filter the run based on the init sg config
                    if skip_non_init_sg_runs and ("model_retrain_from_scratch" not in full_name and "_init_sg_" not in full_name):
                        assert selected_init_sg is not None
                        if selected_init_sg not in full_name:  # keep the init sg run as well
                            continue
                    elif skip_init_sg_runs and "_init_sg_" in full_name:
                        continue

                    # Filter for fig 1 methods
                    unlearning_method = extract_unlearning_method_name(name_parts)
                    legend_label = unlearning_method.replace("alternating ", "").title()
                    if plot_fig_1 and ("model_retrain_from_scratch" not in full_name and legend_label.lower() not in selected_method_list):
                        print("!! Skipping model:", legend_label.lower())
                        continue

                    processed_models += 1
                    model_summary_stats = summary_list[idx]

                    # Compute the legend label
                    if legend_label == "Mode Connectivity":
                        legend_label = "CBFT"  # update the label

                    if legend_label.lower() in ["scrub", "ssd", "tar"]:
                        legend_label = legend_label.upper()
                    if legend_label == "Gradient Ascent":
                        legend_label = "NegGrad+"  # update the label
                    elif legend_label == "Circuit Breakers":
                        # Extract layer info
                        layer_idx = [i for i, x in enumerate(name_parts) if x == "layer"]
                        assert len(layer_idx) == 1, layer_idx
                        layer_idx = layer_idx[0] + 1  # next index
                        # legend_label = f"{legend_label} (L: {name_parts[layer_idx]})"
                    elif "_init_sg_" in full_name:
                        # Extract init sg
                        if legend_label == "TAR" and "_4_16_adamw_firord_adv" not in full_name:
                            print("Discarding older TAR run:", full_name)
                            continue
                        init_sg_idx = [i for i in range(len(name_parts)-1) if name_parts[i] == "init" and name_parts[i+1] == "sg"]
                        assert len(init_sg_idx) == 1, init_sg_idx
                        init_sg_idx = init_sg_idx[0] + 2  # starting index
                        end_init_sg_idx = init_sg_idx
                        while name_parts[end_init_sg_idx] != "lr":
                            end_init_sg_idx += 1
                        init_sg = '_'.join(name_parts[init_sg_idx:end_init_sg_idx])
                        assert init_sg in init_sg_label_map.keys(), f"{init_sg} not in {init_sg_label_map.keys()}"
                        if selected_init_sg is not None:  # no need to update the legend label
                            if init_sg != selected_init_sg:
                                continue
                        else:
                            legend_label = f"{legend_label} ({init_sg_label_map[init_sg]})"
                    elif legend_label == "Catastrophic Forgetting":
                        if "_wd_0.001_" not in full_name:
                            continue  # skip the new run

                    relearning_example_type = extract_relearning_example_type(name_parts)
                    alpha = 0.2 if plot_fig_1 and k == "unlearned" else 0.7
                    if compare_relearning_example_type:
                        label_map = {"retain": r"$\mathcal{D}_{R}$", "test": r"$\mathcal{D}_{te}$", "only_forget": r"$\mathcal{D}_{F}$",
                                     "corrupted_test": r"$\mathcal{D}_{cte}$", "test+retain_cls": r"$\mathcal{D}_{te}^{\neg C} \cup \mathcal{D}_{R}^{C}$",
                                     "corrupted_test+retain_cls": r"$\mathcal{D}_{cte}^{\neg C} \cup \mathcal{D}_{R}^{C}$"}
                        legend_label = f"[{label_map[relearning_example_type]}] {legend_label}"
                        alpha = 0.6

                    if not (len(loaded_model_type_list[idx]) == 1 and loaded_model_type_list[idx][0] == "retrained_from_scratch"):
                        assert legend_label not in current_unlearning_method_list, \
                            f"Repeated labels / current: {current_unlearning_method_list} / new: {legend_label} / full name: {full_name}"
                        current_unlearning_method_list.append(legend_label)

                    for model_idx, model_type in enumerate(loaded_model_type_list[idx]):
                        selected_k = 0 if k == "unlearned" else k   # k doesn't matter for unlearned model as we'll just take the loss at the first step
                        marker = 's' if plot_fig_1 and k == "unlearned" else 'o'
                        if skip_non_init_sg_runs:
                            if model_type == "unlearned" and "_init_sg_" not in full_name and selected_init_sg not in full_name:
                                continue  # skip any additional runs included due to retrain from scratch
                            selected_k = 0  # don't iterate over the number of examples
                            if "_init_sg_" not in full_name and selected_init_sg in full_name:
                                marker = 'P'  # plus
                        values = container[idx][model_type][selected_k]
                        if model_type == "unlearned":  # unlearning method name makes sense
                            current_label = legend_label
                        else:  # no need to consider unlearning method name
                            current_label = "Retrain from scratch"
                            color_map[current_label] = "black"
                            marker = 's' if plot_fig_1 and k == "unlearned" else '*'
                            if rss_plotted:
                                continue

                        assert len(values.columns) == 1, values.columns
                        values = values[values.columns[0]].to_list()

                        if ignore_init_sg_marker and marker == "P":
                            color = None
                        else:
                            if current_label not in color_map:
                                color_map[current_label] = color_list[color_counter]  # we assume out here that we have a unique color for each config
                                color_counter += 1
                            color = color_map[current_label]

                        if plot_type == "line":
                            (line,) = current_ax.plot(x_axis, values, color=color, linewidth=3, linestyle='solid', label=current_label)
                            x_lim = len(x_axis)
                        else:
                            assert plot_type == "scatter", plot_type
                            if k == "unlearned":
                                mean_val = values[0]  # first value indicates before unlearning
                                test_acc = model_summary_stats[f"{model_type}_model/test_set"]["accuracy"]
                            else:
                                assert isinstance(k, int) or skip_non_init_sg_runs, k
                                mean_val = np.mean(values[-num_points_to_avg:])
                                if test_set_acc_list is None:  # use the proxy clean acc from the unlearned model as no other info is available
                                    test_acc = model_summary_stats[f"{model_type}_model/test_set"]["accuracy"]
                                else:  # get the average test accuracy during the relearning phase -- similar to the relearning acc computation
                                    test_values = test_set_acc_list[idx][model_type][selected_k]
                                    test_acc = np.mean(test_values[-num_points_to_avg:])
                            x_lim = None

                            kwargs = {"alpha": alpha} if plot_fig_1 else {}
                            line = current_ax.scatter(test_acc, mean_val, color=color, marker=marker, linewidth=0, label=current_label,
                                                      edgecolor='k', sizes=[250], **kwargs)

                        current_ax.tick_params(axis='both', which='major', labelsize=fontsize)
                        current_ax.tick_params(axis='both', which='minor', labelsize=fontsize)
                        if x_lim is not None:
                            current_ax.set_xlim(-1, x_lim+1)
                        # if plot_type != "line":
                        #     current_ax.set_xlim(-0.05, 1.05)
                        current_ax.set_ylim(-0.05, 1.05)

                        if plot_type == "scatter":
                            current_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                            current_ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        else:
                            assert plot_type == "line", plot_type
                            current_ax.set_xscale('log')

                        # Only store the handle the first time we see this label
                        if line is not None and current_label not in handles_dict and not (plot_fig_1 and k == "unlearned"):
                            if not ignore_init_sg_marker or marker != "P":
                                handles_dict[current_label] = line
                        if plot_fig_1 and k == "unlearned" and current_label not in unlearned_point:
                            unlearned_point[current_label] = (test_acc, mean_val)
                        if plot_fig_1 and k != "unlearned":  # draw a line between the unlearned model and the relearned model
                            assert current_label in unlearned_point, unlearned_point
                            base_test_acc, base_mean_val = unlearned_point[current_label]
                            add_arrows = False
                            if add_arrows:
                                arrowprops = dict(arrowstyle="->", color=color, linewidth=5, alpha=0.5, shrinkA=0,
                                                  shrinkB=4)  # shrink to avoid overlap with the markers
                                current_ax.annotate("", xy=(test_acc, mean_val), xytext=(base_test_acc, base_mean_val),
                                                    arrowprops=arrowprops)
                            else:
                                current_ax.plot([base_test_acc, test_acc], [base_mean_val, mean_val], color=color, linewidth=5, alpha=0.4)

                        if model_type == "retrained_from_scratch":
                            rss_plotted = True  # only plot RSS once

                if not plot_fig_1 and i == total_plots - 1:
                    break

            if plot_type == "line":
                x_label = "Number of training steps (log-scale)"
            else:
                assert plot_type == "scatter", plot_type
                x_label = "Test set accuracy"
            if plot_fig_1:
                plt.xlabel(x_label, fontsize=fontsize)
            else:
                fig.supxlabel(x_label, fontsize=fontsize)
            if container_type == "held_out_acc":
                y_label = "Accuracy on the held-out forget set"
                if plot_type == "scatter_test_vs_unlearned":
                    y_label = f"{y_label} after unlearning"
                elif plot_type == "scatter_test_vs_relearned":
                    y_label = f"{y_label} after relearning"
            else:
                assert container_type == "canaries_acc", container_type
                y_label = "Accuracy on the canaries set"
                if plot_type == "scatter_test_vs_unlearned":
                    y_label = f"{y_label} after unlearning"
                elif plot_type == "scatter_test_vs_relearned":
                    y_label = f"{y_label} after relearning"
            y_label = "Forget set accuracy"  # override for the final plot
            if plot_fig_1:
                plt.ylabel(y_label, fontsize=fontsize)
            else:
                fig.supylabel(y_label, x=0.01, fontsize=fontsize)
            fig.tight_layout()

            if len(list(handles_dict.keys())) == 0:
                print("Empty config. Skipping model!")
                continue
            labels, handles = zip(*handles_dict.items())
            is_init_sg_plot = not skip_init_sg_runs and skip_non_init_sg_runs
            legend_cols = 2 if plot_fig_1 else 4 if is_init_sg_plot else 3
            legend_rows = int(np.ceil(len(labels) / legend_cols))
            if plot_fig_1:
                bbox_to_anchor = (0.5, 1.15 + 0.05 * legend_rows)
            elif compare_relearning_example_type:
                bbox_to_anchor = (0.5, (1.20 if plot_type == "scatter" else 1.25) + 0.05 * legend_rows)
            elif num_rows == 1:
                bbox_to_anchor = (0.5, (1.15 if is_init_sg_plot else 1.25 if plot_type == "scatter" else 1.3) + ((0.07 if plot_type == "scatter" else 0.08) * legend_rows))
            else:
                bbox_to_anchor = (0.5, 1.05 + (0.045 * legend_rows))
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=bbox_to_anchor, ncol=legend_cols, fontsize=fontsize-4)

            if plot_fig_1 and plot_type == "scatter":  # add an additional legend to highlight 'unlearned' ('s') and the 'relearned' ('o') models
                marker_handles = [
                    Line2D([], [], marker='s', color='k', alpha=0.2, linestyle='None', markeredgewidth=0, markersize=12, label='Unlearned'),
                    Line2D([], [], marker='o', color='k', alpha=0.6, linestyle='None', markeredgewidth=0, markersize=12, label='Relearned'),
                ]

                current_ax = ax_list[0]
                marker_legend = current_ax.legend(handles=marker_handles, loc='upper left', fontsize=fontsize - 4, frameon=True, fancybox=True,
                                                  borderpad=0.3, labelspacing=0.2, handletextpad=0.4,)
                # Make the legend frame visible and sharp.
                marker_legend.get_frame().set_edgecolor('black')
                marker_legend.get_frame().set_linewidth(0.8)

            fig.tight_layout()
            container_suffix = f"_{container_type}" if container_type != "held_out_acc" else ""
            base_name = f"{'fig_1_' if plot_fig_1 else ''}grid_{model_config}_unlearning_{selected_unlearning_criterion}"
            base_name += f"_relearning_{selected_relearning_criterion}_{plot_type}"
            base_name = f"{base_name}" + ("_init_sg" if skip_non_init_sg_runs else "_non_init_sg" if skip_init_sg_runs else "")
            plt.savefig(os.path.join(plots_output_dir, f"{base_name}{container_suffix}.{output_file_format}"), dpi=300, bbox_inches="tight")
            plt.close()
