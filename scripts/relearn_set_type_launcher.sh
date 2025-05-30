#!/bin/bash

dataset="cifar10"
model_name="resnet18"
method_name="alternating_scrub"
unlearning_selection_criterion="high_mem"
relearning_selection_criterion="random"
generate_relearning_grid="false"
perform_lmc="false"
eval_param_diff="false"
init_sg=""

for relearning_example_type in "test" "retain" "only_forget" "test+retain_cls" "corrupted_test" "corrupted_test+retain_cls"; do
    ./scripts/train.sh ${dataset} ${model_name} ${method_name} ${unlearning_selection_criterion} ${relearning_selection_criterion} \
        ${relearning_example_type} ${generate_relearning_grid} ${perform_lmc} ${eval_param_diff} ${init_sg}
done
