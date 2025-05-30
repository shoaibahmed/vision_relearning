#!/bin/bash

dataset="cifar10"
model_name="resnet18"
generate_relearning_grid="false"
perform_lmc="false"
eval_param_diff="false"

unlearning_selection_criterion="high_mem"
relearning_selection_criterion="random"
relearning_example_type="retain"

# Iterate over methods without initial safeguard
init_sg=""
for method_name in 'retrain_from_scratch' 'catastrophic_forgetting' 'random_relabeling' 'alternating_gradient_ascent' 'circuit_breakers' 'alternating_scrub' 'weight_distortion' 'weight_attenuation' 'ssd' 'mode_connectivity' 'weight_dist_reg' 'weight_dropout' 'l1_sparse'; do
    echo "Evaluating method: ${method_name}"
    ./scripts/train.sh ${dataset} ${model_name} ${method_name} ${unlearning_selection_criterion} ${relearning_selection_criterion} \
        ${relearning_example_type} ${generate_relearning_grid} ${perform_lmc} ${eval_param_diff} ${init_sg}
done

# Iterate over methods that support initial safeguard
for method_name in 'tar' 'mode_connectivity' 'weight_dist_reg'; do
    for init_sg in "circuit_breakers_lr_1e-05_a_1.0_l_4,7_new_cu_sched" "alt_scrub_lr_1e-05_a_1.0_g_1.0" "weight_distortion_lr_1e-05_std_0.02"; do
        echo "Evaluating method: ${method_name} with init_sg: ${init_sg}"
        ./scripts/train.sh ${dataset} ${model_name} ${method_name} ${unlearning_selection_criterion} ${relearning_selection_criterion} \
            ${relearning_example_type} ${generate_relearning_grid} ${perform_lmc} ${eval_param_diff} ${init_sg}
    done
done
