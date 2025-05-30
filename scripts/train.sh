#!/bin/bash

DEFAULT_DATASET="cifar10"
DEFAULT_MODEL_NAME="resnet18"
DEFAULT_METHOD_NAME="alternating_scrub"
DEFAULT_UNLEARNING_EXAMPLE_SELECTION_CRITERION="high_mem"
DEFAULT_RELEARNING_EXAMPLE_SELECTION_CRITERION="random"
DEFAULT_RELEARNING_EXAMPLE_TYPE="retain"
DEFAULT_GENERATE_GRID="false"
DEFAULT_PERFORM_LMC="false"
DEFAULT_EVALUATE_PARAM_DIFF="false"
DEFAULT_INIT_SAFEGUARD="alt_scrub_lr_1e-05_a_1.0_g_1.0"

DATASET=${1:-$DEFAULT_DATASET}
MODEL_NAME=${2:-$DEFAULT_MODEL_NAME}
METHOD_NAME=${3:-$DEFAULT_METHOD_NAME}
UNLEARNING_EXAMPLE_SELECTION_CRITERION=${4:-$DEFAULT_UNLEARNING_EXAMPLE_SELECTION_CRITERION}
RELEARNING_EXAMPLE_SELECTION_CRITERION=${5:-$DEFAULT_RELEARNING_EXAMPLE_SELECTION_CRITERION}
RELEARNING_EXAMPLE_TYPE=${6:-$DEFAULT_RELEARNING_EXAMPLE_TYPE}
GENERATE_GRID=${7:-$DEFAULT_GENERATE_GRID}
PERFORM_LMC=${8:-$DEFAULT_PERFORM_LMC}
EVALUATE_PARAM_DIFF=${9:-$DEFAULT_EVALUATE_PARAM_DIFF}
INIT_SAFEGUARD=${10:-$DEFAULT_INIT_SAFEGUARD}
echo "Dataset: ${DATASET} | model name: ${MODEL_NAME} | method name: ${METHOD_NAME}"
echo "unlearning example selection criterion: ${UNLEARNING_EXAMPLE_SELECTION_CRITERION}"
echo "relearning example selection criterion: ${RELEARNING_EXAMPLE_SELECTION_CRITERION}"
echo "relearning example type: ${RELEARNING_EXAMPLE_TYPE}"
echo "init safeguard: ${INIT_SAFEGUARD}"

eval_after_steps=100
relearning_epochs=100
wandb_project="vision-unlearning"
EXTRA_ARGS=""
if [ "${GENERATE_GRID}" = "true" ]; then
    echo "Setting args for grid generation..."
    eval_after_steps=10
    relearning_epochs=10
    relearn_grid_eval_model="unlearned"
    relearn_grid_eval_ex="limited"
    if [ "${METHOD_NAME}" = "retrain_from_scratch" ]; then
        relearn_grid_eval_model=${METHOD_NAME}
        METHOD_NAME=${DEFAULT_METHOD_NAME}
    fi
    wandb_project="${wandb_project}-grid-relearning"
    EXTRA_ARGS="--generate-relearning-grid --relearn-grid-eval-ex ${relearn_grid_eval_ex} --relearn-grid-eval-model ${relearn_grid_eval_model}"
elif [ "${PERFORM_LMC}" = "true" ]; then
    echo "Setting args for linear mode connectivity analysis..."
    wandb_project="${wandb_project}-linear-mode-conn"
    EXTRA_ARGS="--evaluate-linear-mode-connectivity --mode-connectivity-stride 0.05"
elif [ "${EVALUATE_PARAM_DIFF}" = "true" ]; then
    echo "Setting args for parameter diff eval..."
    wandb_project="${wandb_project}-param-diff"
    EXTRA_ARGS="--evaluate-parameter-diff"
fi

# Set default method-specific hyperparameters
unlearning_epochs=100
unlearning_alpha=1
unlearning_gamma=1
unlearning_weight_decay=0.0
if [ "${METHOD_NAME}" = "tar" ]; then
    unlearning_epochs=25
    unlearning_gamma=4
elif [ "${METHOD_NAME}" = "catastrophic_forgetting" ]; then
    unlearning_weight_decay=0.001
elif [ "${METHOD_NAME}" = "weight_attenuation" ]; then
    unlearning_alpha=0.5
elif [ "${METHOD_NAME}" = "weight_distortion" ]; then
    unlearning_alpha=0.02
elif [ "${METHOD_NAME}" = "l1_sparse" ]; then
    unlearning_alpha=1.0
elif [ "${METHOD_NAME}" = "weight_dropout" ]; then
    unlearning_alpha=0.2
elif [ "${METHOD_NAME}" = "mode_connectivity" ]; then
    unlearning_alpha=0.001
    unlearning_gamma=50
elif [ "${METHOD_NAME}" = "weight_dist_reg" ]; then
    unlearning_alpha=1.0
fi
echo "Unlearning epochs: ${unlearning_epochs} / alpha: ${unlearning_alpha} / gamma: ${unlearning_gamma} / unlearning weight decay: ${unlearning_weight_decay}"

python vision_unlearner.py \
    --dataset-name ${DATASET} \
    --model-name ${MODEL_NAME} \
    --batch-size 128 \
    --eval-after-steps ${eval_after_steps} \
    --train-epochs 300 \
    --fraction-of-canaries 0.0 \
    --unlearning-method ${METHOD_NAME} \
    --unlearning-layer-idx "4,7" \
    --unlearning-alpha ${unlearning_alpha} \
    --unlearning-gamma ${unlearning_gamma} \
    --unlearning-epochs ${unlearning_epochs} \
    --fraction-to-unlearn 0.1 \
    --unlearning-learning-rate 1e-5 \
    --unlearning-weight-decay ${unlearning_weight_decay} \
    --unlearning-target-class "0" \
    --unlearning-example-selection-criterion ${UNLEARNING_EXAMPLE_SELECTION_CRITERION} \
    --relearning-example-selection-criterion ${RELEARNING_EXAMPLE_SELECTION_CRITERION} \
    --initial-safeguard-args ${INIT_SAFEGUARD} \
    --relearning-learning-rate 1e-5 \
    --relearning-weight-decay 0.0 \
    --relearning-epochs ${relearning_epochs} \
    --relearn-example-type ${RELEARNING_EXAMPLE_TYPE} \
    --wandb-project ${wandb_project} \
    --seed 43 \
    ${EXTRA_ARGS}
