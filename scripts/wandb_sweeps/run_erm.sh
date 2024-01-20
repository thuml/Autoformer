#!/bin/bash
set -x
# This script runs all the ERM Runs from the sweep files.
#Assumes running from repo base.

# Initialize variables
SWEEP_NAME=""
debug_mode=0

# Function to display usage
usage() {
    echo "Usage: $0 --name <name> [--debug] <SWEEP_FILE>"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            SWEEP_NAME="$2"
            shift # past argument
            shift # past value
            ;;
        --debug)
            debug_mode=1
            shift # past argument
            ;;
    esac
done

# if debug, use Autoformer-dev for project, else use Autoformer
if [ $debug_mode -eq 1 ]; then
    echo "Running in debug mode, using Autoformer-javierdev project"
    export WANDB_PROJECT="Autoformer-javierdev"
else
    echo "Running in normal mode, using Autoformer project"
    export WANDB_PROJECT="Autoformer"
fi

# Check required arguments
if [ -z "$SWEEP_NAME" ]; then
    usage
fi

# Your script logic here
echo "Name: $SWEEP_NAME"
echo "Debug mode: $debug_mode"

# Running each sweep three times for three seeds
for i in {1..3}
do
    echo "Running sweep with the following command..."
    echo "Weather ERM"
    #append name of seed
    wandb sweep --name "${SWEEP_NAME}-erm-weather-seed${i}" \
        --project $WANDB_PROJECT \
        sweeps/sweep_11_multimodel_stat_informed_erm_allpreds.yaml

    echo "Electricity ERM"
    wandb sweep --name "${SWEEP_NAME}-erm-electricity-seed${i}" \
        --project $WANDB_PROJECT \
        sweeps/sweep_12_multimodel_electricity_erm_allpreds.yaml

    echo "Exchange ERM"
    wandb sweep --name "${SWEEP_NAME}-erm-exchange-seed${i}" \
        --project $WANDB_PROJECT \
        sweeps/sweep_17_multimodel_exchange_erm_allpreds.yaml
done

set +x
