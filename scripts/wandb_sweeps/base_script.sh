#!/bin/bash
# This script runs a sweep, basically a wrapper for wandb sweep with a debug mode to change the project.


# Initialize variables
SWEEP_NAME=""
debug_mode=0
SWEEP_FILE=""

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
        *)
            SWEEP_FILE="$1" # Save the last argument as the SWEEP_FILE
            shift # past argument
            ;;
    esac
done

# if debug, use Autoformer-dev for project, else use Autoformer
if [ $debug_mode -eq 1 ]; then
    echo "Running in debug mode, using Autoformer-dev project"
    export WANDB_PROJECT="Autoformer-dev"
else
    echo "Running in normal mode, using Autoformer project"
    export WANDB_PROJECT="Autoformer"
fi

# Check required arguments
if [ -z "$SWEEP_NAME" ] || [ -z "$SWEEP_FILE" ]; then
    usage
fi

# Your script logic here
echo "Name: $SWEEP_NAME"
echo "Debug mode: $debug_mode"
echo "Sweep file: $SWEEP_FILE"

echo "Running sweep with the following command..."
echo "wandb sweep $SWEEP_FILE --name $SWEEP_NAME --project $WANDB_PROJECT"

wandb sweep $SWEEP_FILE --name $SWEEP_NAME --project $WANDB_PROJECT