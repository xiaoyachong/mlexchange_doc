#!/bin/bash
source .env

# Check if all arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <conda_environment> <python_file> <yaml_file> <folder_name>"
    exit 1
fi

source "$CONDA_PATH/etc/profile.d/conda.sh"

# Assign arguments to variables
conda_environment=$1
python_file=$2
yaml_file=$3
folder_name=$4

# Change to algorithm directory if ALGORITHMS_DIR is set
if [ -n "$ALGORITHMS_DIR" ] && [ -n "$folder_name" ]; then
    cd "$ALGORITHMS_DIR/$folder_name" || exit 1
    echo "Working directory: $(pwd)"
fi

echo "Calling model with: conda run --no-capture-output -n $conda_environment python $python_file $yaml_file"

# Call python with the python file and yaml file as arguments
conda run --no-capture-output -n $conda_environment python $python_file $yaml_file

# Check if the Conda command was successful
if [ "$?" -ne 0 ]; then
    echo "Failed to run conda command"
    exit 1
fi

echo "Successfully run conda command"
exit 0
