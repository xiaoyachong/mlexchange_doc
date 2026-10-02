#!/bin/bash

# Check if an argument was provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <image_uri> <command> [volumes] [network] [\"env_var1 env_var2 ...\"]"
    exit 1
fi

image_uri=$1
command=$2
volumes=$3
network=$4
env_vars=$5

# Start building the Podman command
cmd="podman run"

# If environment variables were provided, add them to the command
if [ -n "$env_vars" ]; then
    IFS=' ' read -ra vars <<< "$env_vars"
    for var in "${vars[@]}"; do
        cmd="$cmd -e $var"
    done
fi

# If a network was provided, add it to the command
if [ -n "$network" ]; then
    cmd="$cmd --network=$network"
fi

# If volumes were provided, add them to the command
if [ -n "$volumes" ]; then
    IFS=' ' read -ra volume_list <<< "$volumes"
    for volume in "${volume_list[@]}"; do
        cmd="$cmd -v $volume"
    done
fi

# Add the image and command to the Podman command
cmd="$cmd $image_uri /bin/sh -c \"$command\""

# Run the Podman command
eval $cmd

# Check if the Podman command was successful
if [ "$?" -ne 0 ]; then
    echo "Failed to run Podman container"
    exit 1
fi

echo "Successfully launched Podman container"
exit 0
