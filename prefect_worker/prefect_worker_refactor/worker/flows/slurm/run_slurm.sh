#!/bin/bash

# Check if all arguments are provided
if [ $# -ne 10 ]; then
    echo "Usage: $0 <job_name> <num_nodes> <partitions> <reservations> <max_time> <conda_env> <forward_ports> <submission_ssh_key> <python_file> <yaml_file>"
fi

# Assign arguments to variables
job_name=$1
num_nodes=$2
partitions=$3
reservations=$4
max_time=$5
conda_env=$6
forward_ports=$7
submission_ssh_key=$8
python_file=$9
yaml_file=${10}

# Create a temporary Slurm batch script
BATCH_SCRIPT=$(mktemp ./tmp_script.XXXXXX)

# Write the Slurm batch script
echo "#!/bin/bash" > $BATCH_SCRIPT

if [ -z "$partitions" ]
then
    echo "Partitions is unset or empty"
else
    echo "#SBATCH --partition=$partitions" >> $BATCH_SCRIPT
fi

if [ -z "$reservations" ]
then
    echo "Reservations is unset or empty"
else
    echo "#SBATCH --reservation=$reservations" >> $BATCH_SCRIPT
fi

JOB_SUBMISSION_HOST=$(hostname)

echo "#SBATCH --nodes=$num_nodes" >> $BATCH_SCRIPT
echo "#SBATCH --job-name=$job_name" >> $BATCH_SCRIPT
echo "#SBATCH --time=$max_time" >> $BATCH_SCRIPT
echo "unset LD_PRELOAD" >> $BATCH_SCRIPT
echo "source /etc/profile.d/modules.sh" >> $BATCH_SCRIPT
echo "module load maxwell mamba" >> $BATCH_SCRIPT
echo ". mamba-init" >> $BATCH_SCRIPT
echo "mamba activate $conda_env" >> $BATCH_SCRIPT
if [ ! -z "$forward_ports" ]; then
    echo -n "ssh " >> $BATCH_SCRIPT
    # If an ssh key was specified, pass it as an additional argument
    if [ ! -z "$submission_ssh_key" ]; then
        echo -n "-i $submission_ssh_key " >> $BATCH_SCRIPT
    fi
    # Loop over forward_ports and append each forwarding rule
    IFS=',' read -ra ADDR <<< "$forward_ports"
    for port_pair in "${ADDR[@]}"; do
        echo $port_pair
        local_port=${port_pair%:*}
        remote_port=${port_pair#*:}
        echo -n "-L $local_port:localhost:$remote_port " >> $BATCH_SCRIPT
    done
    # -N tells SSH to not execute a remote command
    echo -n "-N " >> $BATCH_SCRIPT
    # Finally, specify host to tunnel to
    # & at the end of the line makes the command run in the background
    echo "$USER@$JOB_SUBMISSION_HOST &" >> $BATCH_SCRIPT
fi
echo "srun python $python_file $yaml_file" >> $BATCH_SCRIPT
echo $BATCH_SCRIPT

# Submit the Slurm batch script and capture the job ID
JOB_ID=$(sbatch --export=ALL,JOB_SUBMISSION_HOST=$JOB_SUBMISSION_HOST,JOB_SUBMISSION_SSH_KEY="$submission_ssh_key" $BATCH_SCRIPT | awk '{print $4}')

# Print the job ID
echo "Submitted job with ID $JOB_ID"

# Track the progress of the job
while true; do
    # Get the job status
    JOB_STATUS=$(sacct -j $JOB_ID --format=State --noheader | head -1 | awk '{print $1}')

    if [[ $JOB_STATUS == *"COMPLETED"* ]]; then
        # If the job is completed, break the loop
        echo "Job $JOB_ID has completed"
        break
    elif [[ $JOB_STATUS == *"FAILED"* ]]; then
        # If the job has failed, print an error message and exit with a non-zero status
        echo "Job $JOB_ID has failed" >&2

        # Remove the temporary Slurm batch script
        rm $BATCH_SCRIPT

        exit 1
    else
        # If the job is neither completed nor failed, print its status
        echo "Job $JOB_ID is $JOB_STATUS"
    fi
    sleep 10
done

# Remove the temporary Slurm batch script
rm $BATCH_SCRIPT
