#!/bin/bash
#SBATCH -JTranslationEval                    # Job name
#SBATCH -N1 --ntasks-per-node=1              # Number of nodes and tasks per node
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task
#SBATCH --mem=32G                            # Memory per node
#SBATCH --gres=gpu:A100:1                    # Request 1 A100 GPU (adjust as needed)
#SBATCH -t 24:00:00                          # Duration (hh:mm:ss)
#SBATCH --output=translation_eval_%j.out     # Standard output log (%j = job ID)
#SBATCH --error=translation_eval_%j.err      # Standard error log
#SBATCH --qos=coc-ice


# Load required modules
module purge
module load anaconda3/2023.03
module load cuda/11.8

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of cores: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"

# Set up environment
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate your existing conda environment
echo "Activating conda environment: llm_bti"
conda activate llm_bti

# Verify environment activation
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with the following variables:"
    echo "  PORTKEY_API_KEY=your_portkey_api_key"
    echo "  GROQ_VIRTUAL_KEY=your_groq_virtual_key"
    echo "  HUGGING_FACE_HUB_TOKEN=your_hf_token"
    exit 1
fi

# Run the Python script
echo "Starting translation evaluation pipeline..."
python translation_evaluation.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully on $(date)"
else
    echo "Job failed with exit code $? on $(date)"
    exit 1
fi

# Deactivate conda environment
conda deactivate

echo "Job finished on $(date)"














