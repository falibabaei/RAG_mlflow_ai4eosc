#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38

export HOME=/hkfs/home/haicore/scc/se1131
source /hkfs/home/haicore/scc/se1131/tf_algos/bin/activate


export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_asyncp

 
 
 

cd /hkfs/home/haicore/scc/se1131/RAG_mlflow_ai4eosc
# Run FedAvg with different alpha values
#for alpha in 1.0 0.5 0.3 0.1; do
source .env
 
 
python scripts/log_model_mlflow.py