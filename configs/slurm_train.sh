#!/bin/bash
#SBATCH -J test                         # 作业名
#SBATCH -p kslgexclu01                  # 队列名  使用whichpartition 查看
#SBATCH -N 1                            # 节点数量
#SBATCH --ntasks-per-node=1             # 每个节点的进程数
#SBATCH --cpus-per-task=88              # 每个进程使用的核心数
#SBATCH --gres=gpu:8                    # 每个节点申请的dcu数量
#SBATCH -o ./log/slurm-%j               # 作业输出
#SBATCH -e ./log/slurm-%j               # 作业输出
#SBATCH --exclusive  

module purge 
module load compiler/gnu/11.2.0
module load nvidia/cuda/11.6
export CUDA_HOME=/public/software/compiler/nvidia/cuda/11.6.0



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

echo Node: $nodes
echo Head Node:  $head_node
echo Node Array: {$nodes_array}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
echo head_node_ip
echo Node IP: $head_node_ip
echo SLURM_NODEID: $SLURM_NODEID

NODE_RANK=$SLURM_NODEID
echo $NODE_RANK

CONFIG=configs/users/mahdi/configs/batch7+9_pointpillars_light_6class_2.py
CONFIG=configs/users/mahdi/configs/batch7+9_pointpillars_light_6class.py

# CONFIG=$1
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node $SLURM_GPUS_ON_NODE  \
    --rdzv_backend c10d \
    --rdzv_id $RANDOM \
    --master_addr $head_node_ip \
    --master_port 29500 \
    tools/train.py \
    $CONFIG \
    --launcher pytorch
    # --resume work_dirs/batch5_pointpillars_heavy_v3_0_5class_cyclic/epoch_17.pth
