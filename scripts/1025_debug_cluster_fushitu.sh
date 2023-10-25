

#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

# python scripts/1025_debug_cluster_fushitu.py --cluster_size=500


cluster_sizes=(500 5000 10000 20000 30000 40000 50000)

# Iterate over cluster sizes
for size in "${cluster_sizes[@]}"; do
    echo "Running script with cluster_size=$size"
    python scripts/1025_debug_cluster_fushitu.py --cluster_size=$size
done