#!/bin/bash
#SBATCH -J vit-b_16_ve
# Define a variable with the current date
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodelist=leon08

# Load Singularity
# module load shared singularity
# singularity exec --nv /home/sebastian/qve/cuda6.img nvidia-smi
# singularity exec --nv /home/sebastian/qve/cuda6.img python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
# Run Python script inside the Singularity image
# singularity exec /home/sebastian/qve/cuda6.img python3 cross_validation_clip.py --qubits 16 --output_dir ../EXPERIMENTS/vit-b_16_ve --folds 5 --total_train_samples 1600 --total_test_samples 400
# singularity exec /home/sebastian/qve/cuda6.img python3 cross_validation_clip.py --data_path ../data/mnist_embeddings/vit-l_14_at_336px_ve_1536/stacked/ --qubits 16 --output_dir ../EXPERIMENTS/vit-l_14_at_336px_ve/ --folds 5 --total_train_samples 1600 --total_test_samples 400
# singularity exec /home/sebastian/qve/cuda6.img python3 cross_validation_clip.py --data_path ../data/mnist_embeddings/vit-l_14_ve_1536/stacked/
#  --qubits 16 --output_dir ../EXPERIMENTS/vit-l_14_ve
# / --folds 5 --total_train_samples 1600 --total_test_samples 400

# # Baseline
# python qsvm_cuda_embeddings.py --data_path ../data/mnist_embeddings/baseline --qubits 2 --output_dir ../EXPERIMENTS/delete --folds 5 --total_train_samples 200 --total_test_samples 50

# 
mpirun -np 2 python3 qsvm_cuda_embeddings.py  --data_path ../data/fashionmnist_embeddings/efficientnet_512/ --qubits 16 --output_dir ../EXPERIMENTS/efficientnet_512_fmnist --folds 5 --total_train_samples 1600 --total_test_samples 400


# # Baseline (opt)
# python cross_validation_baseline-opt.py --data_path ../data/FashionMNIST_embeddings --qubits 16 --output_dir ../EXPERIMENTS/Baseline-opt_fmnist --folds 5 --total_train_samples 1600 --total_test_samples 400

# # Ours: EffNet-512 
# python cross_validation_ve.py --data_path ../data/FashionMNIST_embeddings/enet_ve/stacked/ --qubits 16 --output_dir ../EXPERIMENTS/EffNet-512_fmnist --folds 5 --total_train_samples 1600 --total_test_samples 400

# # Ours: EffNet-1536 
# python cross_validation_ve.py --data_path ../data/FashionMNIST_embeddings/efficientnet_ve_1536/stacked/ --qubits 16 --output_dir ../EXPERIMENTS/EffNet-1536_fmnist --folds 5 --total_train_samples 1600 --total_test_samples 400

# # Ours: EffNet-1536 
# python cross_validation_ve.py --data_path ../data/FashionMNIST_embeddings/efficientnet_ve_1536/stacked/ --qubits 16 --output_dir ../EXPERIMENTS/EffNet-1536_fmnist --folds 5 --total_train_samples 1600 --total_test_samples 400

# # Ours: CLIP-512 --> todo!
# python cross_validation_ve.py --data_path ../data/FashionMNIST_embeddings/clip_ve_1536/stacked/ --qubits 16 --output_dir ../EXPERIMENTS/Ours: CLIP-512_fmnist --folds 5 --total_train_samples 1600 --total_test_samples 400