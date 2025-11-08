#!/bin/bash
#SBATCH --job-name=badge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/badge_%j.out
#SBATCH --error=logs/badge_%j.err
#SBATCH --partition=h200


set -euo pipefail

source /local/scratch/zyu273/miniconda3/etc/profile.d/conda.sh
# conda create -p /local/scratch/zyu273/badge/env python=3.10 -y
conda activate /local/scratch/zyu273/badge/env
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# pip3 install -U scikit-learn
# pip install numpy pandas tqdm 
# pip install openml


# parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
# parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
# parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
# parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
# parser.add_argument('--path', help='data path', type=str, default='data')
# parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
# parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
# parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
# parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=50000)
# parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=128)
# parser.add_argument('--rounds', help='number of rounds (0 does entire dataset)', type=int, default=0)
# parser.add_argument('--trunc', help='dataset truncation (-1 is no truncation)', type=int, default=-1)
# parser.add_argument('--aug', help='do augmentation (for cifar)', type=int, default=0)
# parser.add_argument('--dummy', help='dummy input for indexing replicates', type=int, default=1)

python run.py --alg badge --did 0 --lr 0.001  --model resnet --data data --nQuery 10000 --nStart 10000 --nEnd 50000 --nEmb 256 --trunc -1 --aug 1 --dummy 1 --data CIFAR10 --lr 0.001 
# python run.py --model mlp --nQuery 10000 --did 6 --alg bait