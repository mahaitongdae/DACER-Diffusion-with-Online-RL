# ReLax
Reinforcement Learning with JAX.

## Installation

```bash
# Create environemnt
conda create -n relax python=3.11 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba
conda activate relax

# One of: Install jax WITH CUDA 
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install package
pip install -r requirements.txt
pip install -e .
```

## Run
```bash
# Run one experiment
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python scripts/train_mujoco.py --alg dacer --seed 100
```

```bash
# Run multiple experiments
chmod +x run_experiments.sh
./run_experiments.sh "0 1 2"
```

## Tips
search "other envs" in the code to find the parameters for other mujoco environments. The existing parameters in the code are used for Humanoid-v3 and HalfCheetah-v3.