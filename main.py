#%%
import torch

#%%
# HYPERPARAMETERS
DEVICE = ['mps' if torch.backends.mps.is_available() else 'cpu']

LEARNING_RATE = 0.001
MODEL_LOAD = True

# %%

