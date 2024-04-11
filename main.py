#%%
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import torch
import torch.utils

#%%
# HYPERPARAMETERS
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_LOAD = True

# %%
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights = weights).to(DEVICE)
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# %%
# 

# %%
for epoch in range(EPOCHS):
    