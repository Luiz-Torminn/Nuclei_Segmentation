#%%
import matplotlib.pyplot as plt

# %%
def plot_loss(loss_values:dict, save_path:str, epochs:int, multi_graph:bool = False):
    fig, ax1 = plt.subplots(figsize = (8,8))
    
    for k, v in loss_values.items():
        ax1.plot([i for i in range(len(v))], v, label = k)
    
    ax1.legend()
    ax1.set_ylabel('Loss', labelpad=15)
    ax1.set_xlabel('Iterations', labelpad=15)
    
    plt.show()
    
    fig.savefig(f'{save_path}/loss_graph_{epochs}_epochs.png')
        
# %%
def plot_image_mask(image, mask, pred):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (8,8))
    
    ax1.plot(image)
    ax1.set_title('Original Image')
    
    ax2.plot(mask)
    ax2.set_title('Mask')
    
    ax3.plot(pred)
    ax3.set_title('Prediction')
    
# %%
# VALUES = {
#     'Train Loss':[5,4,3,2,2,2,2,2,2,2,2],
#     'Validation Loss':[5,4,3,3,3,3,3,3,3,3,3]
# }

# plot_loss(VALUES, 'data/saves/loss_graphs', 10)
