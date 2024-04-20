#%%
import copy

# %%
class EarlyStopping():
    def __init__(self, patience:int, min_loss:float, verbose = False,  load_best_state = True):
        self.patience = patience
        self.min_loss = min_loss
        self.verbose = verbose
        self.load_best_model = load_best_state
        self.counter = 0
        self.best_model = None
        self.best_loss = None
        
    def __call__(self, model, current_loss:float) -> bool:
        if not self.best_model:
            self.best_loss = current_loss
            self.best_model = copy.deepcopy(model)
        
        if current_loss - self.best_loss < 0:
            self.counter = 0
            self.best_loss = current_loss
            self.best_model.load_state_dict(model.state_dict())
            
        if current_loss - self.best_loss >= 0:
            self.counter += 1
            
            if self.counter >= self.patience:
                print(f'\nModel stopped at counter = {self.counter}')
                
                if self.load_best_model:
                    model.load_state_dict(self.best_model.state_dict())
                
                return False
            
        if self.verbose:
            print(f'\nEarly stopping status --> [{self.counter}/{self.patience}]')
            
        return True
            
        
