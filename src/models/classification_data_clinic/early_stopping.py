import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience=10, delta=0.1, dir_save=None):
        """
        Args:
            patience (int): Número de épocas que espera sin mejora antes de detener.
            delta (float): Mínima mejora en la métrica para considerarla como mejora.
            dir_save (str): Ruta para guardar el mejor modelo.
        """
        self.patience       = patience
        self.delta          = delta
        self.counter        = 0
        self.best_score     = 0.0
        self.early_stop     = False
        self.dir_save       = dir_save

    def __call__(self, metric_value, model, epoch):
         
        # Queremos maximizar F1-score u otra métrica
        if metric_value < self.best_score + self.delta:
            
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = metric_value
            self.save_checkpoint(metric_value, model, epoch)
            self.counter = 0

    def save_checkpoint(self, metric_value, model, epoch):
        
        torch.save(model.state_dict(), os.path.join(self.dir_save, f"Best_Model.pth"))
       
        
        print(f'Model {epoch} saved with {metric_value:.4f} metric value')
        self.best_metric = metric_value