import torch
from torch import nn
import wandb
from dataloaders.dataloader_images import Loader
import time
import sys
import os
import numpy as np
from metrics import Metrics
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from models.get_model import MLP_Final_Model
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from losses import get_loss 

class TrainModel():

	def __init__(self, options:dict):

		self.options        = options
		self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
		os.makedirs(os.path.join(options.result_dir, options.exp_name, "Saved_Models"), exist_ok=True)

		
		# Initialize wandb
		wandb.init(
			project = "INC-Classification-Images",
			entity  = "kevin-osorno-castillo",
			name    = options.exp_name,
			config  = vars(options),
			tags    = options.tag_exp,
		)

		# Initialize model, print summary and save configuration
		self.model = MLP_Final_Model(options)
		self.model.to(self.device)
  
  		# Get loss function
		if (options.class_balance):
			
			self.criterion = get_loss(
				name 			= options.loss,
				positive_weight = options.pos_weight,
				negative_weight = options.neg_weight,
				gamma 			= options.gamma
			)
   
			self.criterion.to(self.device)
			
		else:
			self.criterion = nn.CrossEntropyLoss()

		self.criterion.to(self.device)
		
		# Get loaders
		loader 				= Loader(options.images_dir, options.csv_data_path, options.augmentation)
		self.train_loader 	= loader.train_dataloader(batch_size=options.batch_size)
		self.val_loader 	= loader.val_dataloader(batch_size=options.batch_size)
		self.test_loader 	= loader.test_dataloader(batch_size=options.batch_size)

		# Define optimizer and lr scheduler
		self.optimizer 	= torch.optim.Adam(self.model.parameters(), lr=options.lr, betas=(options.b1, options.b2))
		self.metrics  	= Metrics(self.device)
		self.best_loss  = 0.
  
		self.early_stopping = EarlyStopping(
	  							patience 	= options.patience_early,
				  				dir_save	= os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models")
	   						)

		self.scheduler 		= CosineAnnealingLR(
	  							self.optimizer, 
	  							T_max 	= options.n_epochs,
	  							eta_min = options.min_lr
						 	)
  
		show_batch_images(self.train_loader, save_dir=os.path.join(self.options.result_dir, self.options.exp_name))
  
		if(options.test and options.best_model):
			self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", "Best_Model.pth")))
		elif(options.test and not options.best_model):
			self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", "Last_Model.pth")))
	
	def train_model(self):
		
		print("\n [*] -> Starting training....\n\n")

		self.prev_time = time.time()

		for self.epoch in range(self.options.init_epoch, self.options.n_epochs):

			self.epoch_stats = {
				"Loss-BCE"	: [],
				"Train_Accuracy"	: [],
				"Train_Sensitivity": [],
				"Train_Specificity": [],
				"Train_F1-Score"	: [],
				"Train_VPP"			: []
			}

			for batch_idx, data in enumerate(self.train_loader):

				# Get data
				inputs, data_clinic, targets   	= data
				inputs, data_clinic, targets    = inputs.to(self.device), data_clinic.to(self.device), targets.to(self.device)
				inputs, data_clinic, targets	= inputs.float(), data_clinic.float(), targets.long()

				if(batch_idx == 17):
					print(data_clinic.size())

				self.model.train()

				# Train Generator
				logits 		= self.model(inputs, data_clinic)
				loss 		= self.criterion(logits, targets)
				
				# Backpropagation
				self.optimizer.zero_grad()				
				loss.backward()
				self.optimizer.step()

				# Get predictions 
				probs 	= torch.softmax(logits, dim=1)
				preds  	= torch.argmax(probs, dim=1)

				# Update epoch stats
				self.epoch_stats["Loss-BCE"].append(loss.item())
				self.epoch_stats["Train_Accuracy"].append(self.metrics.accuracy(preds, targets.long()))
				self.epoch_stats["Train_Sensitivity"].append(self.metrics.sensitivity(preds, targets.long()))
				self.epoch_stats["Train_Specificity"].append(self.metrics.specificity(preds, targets.long()))
				self.epoch_stats["Train_F1-Score"].append(self.metrics.f1_score(preds, targets.long()))
				self.epoch_stats["Train_VPP"].append(self.metrics.vpp(preds, targets.long()))

				# Compute the elapsed time since the last log
				elapsed_time = time.time() - self.prev_time

				# Calculate the estimated time remaining for the epoch
				hours   = elapsed_time // 3600
				minutes = (elapsed_time % 3600) // 60
				seconds = elapsed_time % 60
				
				# Format the progress information
				progress_str = (
					f"\r[Epoch {self.epoch}/{self.options.n_epochs}] "
					f"[Batch {batch_idx}/{len(self.train_loader)}] "
					f"Lr {self.optimizer.param_groups[0]['lr']:.6f} "
					f"[BCE loss: {loss.item():.4f}] "
					f"ETA: {int(hours)}h{int(minutes)}m{int(seconds)}s"
				)

				# Write the progress information to the console
				sys.stdout.write(progress_str)

				# Move the cursor to the beginning of the line to overwrite the previous progress information
				sys.stdout.flush()
				sys.stdout.write('\r')
				sys.stdout.flush()
				
				if batch_idx % 5 == 0:
					step_log = {
						"Loss-BCE"	: loss.item(),
					}
					wandb.log(step_log)
	 
			# Log epoch stats
			for key, value in self.epoch_stats.items():
				self.epoch_stats[key] = torch.mean(torch.tensor(value)).item()
			
			self.epoch_stats["epoch"] = self.epoch
			self.epoch_stats["lr"] = self.optimizer.param_groups[0]['lr']
			wandb.log(self.epoch_stats)
			self.validation(plot=False)
			self.scheduler.step()

			if self.early_stopping.early_stop:
				print("Deteniendo entrenamiento temprano 🚨")

				# Compute the elapsed time since the last log
				elapsed_time = time.time() - self.prev_time

				# Calculate the estimated time remaining for the epoch
				hours   = elapsed_time // 3600
				minutes = (elapsed_time % 3600) // 60
			
				# Save training time to the configuration file
				with open(os.path.join(self.options.result_dir, self.options.exp_name, 'config.txt'), 'a') as f:
					f.write("\n--------------- Time Training ------------------\n")
					f.write("Tiempo de entrenamiento: {} horas y {} minutos\n".format(int(hours), int(minutes)))

				print("\n [✓] -> Done Training! \n\n")
	
				break

			# Save the model after every epoch_chkpt
			if self.epoch % 10 == 0:
	   
				dir_save = os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models")
				os.makedirs(dir_save, exist_ok=True)

				# torch.save(self.model.state_dict(), os.path.join(dir_save, f"{self.options.image_model}_{self.epoch:03d}.pth"))
				torch.save(self.model.classifier.state_dict(), os.path.join(dir_save, f"classifier_{self.epoch:03d}.pth"))
				torch.save(self.model.state_dict(), os.path.join(dir_save, f"Model_{self.epoch:03d}.pth"))

		# Compute the elapsed time since the last log
		elapsed_time = time.time() - self.prev_time

		# Calculate the estimated time remaining for the epoch
		hours   = elapsed_time // 3600
		minutes = (elapsed_time % 3600) // 60
	
		# Save training time to the configuration file
		with open(os.path.join(self.options.result_dir, self.options.exp_name, 'config.txt'), 'a') as f:
			f.write("\n--------------- Time Training ------------------\n")
			f.write("Tiempo de entrenamiento: {} horas y {} minutos\n".format(int(hours), int(minutes)))

		print("\n [✓] -> Done Training! \n\n")
	

	def validation(self, plot=False):

		metrics_img = {
			"Val_Accuracy"       : [],
			"Val_Sensitivity"    : [],
			"Val_Specificity"    : [],
			"Val_F1-Score"       : [],
   			"Val_BCE-Loss"   	 : [],
			"Val_VPP"            : []
	  
		}

		# Set the generator to training mode
		self.model.eval()

		with torch.no_grad():

			# Iterate over the training data
			for batch_idx, data in enumerate(self.val_loader):

				inputs, data_clinic, targets   	= data
				inputs, data_clinic, targets    = inputs.to(self.device), data_clinic.to(self.device), targets.to(self.device)
				inputs, data_clinic, targets	= inputs.float(), data_clinic.float(), targets.long()

				# Get predictions
				logits 		= self.model(inputs, data_clinic)

				
				probs 	= torch.softmax(logits, dim=1)
				preds  	= torch.argmax(probs, dim=1)

				# Calculate the metrics
				metrics = self.metrics.get_metrics(preds, targets.long(), probs, "Val")
				
				for key, value in metrics.items():
					metrics_img[key].append(value)

		for key, value in metrics_img.items():
			self.epoch_stats[key] = torch.mean(torch.tensor(value)).item()

		self.early_stopping(self.epoch_stats["Val_F1-Score"], self.model, self.epoch)
		self.epoch_stats["epoch"] = self.epoch_stats["epoch"]
		wandb.log(self.epoch_stats)

	def test_model(self):
     
		# Load the best model
		self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", f"Best_Model.pth")))
		print("Modelo cargado Mejor")
		metrics_img = {
			"Test_Accuracy"       : [],
			"Test_Sensitivity"    : [],
			"Test_Specificity"    : [],
			"Test_F1-Score"       : [],
   			"Test_BCE-Loss"   	: [],
			"Test_VPP"            : []
		}

		# Set the generator to training mode
		self.model.eval()

		all_targets = []
		all_probs   = []

		with torch.no_grad():

			# Iterate over the training data
			for batch_idx, data in enumerate(self.test_loader):

				inputs, data_clinic, targets   	= data
				inputs, data_clinic, targets    = inputs.to(self.device), data_clinic.to(self.device), targets.to(self.device)
				inputs, data_clinic, targets	= inputs.float(), data_clinic.float(), targets.long()

				# Train Generator
				logits 		= self.model(inputs, data_clinic)
				probs 		= torch.softmax(logits, dim=1)
				preds  		= torch.argmax(probs, dim=1)
	
				# Acumulamos para ROC y confusion
				all_targets.append(targets.view(-1).cpu().numpy())
				all_probs.append(probs.cpu().numpy())

				# Calculate the metrics
				metrics = self.metrics.get_metrics(preds, targets.long(), probs, "Test")
				
				for key, value in metrics.items():
					try:
						metrics_img[key].append(value.item())
					except:
						metrics_img[key].append(value)
   
		# Convertir listas planas
		y_true = np.concatenate(all_targets)
		y_prob = np.concatenate(all_probs)
		y_pred = np.argmax(y_prob, axis=1)
		#y_pred = (y_prob > 0.5).astype(int)

		dir_save = os.path.join(self.options.result_dir, self.options.exp_name, "Results")
		os.makedirs(dir_save, exist_ok=True)
  
		# Calcular la media y desviación estándar de las métricas
		summary = {
			metric: {
				"mean": np.mean(values),
				"std":  np.std(values, ddof=0)  # ddof=0 para población, ajusta si prefieres muestra
			}
			for metric, values in metrics_img.items()
		}

		# 2) Crea un DataFrame orientado por índice
		df_summary = pd.DataFrame.from_dict(summary, orient="index")
		df_summary.index.name = "Metric"

		# 3) Guarda en CSV
		csv_path = os.path.join(dir_save, "test_summary.csv")
		df_summary.to_csv(csv_path)

		# Subir en wandb
		wandb.log({
			"Test_Accuracy"       : summary["Test_Accuracy"]["mean"],
			"Test_Sensitivity"    : summary["Test_Sensitivity"]["mean"],
			"Test_Specificity"    : summary["Test_Specificity"]["mean"],
			"Test_F1-Score"       : summary["Test_F1-Score"]["mean"],
			"Test_BCE-Loss"   	: summary["Test_BCE-Loss"]["mean"],
			"Test_VPP"            : summary["Test_VPP"]["mean"]
		})

		print(f"✅ Resumen de métricas guardado en {csv_path}")
		print(df_summary)

		# --- 2) Matriz de confusión ---
		cm = confusion_matrix(y_true, y_pred, labels=[0,1])
		plt.figure(figsize=(10,8))
		sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
					xticklabels=["No Cancer=0","Cancer=1"],
					yticklabels=["No Cancer=0","Cancer=1"])
		plt.xlabel("Predicho")
		plt.ylabel("Verdadero")
		plt.title("Matriz de Confusión")
		plt.tight_layout()
		path_cm = os.path.join(dir_save, "confusion_matrix.png")
		plt.savefig(path_cm, dpi=300)
		print(f"✅ Matriz de confusión guardada en {path_cm}")
		#plt.show()

		# --- 3) Curva ROC–AUC ---
		fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
		roc_auc = auc(fpr, tpr)

		plt.figure(figsize=(6,5))
		plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
		plt.plot([0,1],[0,1], 'k--', label="Aleatorio")
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.title("Curva ROC")
		plt.legend(loc="lower right")
		plt.tight_layout()
		#plt.show()
  
		# Save ROC AUC to CSV
		path_roc = os.path.join(dir_save, "roc_curve.png")
		plt.savefig(path_roc, dpi=300)

		print(f"✅ ROC–AUC: {roc_auc:.3f}")


import matplotlib.pyplot as plt
import torch



def show_batch_images(train_dataloader, num_images=10, save_dir=None):
    
    images_shown = 0

    plt.figure(figsize=(15, 6))

    for images, data_clinic, labels in train_dataloader:
        # images shape: [B, C, H, W]
        batch_size = images.shape[0]

        for i in range(batch_size):
            if images_shown >= num_images:
                plt.savefig(os.path.join(save_dir, "batch_images.png"), dpi=300)
                return

            img = images[i]
            print(img.size())

            # Convertir a CPU numpy
            img_np = img[0,:,:].detach().cpu().numpy()

            plt.subplot(2, 5, images_shown + 1)
            plt.imshow(img_np, cmap="gray", vmin=-1, vmax=1)
            plt.title(f"Label: {labels[i].item()}")
            plt.axis("off")

            images_shown += 1
