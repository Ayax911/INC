import torch
from torch import nn
import wandb
from pathlib import Path
from dataloaders.dataloader_images import Loader
import time
import sys
import os
import numpy as np


def _wandb_credentials_cached() -> bool:
    """True si hay una sesión de `wandb login` ya cacheada en `~/.netrc`.

    Evita que wandb.init() bloquee pidiendo login interactivo o falle si no
    hay credenciales configuradas: si no hay sesión cacheada, la corrida cae
    a modo "offline" en vez de fallar/preguntar (ver también
    src/tracking.py:_wandb_credentials_cached en FedMammoBench).
    """
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.is_file():
        return False
    return "api.wandb.ai" in netrc_path.read_text()
from metrics import Metrics, build_metric_collection
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib
# Backend no interactivo: todos los plt.* de este módulo solo hacen
# savefig(), nunca show(). Sin esto, matplotlib autodetecta "qtagg" porque
# PySide6 está instalado (dependencia de la GUI, sin relación con esto) y
# crashea con SIGSEGV al faltar libxcb-cursor0 en el sistema.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import reporting
from models.get_model import get_model
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from losses import get_loss
from reporting_extras import EpochLogger, compute_extra_metrics, save_metrics_json, save_predictions_csv

class TrainModel():
	
	"""
	Class to train and test the model
	Args:
		options (dict): Dictionary with the training options
	"""

	def __init__(self, options:dict):
		
		# Set options
		self.options        = options
		self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		os.makedirs(os.path.join(options.result_dir, options.exp_name, "Saved_Models"), exist_ok=True)

		# metrics.csv por época + TensorBoard, adicional a lo que ya se loguea a wandb
		# más abajo -- ver reporting_extras.EpochLogger.
		self.epoch_logger = EpochLogger(os.path.join(options.result_dir, options.exp_name))

		
		# Initialize wandb -- sin `entity` fijo: usa la cuenta con la que se haya
		# corrido `wandb login` en esta máquina. Sin sesión cacheada, cae a
		# modo offline en vez de bloquear pidiendo login o fallar.
		wandb.init(
			project = options.wandb_project,
			name    = options.exp_name,
			group   = options.wandb_group,
			config  = vars(options),
			tags    = options.tag_exp,
			mode    = "online" if _wandb_credentials_cached() else "offline",
		)

		# Initialize model, print summary and save configuration
		self.model = get_model(options)
		self.model.to(self.device)
  
  		# Get loss function
		if (options.class_balance):
			
			self.criterion = get_loss(
				name = options.loss,
				positive_weight = options.pos_weight,
				negative_weight = options.neg_weight,
				gamma 			= options.gamma
			)
   
			self.criterion.to(self.device)
			
		else:
			self.criterion = nn.CrossEntropyLoss()
			self.criterion.to(self.device)
		
		# Get loaders
		loader 				= Loader(options.images_dir, options.csv_data_path, options.augmentation, options.img_size, normalize_mean=options.normalize_mean, normalize_std=options.normalize_std)
		self.loader 		= loader  # evaluate_by_database() lee las rutas de loader.{val,test}_dataset.data
		self.train_loader 	= loader.train_dataloader(batch_size=options.batch_size)
		self.val_loader 	= loader.val_dataloader(batch_size=options.batch_size)
		self.test_loader 	= loader.test_dataloader(batch_size=options.batch_size)

		# Define optimizer and lr scheduler
		self.optimizer 	= torch.optim.Adam(self.model.parameters(), lr=options.lr, betas=(options.b1, options.b2))
		self.metrics  	= Metrics(self.device)
		self.best_loss  = 0.

		# Registro con el formato de FedMammoBench (ver reporting.py): metrics.csv, plots/,
		# val/ y test/ bajo <result_dir>/<exp_name>/, y en W&B las mismas claves train_*/val_*
		# con step = época, para que las curvas de ambos proyectos caigan en los mismos paneles.
		self.run_dir 		= Path(options.result_dir) / options.exp_name
		self.history 		= []  # un dict por época, mismo contenido que metrics.csv
		self.metrics_csv 	= reporting.MetricsCsvWriter(self.run_dir / "metrics.csv")
  
		self.early_stopping = EarlyStopping(
	  							patience 	= options.patience_early,
				  				dir_save	= os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models")
	   						)

		self.scheduler 		= CosineAnnealingLR(
	  							self.optimizer, 
	  							T_max 	= options.n_epochs,
	  							eta_min = options.min_lr
						 	)
  
		#show_batch_images(self.train_loader, save_dir=os.path.join(self.options.result_dir, self.options.exp_name))
  
		if(options.test and options.best_model):
			self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", "Best_Model.pth")))
		elif(options.test and not options.best_model):
			self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", "Last_Model.pth")))
	
	def train_model(self):
		
		print("\n [*] -> Starting training....\n\n")

		self.prev_time = time.time()

		for self.epoch in range(self.options.init_epoch, self.options.n_epochs):

			epoch_start 		= time.time()
			train_collection 	= build_metric_collection(self.device)
			train_loss_total 	= 0.0
			n_train_batches 	= 0

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
				inputs, targets, = data

				# image = inputs.detach().cpu()
				# labels = targets.detach().cpu()

				# img = image.numpy()

				# for i in range(img.shape[0]):
				# 	image_i = img[0,0,...]
				# 	plt.imshow(image_i, cmap="gray", vmin=-1, vmax=1)
				# 	plt.xlabel(f"Min: {image_i.min()}, Max: {image_i.max()}\nLabel:labels[0].item(), batch:{i}")
				# 	plt.axis("off")

				# 	save_path = os.path.abspath(f"debug_batch_{batch_idx}.png")
				# 	plt.savefig(save_path)
				# 	plt.close()
				# 	print(f"[batch {batch_idx}] Guardado en: {save_path}")

				# Convertir los inputs en un array
				# Graficar la imagen.
				# Valor minimo y el valor maximo de intensidad
				# Verificar (imprimir) la etiqueta
				inputs, targets     = inputs.to(self.device), targets.to(self.device)
				inputs, targets		= inputs.float(), targets.long()

				self.model.train()

				# Train model
				logits 		= self.model(inputs)
				loss 		= self.criterion(logits, targets)
				
				# Backpropagation
				self.optimizer.zero_grad()				
				loss.backward()
				self.optimizer.step()

				# Get predictions 
				probs 	= torch.softmax(logits, dim=1)
				preds  	= torch.argmax(probs, dim=1)

				# Métricas de época estilo FedMammoBench: probabilidad de la clase positiva,
				# acumulada en toda la época (ver build_metric_collection en metrics.py).
				with torch.no_grad():
					train_collection.update(probs[:, 1].detach(), targets)
				train_loss_total += loss.item()
				n_train_batches  += 1

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
	 
			# Promedios por batch de INC -- ya no van a W&B, se conservan solo porque
			# validation() los sigue acumulando en el mismo dict.
			for key, value in self.epoch_stats.items():
				self.epoch_stats[key] = torch.mean(torch.tensor(value)).item()

			self.epoch_stats["epoch"] = self.epoch
			self.epoch_stats["lr"] = self.optimizer.param_groups[0]['lr']
			self.validation(plot=False)
			wandb.log(self.epoch_stats)
			self.epoch_logger.log(self.epoch, self.epoch_stats)
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

				torch.save(self.model[0].state_dict(), os.path.join(dir_save, f"{self.options.image_model}_{self.epoch:03d}.pth"))
				torch.save(self.model[1].state_dict(), os.path.join(dir_save, f"classifier_{self.epoch:03d}.pth"))
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

		self.epoch_logger.close()

	def validation(self, plot=False):

		metrics_img = {
			"Val_Accuracy"       : [],
			"Val_Sensitivity"    : [],
			"Val_Specificity"    : [],
			"Val_F1-Score"       : [],
   			"Val_BCE-Loss"   	 : [],
			"Val_VPP"            : []
	  
		}

		val_collection 	= build_metric_collection(self.device)
		val_loss_total 	= 0.0
		n_val_batches 	= 0

		# Set the generator to training mode
		self.model.eval()

		with torch.no_grad():

			# Iterate over the training data
			for batch_idx, data in enumerate(self.val_loader):

				inputs, targets  	= data
				inputs, targets     = inputs.to(self.device), targets.to(self.device)
				inputs, targets		= inputs.float(), targets.float()

				# Get predictions
				logits 		= self.model(inputs)

				
				probs 	= torch.softmax(logits, dim=1)
				preds  	= torch.argmax(probs, dim=1)

				# val_loss con el MISMO criterio (ponderado) que train_loss, como FedMammoBench.
				# El "Val_BCE-Loss" de Metrics.get_metrics es otra cosa: CrossEntropy sin pesos
				# aplicada sobre probabilidades ya softmaxeadas.
				val_collection.update(probs[:, 1], targets.long())
				val_loss_total += self.criterion(logits, targets.long()).item()
				n_val_batches  += 1

				# Calculate the metrics
				metrics = self.metrics.get_metrics(preds, targets.long(), probs, "Val")
				
				for key, value in metrics.items():
					metrics_img[key].append(value)

		for key, value in metrics_img.items():
			self.epoch_stats[key] = torch.mean(torch.tensor(value)).item()

		# El criterio de parada NO cambia: sigue siendo el promedio por batch de Val F1 de INC.
		self.early_stopping(self.epoch_stats["Val_F1-Score"], self.model, self.epoch)

		val_metrics = {f"val_{k}": v.item() for k, v in val_collection.compute().items()}
		val_metrics["val_loss"] = val_loss_total / n_val_batches
		return val_metrics

	def _predict(self, loader):
		"""Predicciones completas de un loader SIN shuffle: y_true, y_pred, y_prob, logits y métricas estilo FMB."""
		collection 	= build_metric_collection(self.device)
		loss_total 	= 0.0
		n_batches 	= 0
		y_true, y_pred, y_prob, all_logits = [], [], [], []

		self.model.eval()
		with torch.no_grad():
			for inputs, targets in loader:
				inputs, targets = inputs.to(self.device).float(), targets.to(self.device).long()
				logits 	= self.model(inputs)
				probs 	= torch.softmax(logits, dim=1)
				preds 	= torch.argmax(probs, dim=1)

				collection.update(probs[:, 1], targets)
				loss_total += self.criterion(logits, targets).item()
				n_batches  += 1

				y_true.extend(targets.cpu().tolist())
				y_pred.extend(preds.cpu().tolist())
				y_prob.extend(probs[:, 1].cpu().tolist())
				all_logits.append(logits.cpu())

		metrics = {k: v.item() for k, v in collection.compute().items()}
		metrics["loss"] = loss_total / n_batches
		return y_true, y_pred, y_prob, torch.cat(all_logits), metrics

	def evaluate_split(self, split_name, loader, dataset):
		"""Equivalente a eval_pipeline.evaluate_split() de FedMammoBench, más su desglose por base de datos.

		Escribe en <run_dir>/<split_name>/: metrics.json, confusion_matrix_metrics.json,
		predictions.csv, confusion_matrix.png, roc_curve.png y predictions_<db>.csv. En test
		además metrics_by_database.json, confusion_matrix_by_database.png y
		metrics_by_database.png. A W&B: summary <split>_*, tabla <split>/predictions e imágenes
		<split>/* -- mismas claves que FedMammoBench.
		"""
		y_true, y_pred, y_prob, logits, split_metrics = self._predict(loader)
		cm_metrics = reporting.compute_confusion_matrix_metrics(y_true, y_pred)
		wandb.run.summary.update({f"{split_name}_{k}": v for k, v in {**split_metrics, **cm_metrics}.items()})

		split_dir = self.run_dir / split_name
		reporting.save_metrics_json(split_metrics, split_dir / "metrics.json")
		reporting.save_metrics_json(cm_metrics, split_dir / "confusion_matrix_metrics.json")

		predictions_path = split_dir / "predictions.csv"
		reporting.save_predictions_csv(y_true, y_pred, y_prob, predictions_path)
		wandb.log({f"{split_name}/predictions": wandb.Table(dataframe=pd.read_csv(predictions_path))})

		confusion_matrix_path = split_dir / "confusion_matrix.png"
		reporting.plot_confusion_matrix(y_true, y_pred, confusion_matrix_path)
		wandb.log({f"{split_name}/confusion_matrix": wandb.Image(str(confusion_matrix_path))})

		roc_curve_path = split_dir / "roc_curve.png"
		reporting.plot_roc_curve(y_true, y_prob, roc_curve_path)
		wandb.log({f"{split_name}/roc_curve": wandb.Image(str(roc_curve_path))})

		# Desglose por base de datos: la base es la carpeta de la imagen en la CSV
		# (norm_neg1_1/<db>/<archivo>.tiff). Válido porque val/test no se barajan, así que
		# la posición i de las predicciones es la fila i de dataset.data.
		db_names = [Path(p).parent.name for p in dataset.data.iloc[:, 0]]
		assert len(db_names) == len(y_true), "el loader debe recorrer el dataset completo y sin shuffle"

		metrics_by_db, cm_by_db = {}, {}
		for db_name in dict.fromkeys(db_names):
			idx = [i for i, name in enumerate(db_names) if name == db_name]
			db_true = [y_true[i] for i in idx]
			db_pred = [y_pred[i] for i in idx]
			db_prob = [y_prob[i] for i in idx]
			reporting.save_predictions_csv(db_true, db_pred, db_prob, split_dir / f"predictions_{db_name}.csv")

			if split_name != "test":
				continue  # en val solo el CSV, para calibrar umbrales por base (igual que FMB)

			collection = build_metric_collection("cpu")
			collection.update(torch.tensor(db_prob), torch.tensor(db_true))
			db_metrics = {k: v.item() for k, v in collection.compute().items()}
			db_metrics["loss"] = self.criterion(logits[idx].to(self.device), torch.tensor(db_true, device=self.device)).item()
			db_cm = reporting.compute_confusion_matrix_metrics(db_true, db_pred)

			wandb.run.summary.update({f"test_by_database_{db_name}_{k}": v for k, v in {**db_metrics, **db_cm}.items()})
			metrics_by_db[db_name] = {**db_metrics, **db_cm}
			cm_by_db[db_name] = (db_true, db_pred)

		if metrics_by_db:
			reporting.save_metrics_by_database_json(metrics_by_db, split_dir / "metrics_by_database.json")

			cm_by_db_path = split_dir / "confusion_matrix_by_database.png"
			reporting.plot_confusion_matrix_by_database(cm_by_db, cm_by_db_path)
			wandb.log({"test/confusion_matrix_by_database": wandb.Image(str(cm_by_db_path))})

			metrics_by_db_path = split_dir / "metrics_by_database.png"
			reporting.plot_metrics_by_database(metrics_by_db, metrics_by_db_path, metric_display_names=reporting.METRIC_DISPLAY_NAMES)
			wandb.log({"test/metrics_by_database": wandb.Image(str(metrics_by_db_path))})

		print(f"{split_name.capitalize()}: {split_metrics}")
		return split_metrics

	def test_model(self):

		# Load the best model
		self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", f"Best_Model.pth")))
		print("Modelo cargado Mejor")
		self._evaluate_and_report(self.test_loader, "Test", "Results")

	def validate_model_full(self):
		"""Evaluación completa de validación con el mejor checkpoint, igual que `test_model()`.

		A diferencia de `validation()` (llamada por época dentro de `train_model()`,
		solo para decidir early stopping), esto corre una sola vez al final,
		siempre recargando `Best_Model.pth` -- nunca el estado en el que haya
		quedado `self.model` al terminar el entrenamiento -- y produce en
		`Results_Val/` los mismos artefactos que `test_model()` deja en `Results/`.
		"""

		self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", f"Best_Model.pth")))
		print("Modelo cargado Mejor (evaluación completa de validación)")
		self._evaluate_and_report(self.val_loader, "Val", "Results_Val")

	def _evaluate_and_report(self, loader, stage, dir_name):
		"""Evalúa `self.model` (ya con los pesos del mejor checkpoint) sobre `loader` y persiste/reporta todo.

		Misma lógica que ya usaba `test_model()` antes de esta factorización --
		reusada acá para no duplicarla al agregar `validate_model_full()`. Los
		artefactos que ya existían (test_summary.csv, confusion_matrix.png,
		roc_curve.png, el wandb.log con las 6 métricas) quedan exactamente
		iguales para `stage="Test"`; lo nuevo (metrics.json,
		confusion_matrix_metrics.json, predictions.csv, imágenes/tabla subidas
		a W&B) se agrega al final sin tocar lo anterior.

		Args:
			loader: DataLoader a evaluar (`self.test_loader` o `self.val_loader`).
			stage: prefijo de las métricas ("Test" o "Val") -- debe coincidir con
				las claves que devuelve `self.metrics.get_metrics()`.
			dir_name: subcarpeta bajo `<result_dir>/<exp_name>/` donde se guardan
				los artefactos ("Results" o "Results_Val").
		"""

		metrics_img = {
			f"{stage}_Accuracy"    : [],
			f"{stage}_Sensitivity" : [],
			f"{stage}_Specificity" : [],
			f"{stage}_F1-Score"    : [],
			f"{stage}_BCE-Loss"    : [],
			f"{stage}_VPP"         : []
		}

		# Set the generator to training mode
		self.model.eval()

		all_targets = []
		all_probs   = []

		with torch.no_grad():

			# Iterate over the training data
			for batch_idx, data in enumerate(loader):

				inputs, targets  	= data
				inputs, targets     = inputs.to(self.device), targets.to(self.device)
				inputs, targets		= inputs.float(), targets.float()

				# Train Generator
				logits 		= self.model(inputs)
				probs 		= torch.softmax(logits, dim=1)
				preds  		= torch.argmax(probs, dim=1)

				# Acumulamos para ROC y confusion
				all_targets.append(targets.view(-1).cpu().numpy())
				all_probs.append(probs.cpu().numpy())

				# Calculate the metrics
				metrics = self.metrics.get_metrics(preds, targets.long(), probs, stage)

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

		dir_save = os.path.join(self.options.result_dir, self.options.exp_name, dir_name)
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
		csv_path = os.path.join(dir_save, f"{stage.lower()}_summary.csv")
		df_summary.to_csv(csv_path)

		# Subir en wandb
		wandb.log({metric: values["mean"] for metric, values in summary.items()})

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

		# --- Artefactos adicionales: metrics.json, confusion_matrix_metrics.json,
		# predictions.csv, e imágenes/tabla subidas al summary de W&B. No
		# reemplazan nada de lo de arriba -- ver reporting_extras.py. ---
		save_metrics_json({k: v["mean"] for k, v in summary.items()}, os.path.join(dir_save, "metrics.json"))

		extra_metrics = compute_extra_metrics(y_true, y_pred, y_prob[:, 1])
		save_metrics_json(extra_metrics, os.path.join(dir_save, "confusion_matrix_metrics.json"))

		predictions_path = os.path.join(dir_save, "predictions.csv")
		save_predictions_csv(y_true, y_pred, y_prob[:, 1], predictions_path)

		wandb.summary.update({f"{stage}_{k}": v for k, v in extra_metrics.items()})
		wandb.log({
			f"{stage}/confusion_matrix": wandb.Image(path_cm),
			f"{stage}/roc_curve": wandb.Image(path_roc),
			f"{stage}/predictions": wandb.Table(dataframe=pd.read_csv(predictions_path)),
		})



def show_batch_images(train_dataloader, num_images=10, save_dir=None):
	
	"""
	Show a batch of images from the dataloader.

	Args:
		- train_dataloader: DataLoader, training dataloader.
		- num_images: int, number of images to show.
		- save_dir: str, directory to save the images."""
	
	images_shown = 0

	plt.figure(figsize=(15, 6))

	for images, labels in train_dataloader:
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
