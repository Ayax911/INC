import warnings
import random
import numpy as np
import torch
import argparse
import os
import shutil
from training import TrainModel
from options import get_options
warnings.filterwarnings("ignore")

def main():
	"""
	Main function to train or test the model based on the given options.
	"""

	# Set the random seed for reproducibility
	set_random_seed(42)

	# Get the options from the command line arguments
	options = get_options()

	# Print the options and save the code
	print_options(options)

	# Check if the model needs to be trained or re-trained
	if options.train:
		
		#save_code(options)
		
		trainer = TrainModel(options)
		trainer.train_model()

		# Test the model
		trainer.test_model()
	
	else:
		trainer = TrainModel(options)
		trainer.test_model()

def print_options(options: argparse.Namespace) -> None:
	"""
	Prints and saves the arguments into a configuration file.

	Args:
		options (argparse.Namespace): input options.

	Returns:
		None
	"""
	# Create a message with the options
	message = '\n'.join(f'{key}: {value}' for key, value in sorted(options.__dict__.items()))

	# Print the options
	print(message)

	# Create the experiment directory if it doesn't exist
	experiment_directory = os.path.join(options.result_dir, options.exp_name)
	os.makedirs(experiment_directory, exist_ok=True)

	# Save the options into a configuration file
	configuration_file = os.path.join(experiment_directory, 'config.txt')
	with open(configuration_file, 'a') as file:
		file.write(message + '\n')

def save_code(args):
	"""
	Save code to the results folder.

	Args:
		- args: (Namespace), input options.
	"""
	# Define the path to the directory where the code will be saved
	exp_dir = os.path.join(args.result_dir, args.exp_name, 'Code')
	os.makedirs(exp_dir, exist_ok=True)

	# Recorrer el directorio de origen
	for root, dirs, files in os.walk('./'):
		# Filtrar carpetas a ignorar
		dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
		
		# Calcular la ruta relativa
		relative_path = os.path.relpath(root, './')
		dst_root = os.path.join(exp_dir, relative_path)

		# Crear el directorio de destino si no existe
		if not os.path.exists(dst_root):
			os.makedirs(dst_root)

		# Copiar archivos
		for file in files:
			# Ignorar el archivo README
			if file.lower() == 'readme':
				continue

			# Ruta completa del archivo en origen y destino
			src_file = os.path.join(root, file)
			dst_file = os.path.join(dst_root, file)

			# Copiar archivo de origen a destino
			shutil.copy2(src_file, dst_file)


def set_random_seed(seed: int) -> None:
	"""
	Set the random seed for reproducibility.

	Args:
		- seed: (int), random seed value.
	"""
	torch.cuda.empty_cache()
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
	
	main()