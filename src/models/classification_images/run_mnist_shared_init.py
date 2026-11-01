"""Corre el MNIST smoketest con el backbone y la cabeza inicializados desde los checkpoints
compartidos con FedMammoBench (ver FedMammoBench/scripts/generate_shared_init_mnist.py para
el porqué y cómo se generan/remapean esos pesos).

Comparación de paridad puntual FMB<->INC -- no un modo de uso general de este proyecto. El
backbone ya se resuelve solo con las flags existentes de options.py (--path_image_model
--pretrained False --from_scratch False, ver get_image_model() en models/image_models.py);
solo la cabeza necesita este script, porque options.py no tiene una flag equivalente para
la cabeza (--path_image_model solo cubre el backbone) -- se inyecta acá manualmente sobre
`trainer.model[1]` (el classifier, `self.model = nn.Sequential(image_model, classifier)`
en get_model.py) justo después de construir TrainModel y antes de entrenar. load_state_dict()
copia los tensores IN-PLACE sobre los Parameter ya existentes, así que el optimizer (creado
dentro de TrainModel.__init__ sobre self.model.parameters()) sigue apuntando a los mismos
objetos y no hace falta reconstruirlo.

Correr (mismas flags que run_mnist_shared_init.sh, ver ese archivo):
    python run_mnist_shared_init.py <mismos argumentos que main.py, más --path_image_model>
"""

import torch

from main import print_options, set_random_seed
from options import get_options
from training import TrainModel

HEAD_WEIGHTS_PATH = (
    "/home/akira/snap/steam/federallearning/FedMammoBench/runs/shared_init_mnist/head_inc.pt"
)


def main() -> None:
    set_random_seed(42)
    options = get_options()
    print_options(options)

    trainer = TrainModel(options)
    trainer.model[1].load_state_dict(torch.load(HEAD_WEIGHTS_PATH, map_location=trainer.device))

    trainer.train_model()
    trainer.test_model()
    trainer.validate_model_full()


if __name__ == "__main__":
    main()
