import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from models.densenet import *
from tools.loaders import get_loaders
from tools.dttools import convert_dataset
from tools.cam_metric import get_cam_metrics

from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

# Hiper parâmetros
BATCH_SIZE = 16
EPOCHS = 1
LR = 0.0001

def train_validade_test_model(model, dataset_path, train_loader, val_loader) -> None:

    # JSON com as métricas finais
    final_json = {}

    # Verificando se a máquina possui suporte para treinamento em GPU
    # (Suporte apenas para placas NVIDIA)
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Treinando utilizando: {device}")

    # Definindo o otimizador e a loss-functions
    optimizer = optim.Adamax(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss().to(device)

    val_metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average='weighted'),
        "recall": Recall(average='weighted'),
        "f1": (Precision(average='weighted') * Recall(average='weighted') * 2 / (Precision(average='weighted') + Recall(average='weighted'))),
        "loss": Loss(criterion)
    }

    # Definindo os trainers para treinamento e validação
    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    val_evaluator = create_supervised_evaluator(model, val_metrics, device)

    for name, metric in val_metrics.items():
        metric.attach(val_evaluator, name)

    train_bar = ProgressBar(desc="Treinando...")
    val_bar = ProgressBar(desc="Validando...")
    train_bar.attach(trainer)
    val_bar.attach(val_evaluator)

    # Função que é executada ao fim de toda epoch de treinamento
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics

        final_json[trainer.state.epoch] = metrics

        print(f"Resultados da Validação - Epoch[{trainer.state.epoch}] {final_json[trainer.state.epoch]}")

    # Definição da métrica para realizar o "checkpoint" do treinamento
    # nesse caso será utilizada a métrica F1
    def score_function(engine):
        return engine.state.metrics["f1"]
    
    # Definindo e criando (se necessário) a pasta para armazenar os dados de saída
    # da aplicação
    output_folder = f"../output/{model.__class__.__name__}"

    try:
         os.mkdir(output_folder)
    except OSError as _:
         pass
    
    # Definindo o processo de checkpoint do modelo
    model_checkpoint = ModelCheckpoint(
        output_folder,
        require_empty=False,
        n_saved=1,
        filename_prefix=f"train",
        score_function=score_function,
        score_name="f1",
        global_step_transform=global_step_from_engine(trainer),
    )
        
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    print(f"\nTreinando o modelo {model.__class__.__name__}...")

    trainer.run(train_loader, max_epochs=EPOCHS)

    print(f"\nTrain finished for model {model.__class__.__name__}")

    # Salvando as métricas em um arquivo .json
    with open(f"{output_folder}/training_results.json", "w") as f:
        json.dump(final_json, f)

    model.load_state_dict(torch.load(model_checkpoint.last_checkpoint))
    
    # Iniciando testes e coletando as métricas de qualidade dos mapas de ativação
    get_cam_metrics(model, dataset_path+"/test")

if __name__ == "__main__":

    cvt_dataset_path = convert_dataset('../datasets/PetImages')
    train_loader, val_loader = get_loaders(cvt_dataset_path, batch_size=BATCH_SIZE)

    backbone_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model = DenseNet201EncoderDecoder(backbone_model, 2)

    train_validade_test_model(model, cvt_dataset_path, train_loader, val_loader)