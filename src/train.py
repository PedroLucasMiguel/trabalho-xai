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
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator

def train(model:nn.Module, 
          dataset_path:str,
          dataset_name:str,
          batch_size:int = 16,
          epochs:int = 10,
          lr:float = 0.0001,
          gpu:int = 0) -> None:

      # JSON com as métricas finais
      final_json = {}

      cvt_dataset_path = convert_dataset(dataset_path, dataset_name)
      train_loader, val_loader = get_loaders(cvt_dataset_path, batch_size=batch_size)

      # Verificando se a máquina possui suporte para treinamento em GPU
      # (Suporte apenas para placas NVIDIA)
      device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
      model = model.to(device)

      print(f"Treinando utilizando: {device}")

      # Definindo o otimizador e a loss-functions
      optimizer = optim.Adam(model.parameters(), lr=lr)
      criterion = nn.CrossEntropyLoss().to(device)

      val_metrics = {
            "accuracy": Accuracy(),
            "precision": Precision(average='weighted'),
            "recall": Recall(average='weighted'),
            "f1": (Precision(average='weighted') * Recall(average='weighted') * 2 / (Precision(average='weighted') + Recall(average='weighted'))),
            "loss": Loss(criterion)
            }

      # Definindo os trainers para treinamento e validação

      # Checando se estamos treinando o modelo ABN para deifnir o traid/validation steps corretos
      if model.__class__.__name__ == "ABN":
            def train_step(engine, batch):
                  model.train()
                  optimizer.zero_grad()
                  x, y = batch[0].to(device), batch[1].to(device)
                  att_outputs, outputs, _ = model(x)

                  att_loss = criterion(att_outputs, y)
                  per_loss = criterion(outputs, y)
                  loss = att_loss + per_loss
                        
                  loss.backward()
                  optimizer.step()

                  return loss.item()
                        
            def validation_step(engine, batch):
                  model.eval()
                  with torch.no_grad():
                        x, y = batch[0].to(device), batch[1].to(device)
                        _, y_pred, _ = model(x)
                        return y_pred, y
                  
            trainer = Engine(train_step)
            val_evaluator = Engine(validation_step)

      else:
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
      output_folder = os.path.join("..", "output", model.__class__.__name__, dataset_name)

      # Criando diretórios para armazenar as saídas
      try:
            os.mkdir(os.path.join("..", "output"))
      except OSError as _:
            pass

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

      trainer.run(train_loader, max_epochs=epochs)

      print(f"\nTreinamento finalizado para o modelo {model.__class__.__name__}")

      # Salvando as métricas em um arquivo .json
      with open(os.path.join(output_folder, "training_results.json"), "w") as f:
            json.dump(final_json, f)

      model.load_state_dict(torch.load(model_checkpoint.last_checkpoint))
      
      # Iniciando testes e coletando as métricas de qualidade dos mapas de ativação
      get_cam_metrics(model, os.path.join(cvt_dataset_path, "test"), dataset_name)

      os.system('cls' if os.name == 'nt' else 'clear')
      print("Treinamento e teste finalizado!")
      print(f"Todas as métricas obtidas para este modelo estão presentes em: ../output/{model.__class__.__name__}")
      input("Pressione ENTER para finalizar...")