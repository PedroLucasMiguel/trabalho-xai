import os
import argparse
from torch import manual_seed
from models.densenet import *
from models.resnet import *
from models.efficientnet import *
from torchvision.models import densenet201, resnet50, efficientnet_b0
from train import train

# Hiper-parâmetros de treinamento
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001

if __name__ == "__main__":

    # Garantindo reprodutibilidade
    manual_seed(0)

    parser = argparse.ArgumentParser(prog='Aprendizado profundo - Qualidade de explicações',
                                     description='Esse programa realiza a coleta de métricas quantitativas referetes a qualidade de explicações geradas por diversos modelos')
    
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, required=True)

    args = parser.parse_args()

    datasets_path = os.path.join("..", "datasets")
    datasets = ("CR", "LA", "LG", "NHL", "UCSB")

    model_name = args.model.upper()
    gpu = args.gpu

    for dataset in datasets:

        n_classes = len(os.listdir(os.path.join(datasets_path, dataset)))

        match model_name:
            case "DENSENET201":
                model = DenseNet201GradCam(densenet201(weights='IMAGENET1K_V1'), n_classes)
            case "DENSENET201AB":
                model = DenseNet201EncoderDecoder(densenet201(weights='IMAGENET1K_V1'), n_classes)
            case "RESNET50":
                model = Resnet50GradCam(resnet50(weights='IMAGENET1K_V1'), n_classes)
            case "EFFICIENTNETB0":
                model = EfficientNetB0GradCam(efficientnet_b0(), n_classes)

        train(model = model, 
              dataset_path = datasets_path,
              dataset_name = dataset, 
              batch_size = BATCH_SIZE, 
              epochs = EPOCHS, 
              lr = LR, 
              gpu = gpu)
        
        torch.cuda.empty_cache()


    
