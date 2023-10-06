import os
import random
import shutil
import numpy as np

# Função responsável por converter um determinado dataset para o formato suportado pela aplicação
def convert_dataset(dataset_dir:str) -> str:
    # Criando estrutura para o novo dataset
    classes = os.listdir(dataset_dir)
    classes.sort()

    n_classes = len(classes)
    print(f"Conversão de Dataset - O dataset a ser convertido possui {n_classes} classes.")

    dataset_path = dataset_dir + f"_{n_classes}_CV"
    dataset_train_path = dataset_path + "/train/"
    dataset_test_path = dataset_path + "/test/"
    dataset_val_path = dataset_path + "/val/"

    try:
        # Criando os diretórios necessários
        print("Conversão de Dataset - Criando diretórios...")
        os.mkdir(dataset_path)
        os.mkdir(dataset_train_path)
        os.mkdir(dataset_test_path)
        os.mkdir(dataset_val_path)

        total_imgs = 0

        for c in classes:
            os.mkdir(f"{dataset_train_path}/{c}/")
            os.mkdir(f"{dataset_val_path}/{c}/")
            total_imgs += len(os.listdir(f"{dataset_dir}/{c}"))

        # Esse split indica que:
        # 80% do dataset será dedicado para treino do modelo;
        # 15% do dataset será dedicado a validação do modelo;
        # 15% do dataset será dedicado ao testes do modelo pós treinamento;
        val_split = test_split = round(total_imgs * 0.30)/2

        val_imgs_counter = 0
        test_imgs_counter = 0

        print('Conversão de Dataset - Iniciando conversão... (Esse processo pode demorar um pouco)')
        for _ in range(total_imgs):
            c = random.choice(classes)
            imgs = os.listdir(f"{dataset_dir}/{c}")
            img = np.random.choice(imgs)

            # Nessa etapa, um valor booleano aleatório é gerado de forma a decidir se a imagem sob iteração
            # será utilizada para a validação, teste ou treinamento do modelo;

            # Checagem para enviar a imagem ao conjunto de validação
            if bool(random.getrandbits(1)) and val_imgs_counter < val_split:
                shutil.copyfile(f"{dataset_dir}/{c}/{img}", f"{dataset_val_path}/{c}/{img}")
                val_imgs_counter += 1
                continue
            
            # Checagem para enviar a imagem ao conjunto de teste
            elif bool(random.getrandbits(1)) and test_imgs_counter < test_split:
                shutil.copyfile(f"{dataset_dir}/{c}/{img}", f"{dataset_test_path}/{classes.index(c)}_{test_imgs_counter}.png")
                test_imgs_counter += 1
                continue
            
            # Caso a imagem não seja enviada para nenhum dos conjuntos anteriores, envie ela para
            # o conjunto de treino
            shutil.copyfile(f"{dataset_dir}/{c}/{img}", f"{dataset_train_path}/{c}/{img}")

    except OSError as _:
        print("Conversão de Dataset - O dataset já foi convertido!")

    return dataset_path