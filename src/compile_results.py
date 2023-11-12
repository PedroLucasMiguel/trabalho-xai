import os
import json
import numpy as np

# Função responsável por compilar os resultados finais obtidos por um modelo
def compile_cam_metrics(model_output_folder:str, model_name:str) -> None:

    datasets = os.listdir(model_output_folder)

    metrics_keys = ("coherency", "complexity", "average_drop", "adcc")
    metrics_labels = ("Coherency", "Complexity", "Average Drop", "ADCC")

    try:
        datasets.index("latex_table.txt")
        datasets.remove("latex_table.txt")
    except ValueError as _:
        pass

    metrics_values = np.zeros((len(datasets)+1, 4))
    j = 0 # Indice para armazenar as médias na matriz de metricas

    print(f"Resultados referentes ao modelo: {model_name}")
    for dataset in datasets:
        print(f"\nMédia das métricas referentes ao dataset: {dataset}")

        with open(os.path.join(model_output_folder, dataset, "cam_metrics.json"), "r") as f:
            data = json.load(f)
            n_images = len(list(data.keys()))

            avg_array = np.zeros((len(metrics_keys)), dtype=np.float32)

            for image in data.keys():
                for i in range(len(metrics_keys)):
                    avg_array[i] += data[image][metrics_keys[i]]

            avg_array /= n_images

            for i in range(len(metrics_labels)):
                print(f"{metrics_labels[i]} - {100*avg_array[i]:.2f}")

            metrics_values[j][:] = avg_array.copy()
            j += 1

    metrics_values[len(datasets)][:] = metrics_values[0:len(datasets)][:].sum(axis=0)
    metrics_values[len(datasets)][:] /= len(datasets)

if __name__ == "__main__":

    # Para compilar os resultados referentes a um modelo específico, retire o '#' do modelo desejado

    compile_cam_metrics(os.path.join("..", "output", "ABN"), "Attention Branch Network (ABN)")
    #compile_cam_metrics(os.path.join("..", "output", "DenseNet201GradCam"), "DenseNet-201")
    #compile_cam_metrics(os.path.join("..", "output", "EfficientNetB0GradCam"), "EfficientNet-b0")
    #compile_cam_metrics(os.path.join("..", "output", "Resnet50GradCam"), "ResNet-50")
    #compile_cam_metrics(os.path.join("..", "output", "XCNN"), "Explainable Convolutional Neural Network (XCNN)")
    