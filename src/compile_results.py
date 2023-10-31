import os
import json
import numpy as np

def write_to_latex_table(metrics, model_output_folder:str, model_name:str) -> None:
    
    datasets = os.listdir(model_output_folder)
    datasets.remove("latex_table.txt")

    with open(os.path.join(model_output_folder, "latex_table.txt"), "w") as f:
        f.write("\\begin{table*}[!h]\n")
        f.write("\\caption{Valores de \\textit{coherency}, \\textit{complexity}, \\texit{average drop} e ADCC referentes ao modelo " + model_name + " em relação a todos os \\textit{datasets} utilizados.}")
        f.write("\\begin{tabular}{lllll}\n")
        f.write("\\textbf{Dataset} & Coherency & Complexity & Average Drop & ADCC \\\\ \\hline\n")

        for i in range(len(datasets)):
            line = f"\t {datasets[i]} & "
            for m_i in range(len(metrics[i])):

                if m_i < len(metrics[i]) - 1:
                    line += f"{100*metrics[i][m_i]:.2f} & "
                else:
                    line += f"{100*metrics[i][m_i]:.2f} \\\\"
                
            if i == len(datasets) - 1:
                line += " \\hline"

            f.write(f"{line}\n")

        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def compile_cam_metrics(model_output_folder:str, model_name:str) -> None:

    datasets = os.listdir(model_output_folder)
    datasets.remove("latex_table.txt")

    metrics_values = np.zeros((len(datasets), 4))
    j = 0 # Indice para armazenar as médias na matriz de metricas

    for dataset in datasets:
        print(f"\nMédia das métricas referentes ao dataset: {dataset}")

        with open(os.path.join(model_output_folder, dataset, "cam_metrics.json"), "r") as f:
            data = json.load(f)
            n_images = len(list(data.keys()))

            metrics_keys = ("coherency", "complexity", "average_drop", "adcc")
            metrics_labels = ("Coherency", "Complexity", "Average Drop", "ADCC")

            avg_array = np.zeros((len(metrics_keys)), dtype=np.float32)

            for image in data.keys():
                for i in range(len(metrics_keys)):
                    avg_array[i] += data[image][metrics_keys[i]]

            avg_array /= n_images

            for i in range(len(metrics_labels)):
                print(f"{metrics_labels[i]} - {100*avg_array[i]:.2f}")

            metrics_values[j][:] = avg_array.copy()
            j += 1

    write_to_latex_table(metrics_values, model_output_folder, model_name)

if __name__ == "__main__":

    compile_cam_metrics(os.path.join("..", "output", "DenseNet201GradCam"), "DenseNet-201")
    