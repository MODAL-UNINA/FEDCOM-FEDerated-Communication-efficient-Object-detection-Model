# FEDCOM: FEDerated Communication-efficient Object detection Model

##  Abstract
The emergence of 6G edge intelligence demands highly efficient, adaptive Machine Learning (ML) solutions that operate under severe resource constraints. In this paper, we introduce FEDCOM, a scalable framework designed to support Continual Learning (CL) in federated edge environments through selective communication and adaptive model updates. FEDCOM uses Federated Continual Learning (FCL), which combines Federated Learning with CL, and low-power optimization strategies, to minimize energy usage and bandwidth consumption during communication between server and clients. 

The core idea is to reduce the environmental and computational costs of distributed training by transmitting only unfrozen (trainable) layers during communication and scheduling client participation based on a data novelty metric. This approach limits unnecessary computation and avoids redundant communication, aligning with the goals of sustainable and low-latency 6G edge intelligence. 

To demonstrate its effectiveness, we tested FEDCOM in an application scenario involving object detection (OD) in precision agriculture, using the YOLOv12s model. The system leverages diverse edge datasets across multiple domains and evaluates energy consumption and OD performance in realistic scenarios. Experimental results demonstrate that FEDCOM reduces computational costs and carbon footprint compared to full retraining methods and FCL without our communication strategy, while maintaining practical detection performance. Hence, FEDCOM shows how FCL, along with scheduling and communication-efficient strategies, can be powerful in 6G TinyML real-world applications.

![Framework](framework_backgroud.png)

## Acknowledgments
PNRR project FAIR - Future AI Research (PE00000013), Spoke 3, under the NRRP MUR program funded by the NextGenerationEU.

## Data Availability
Datasets employed in this work are from various free available sources.  We selected the following 11 image datasets representing various fruits and assigned each type of fruit to only one client, to simulate a non-IID scenario.

### Original Datasets
Below is the list of datasets used in the experiments:

- **[Cherry tomato images in the greenhouse](#)**  : five types of cherry tomato labeled images in the greenhouse.
    
- **[Tomato Plantfactory Dataset](#)** : comprises 520 images of the Micro tomato variety, captured at two different resolutions and at two stages of fruit growth. The images were acquired under artificial light and present various complexities, including variations in perspectives, lighting quality, distance, and occlusions and blurs of the fruits, resulting in a total of 9112 instances, including 5996 green fruits and 3116 red fruits.
    
- **[LaboroTomato Dataset](#)** : images of tomatoes in various stages of ripening, acquired at a local farm, utilizing two cameras, each contributing to varying resolutions and image quality. The dataset consists of 804 images with 10610 labeled objects.

- **[tomatOD: Tomato Fruit Localization and Ripening Classification](#)** : consists of 277 images with 2418 annotated tomatoes.
    
- **[Embrapa Wine Grape Instance Segmentation Dataset (Embrapa WGISD)](#)** : 300 RGB images showing 4432 grape clusters of five different grape varieties captured in the field, exhibiting variations in grape pose, illumination, focus, and genetic and phenological characteristics such as shape, color, and compactness. The images were captured at the Guaspari Winery in São Paulo, Brazil, using a Canon EOS REBEL T3i DSLR camera and a Motorola Z2 Play smartphone.

- **[A dataset of grape multimodal OD and semantic segmentation](#)** : 3954 labeled image samples are extracted from high-quality multimodal video stream data of green and purple grapes, including six varieties, under different illumination and obscuration conditions.
    
- **[Grapes dataset Computer Vision Project](#)** : 425 images of grapes (ripe, unripe, rotten, spotted, and picking point).
    
- **[Strawberry Dataset for Object Detection](#)** : 813 images with 4568 labeled objects belonging to 3 different classes (ripe, peduncle, and unripe).
    
- **[StrawDI: The Strawberry Digital Images Data Set (StrawDI\_Db1)](#)** : 8000 images of strawberries, taken from 20 plantations, within an approximate area of 150 hectares, in the province of Huelva, Spain.
    
- **[Strawberry-DS](#)** : 247 RGB digital images of strawberry fruits taken at the Central Laboratory for Agricultural Climate (CLAC), Agricultural Research Center, Cairo, Egypt. The images were captured from the fruit top view, considering different view angles using a Sony Xperia Z2 LTE-A D6503 smartphone 20.7 MP camera. The images contain both fully and partially visible strawberry fruits.
    
- **[Strawberry.00 Computer Vision Project](#)**  : 450 strawberry images.

### Final Datasets
The original datasets were preprocessed to generate the final datasets used in the experiments. The preprocessing steps included:
- Converting the dataset format to YOLO format;
- Splitting the datasets into training, validation and test sets;
- Applying data augmentation techniques to enhance the training data;

plus additional manual operations. These final data are available at the following url: (will be available soon).

## Installation
### Requirements
The provided code allows you to run the FEDCOM framework (and other comparison frameworks) in a local machine. For the execution of the experiments and an accurate measurement of emissions, you need to have at least 4 NVIDIA GPUs available with `CUDA` at least 12.1.

### Environment Setup
To run the FEDCOM framework, you need to set up a Python environment with the required dependencies. You can use the provided `environment.yaml` file to create a conda environment in the `install` folder.

```bash
conda env create -f install/environment.yaml -n <environment_name>
```

with `<environment_name>` being the name you want to assign to the environment (e.g., `fedcom_env`).

After creating the environment, activate it with:

```bash
conda activate <environment_name>
```

## Execution
### Preprocessing
After downloading the original datasets in the `data/original_datasets` folder, you need to preprocess them to generate the required files for training and evaluation. This can be done by running the following command from the `code` folder:

```bash
$CONDA_PREFIX/bin/python 0_preprocess.py
```

. This script will process the original datasets and save the preprocessed data in the `data/preprocessed_datasets` folder. Additional manual operations were done to generate the final data. The final datasets are then saved in the `data/datasets` folder.

In case you want to use the final datasets directly, you can skip the preprocessing step and download the final datasets in the `data/datasets` folder. Due to the datasets size, please contact the corresponding author at the following email address to request access: francesco.piccialli@unina.it.

### Model generation
After setting up the environment, you need to build the model configuration adapted to use only 3 classes with the following command from the `code`
 folder:

```bash
$CONDA_PREFIX/bin/python generate_model.py --model-config yolo12s.yaml --output-model-config yolo12s_upd.yaml --gpu-id=0
```

this will generate a new model configuration file `yolo12s_upd.yaml` in the `Base_Model` folder.

### Running one scenario
After generating the model configuration file, you can run one complete scenario depending on the framework by executing the following command from the `code` folder:

```bash
bash run_scenario.sh <framework> <scenario_id> <domain_ids> <server_gpu_id> <clients_gpu_ids> <num_rounds> <number_of_epochs> 0.0.0.0:<port> <similarity_threshold> <max_similarity_images> <model_name>
```

with the following parameters:
- `<framework>`: The framework you want to run (e.g., `FEDCOM`, `Baseline`, `FCL`). Defaults to `FEDCOM`.
- `<scenario_id>`: The ID of the scenario you want to run (integer from `1` to `3`). Defaults to `1`.
- `<domain_ids>`: Comma-separated list of domain IDs to be run in the scenario (e.g., `1,2,3`). Defaults to `1,2,3`.
- `<server_gpu_id>`: The GPU ID for the server. Defaults to `0`.
- `<clients_gpu_ids>`: Comma-separated list of GPU IDs for the clients (e.g., `0,1,2`). Defaults to `0,1,2`.
- `<num_rounds>`: The number of communication rounds for the training (plus 1 for the final results evaluation). Defaults to `51`.
- `<number_of_epochs>`: The number of epochs for each training round. Defaults to `5`.
- `<port>`: The port number to be used by the server for communication. Defaults to `8080`.
- `<similarity_threshold>`: The threshold for the similarity metric to determine if a client should participate in the training. Defaults to `0.75`.
- `<max_similarity_images>`: The maximum number of images to be used for the similarity metric computation. Defaults to `20`.
- `<model_name>`: The name of the model to be used (e.g., `yolo12s`). Defaults to `yolo12s_upd.yaml`, which is generated starting from `yolo12s.yaml` to have 3 classes.

### Example usage
To run the FEDCOM framework for scenario 1 with domains 1, 2, and 3, using the server GPU ID 0 and client GPU IDs 0, 1, and 2, with 51 communication rounds and 5 epochs per round, you can use the following command:

```bash
bash run_scenario.sh FEDCOM 1 0,1,2 0 1,2,3 51 5
```
