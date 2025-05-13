# Usage Guide: How to Use Ecto-Trigger

## Overview

Ecto-Trigger comprises a basic [collection of files](<redacted>). Where each file defines a class and command-line interface for independent tasks. In other words, it is made up of a small number of Python scripts, that each do one job. Ecto-Trigger contains files for training a model, evaluating its performance, preparing to run it on a field device or computing an example saliency map for a given model and image. 

Each section in the guide below explains how to use each of the scripts in the Ecto-Trigger toolkit. All the tools are modular, so you can use them together, or independently depending on your needs. Each script is also well documented with comments to make it easy to extend. 

## [model_loader.py](../model_loader.py)

This script provides a class, `ModelLoader`, which has static methods `create_model()`, `load_keras_model()` and `load_tflite_model()` to create, load or prepare models for use in training, evaluation or inference. While you won't typically use this file directly from the command line, its essential for internal use throughout the Ecto-Trigger workflow. 

### What it does

- Creates new models based on configurable parameters (e.g. image shape, or width multiplier)
- Loads trained Keras models from `.hdf5` files
- Loads quantised TensorFlow Lite models (`.tflite`)

### Python usage example

Below is shown how to use the `ModelLoader` class via a Python programme. 

```python
from model_loader import ModelLoader

# this line will load a Keras model from a .hdf5 checkpoint file
k_model = ModelLoader.load_keras_model("path/to/weights.hdf5")

# this line will load a compressed tflite model from a .tflite file
q_model = ModelLoader.load_tflite_model("path/to/weights.tflite")

# to create your own model, use the function below
k_model = ModelLoader.create_model((1080, 1080, 3), 0.5, dropout_rate=0.2, freeze_base=False)
```

### Inputs and Outputs

```
> Input:
  - A model file (.hdf5 or .tflite), or model creation parameters

> Output:
  - A TensorFlow Keras model object (for training/evaluation), or
  - A TensorFlow Lite interpreter object (for on-device inference)

```

## [model_trainer.py](../model_trainer.py)

This script contains a class, `ModelTrainer`, which allows you to train a binary classification model using your own dataset of labelled images. You can train directly in Python or from the command line. 

It supports custom parameterisation, so you can change the input size or model width (see [our paper]() to understand how these affected our trained models). It saves the `hdf5` model along with logs throughout the training process, which can be monitored using TensorBoard. 

### What it does

- Loads images and YOLO-format labels from your training and validation directories
- Builds and compiles a MobileNetv2-based model for binary classifiation of your data
- Trains the model for a given number of epochs
- Logs performance metrics (such as accuracy) to a directory for visualisation with TensorBoard
- Saves the trained model as a `.hdf5` file

You can load the `ModelTrainer` class from a Python programme as follows: 

```python
from model_trainer import ModelTrainer

mt = ModelTrainer({"train_data_dir": "/path/to/train/data", "val_data_dir": "/path/to/val/data", "batch_size" : 16, "input_shape" : (120,160,3), "alpha": 0.35, "log_dir" : "logs", "model_type": "Mobnetv2", "epochs" : 2, "use_pretrained_weights" : False})

mt.train()
```


Alternatively, you can also call `model_trainer.py` directly from the command line:

```bash
python model_trainer.py \
    --train_data_dir "/path/to/train/data" \
    --val_data_dir "/path/to/validation/data" \
    --batch_size 16 \
    --input_shape "(120, 160, 3)" \
    --alpha 0.35 \
    --log_dir "logs" \
    --epochs 20 \
```

Where:

- `train_data_dir`` is the path to your directory of training images and YOLO-format labels
- `val_data_dir` is the path to your directory of validation images and YOLO-format labels
- `batch_size` is the number of images per training step (e.g. 16)
- `input_shape` is the dimensions for the input size, specified as a string tuple, e.g. (120, 160, 3), height, width, number of channels.   
- `alpha` controls the model size through changing the model width - smaller values of alpha produces lighter models, but this has an accuracy penalty, see our paper for further details.
- `log_dir` is the directory where training logs will be saved for analysis with TensorBoard.
- `epochs` is the number of full passes though the training data to run before exiting training. 

### Inputs and Outputs

```

> Input:
  - Folder of labelled training images (YOLO format)
  - Folder of labelled validation images (YOLO format)

> Output:
  - Trained `.hdf5` model file
  - Training logs viewable in TensorBoard

```

### YOLO Dataset Format

To organise your dataset in YOLO-format for training, follow this structure:

```
your_data_train/
|-- img001.jpg
|--  img001.txt
|--  img002.jpg
|--  img002.txt
|--  ...

your_data_val/
|--  img001.jpg
|--  img001.txt
|--  img002.jpg
|--  img002.txt
|--  ...
```
Each `.txt` file should contain the YOLO-style annotation for its corresponding image (for more information about the YOLO format, see [here](https://roboflow.com/formats/yolo-darknet-txt)):

For images with the object of interest: each line in the .txt file should contain a bounding box in this format:

```
0 x_center y_center width height
```

For images without the object: the .txt file will be empty (zero length).

### Monitoring Training with TensorBoard

You can check training progress using `Tensorboard` by passing it the log directory:

```bash
tensorboard --logdir=logs
```

This allows you to monitor accuracy values throughout training, so you can see how its going and when you might want to stop model training. 

## [model_evaluator.py](../model_evaluator.py)

This script contain a class, `ModelEvaluator`, which helps you evaluate how well your trained model performas on a labelled test dataset. It calculates and prints out the model's accuracy and loss, giving you a quick sense of how confidently and correctly your model is making predictions.

### What it does

- Loads a trained Keras model (given in `.hdf5`) format.
- Loads a test dataset from a directory in YOLO-format.
- Evaluates the models performance on the dataset.
- Prints out the accuracy and loss values to the terminal window.

### Python usage example

You can call the `ModelEvaluator` directly inside your Python programmes:

```python
from model_evaluator import ModelEvaluator

me = ModelEvaluator(16, (120, 160, 3), "model_weights/8/checkpoints/weights.10.hdf5", "/path/to/validation/data") 

me.evaluate()
```

You can also call `model_evaluator.py` from the command line:

```bash
python model_evaluator.py \
    --batch_size 16 \
    --weights_path "model_weights/8/checkpoints/weights.10.hdf5" \
    --test_data_dir "/path/to/validation/data"
```

Where: 

- `batch_size` is the number of images to process per step (e.g. 16)
- `weights_path` is the path to the trained Keras model to evaluate (`.hdf5` format)
- `test_data_dir` is the folder of labelled images in YOLO format for evaluation

### Inputs and Outputs

```
> Input: Trained `.hdf5` model, test image folder 
> Output: Accuracy score, loss printed to terminal window
```


## [model_quantiser.py](../model_quantiser.py)

This script defines a class, `ModelQuantiser` which can be instantiated in Python as shown to quantise a given model in keras `.hdf5` format and save it as a `.tflite` file. Quantisation helps you convert an Ecto-Trigger Keras model into a smaller, more computationally efficient representation for use on low-powered devices such as Raspberry Pi or ESP32-S3. This allows real-time, on-device object detection in the field. 

### What it does

- Loads a trained Keras model (supplied in `.hdf5` format)
- Calibrates the quantisation process using a representative dataset (a folder containing a small number (around 100) of example images)
- Converts the `.hdf5` model into TensorFlow Lite (`.tflite`) format.
- Saves the now quantised model, ready for deployment.  

### Python usage example 

You can call the `ModelQuantiser` directly inside your Python programmes:

```python
from model_quantiser import ModelQuantiser

mq = ModelQuantiser(
        weights_file="/path/to/weights.hdf5",
        representative_dataset="/path/to/representative_dataset,
        representative_example_nr=100
    )

# Quantise the model and save it
mq.quantise_model(output_path="/path/to/weights.tflite")
```

You can also call `model_quantiser.py` from the command line:

```bash
python model_quantiser.py \
  --weights_file /path/to/weights.hdf5 \
  --representative_dataset /path/to/representative_dataset \
  --representative_example_nr 100 \
  --output /path/to/weights.tflite
```

Where:

- `weights_file` is the path to the trained Keras model (`.hdf5`) file, to be quantised.
- `representative_dataset` is the path to the folder of sample images to use for calibrating the quantisation process
- `representative_example_nr` is the number of images to use from that folder, e.g. 100.
- `output` is the filepath for saving the resultant `.tflite` model. 

### Inputs and Outputs

```
> Input:
  - A trained `.hdf5` model
  - A validation dataset (images and matching YOLO-style `.txt` files)

> Output:
  - Accuracy and loss values printed to the terminal
```

## [saliency_map_evaluator.py](../saliency_map_evaluator.py)

This script provides a visual way to understand what the model is "looking at" when it makes a prediction. It generates saliency maps, these are heatmaps that highlight the parts of a given input image which most influenced a model's decision. This is a helpful tool for interpretation, debugging and visualisations. 

### What it does

- Loads a given Keras model (given to the script in `.hdf5` format)
- Processes a given input image (given to the script in `.jpg` or `.png` format)
- Runs the model and computes a saliency map showing where the model focused.
- Creates a composite image which includes: the original image, the saliency heatmap, the confidence score at the output of the model. 

### Python usage example

If you want to use the script in your own programmes, you can call it programmatically as follows:

```python
from saliency_map_evaluator import SaliencyMapGenerator

smg = SaliencyMapGenerator(weights_file="model_weights/8/checkpoints/weights.10.hdf5")

smg.generate_saliency_map(input_image_path="input.png", output_path="saliency_plot.png")
```


You can also call `saliency_map_evaluator.py` from the command line:

```bash
python saliency_map_evaluator.py \
  --weights_file model_weights/8/checkpoints/weights.10.hdf5 
  --input_image input.png \
  --output saliency_plot.png
```

Where:
- `weights_file` is the path to your trained model file in `.hdf5` format
- `input_image` is the path to the image you wish to analyse
-  `output` defines the filename and path you want to save the final composite plot to. 

### Inputs and Outputs

```
> Input:
  - A trained model file (.hdf5)
  - An image file (e.g., .jpg or .png)

> Output:
  - A .png image showing:
      - The input image
      - A heatmap of attention (saliency map)
      - The prediction confidence score
```


## [generator.py](../generator.py)

This file defines the class `CustomDataGenerator`, which is a data loading utility built for training Ecto-Trigger models. It works with image datasets which follow the YOLO annoation format (as described above), and prepared them for binary classification tasks (e.g. "insect" or "no insect"). 

### What it does

`CustomDataGenerator` creates batches of images and labels which can be used by other scripts for both training and evaluation. It: 

- Loads images from a folder, which is specified as an argument.
- Matches each image with its corresponding `.txt` annotation file in YOLO-style.
- Converts all annotations for a given image into binary labels (1 = object present, 0 = no object).
- Resizes and formats the images for input into the model.
- Optionally shuffles and augments the images (see our paper for details on augmentations).
- Supplies batches of images and labels to a model, via a Keras-compatible interface. 

This is useful when training models using `model_trainer.py`. 

### Python usage example 

If you want to integrate `CustomDataGenerator` into your own programmes, you can use it inside Python as follows:
```python
from generator import CustomDataGenerator
data_gen = CustomDataGenerator(
    data_directory="/path/to/dataset",    #Folder with .jpg and .txt files in YOLO-format, this is your dataset
    batch_size=16,                        #Number of images per batch
    input_shape=(224, 224, 3),            #Input shape to resize all images to
    stop_training_flag={"stop": False},   #Flag for manually stopping training (optional) 
    shuffle=True                          #Shuffle the dataset after each epoch
)
#To get an example batch image images and labels
X, y = data_gen[0] #Returns X, a batch of images, and y, a batch of 0s and 1s (i.e. the labels)
```

### Inputs and Outputs

```
> Input:
  - Folder of .jpg images
  - Corresponding .txt files (YOLO format):
      - If object is present: "0 x_center y_center width height"
      - If object is absent: (empty .txt file)

> Output:
  - X: NumPy array of shape (batch_size, height, width, channels)
  - y: NumPy array of binary labels (1 if object present, 0 if not)

```


## Suggested Workflow

To use each of these files together to create, train, evaluate and deploy a model, you would follow the order of the workflow below:
1. Prepare your dataset in class-based folders
2. Train a model (`model_trainer.py`)
3. Evaluate it (`model_evaluator.py`)
4. Quantise for deployment (`model_quantiser.py`)
5. Deploy to field hardware ([Deployment Guide](deployment.md))
6. Optionally visualize attention maps (`saliency_map_evaluator.py`)


