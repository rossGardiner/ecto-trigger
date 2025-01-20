# Usage

## Overview

Ecto-Trigger comprises a basic [collection of files](https://ross-jg.github.io/ecto-trigger/html/annotated.html). Where each file defines a class and command-line interface for independent tasks. Below, usage examples are shown for each file. 

For more information on how to organise a dataset for training/evaluation, see the [FAQs](about.md). 

## [model_loader.py](../model_loader.py)

This file contains a class, `ModelLoader`, with static methods `create_model()`, `load_keras_model()` and `load_tflite_model()`. 

These are intended to be accessed from other files, as loading a model on its own is not much use. Below is shown how to load a model using `python`. 

```
from model_loader import ModelLoader

# this line will load a Keras model from a .hdf5 checkpoint file
k_model = ModelLoader.load_keras_model("path/to/weights.hdf5")

# this line will load a compressed tflite model from a .tflite file
q_model = ModelLoader.load_tflite_model("path/to/weights.tflite")

# to create your own model, use the function below
k_model = ModelLoader.create_model((1080, 1080, 3), 0.5, dropout_rate=0.2, freeze_base=False)
```


## [model_trainer.py](../model_trainer.py)
This file contains a class, `ModelTrainer`, which can be instantiated in Python as shown:

```
from model_trainer import ModelTrainer

mt = ModelTrainer({"train_data_dir": "/path/to/train/data", "val_data_dir": "/path/to/val/data", "batch_size" : 16, "input_shape" : (120,160,3), "alpha": 0.35, "log_dir" : "logs", "model_type": "Mobnetv2", "epochs" : 2, "use_pretrained_weights" : False}) 
mt.train()
```
You can also call `model_trainer.py` from the command line:
```
python model_trainer.py \
    --train_data_dir "/path/to/train/data" \
    --val_data_dir "/path/to/validation/data" \
    --batch_size 16 \
    --input_shape "(120, 160, 3)" \
    --alpha 0.35 \
    --log_dir "logs" \
    --epochs 20 \
```
These commands will start a model training, you can check its progress using `Tensorboard` by passing it the log directory:
```
tensorboard --logdir=logs
```
## [model_evaluator.py](../model_evaluator.py)
This file contains a class, `ModelEvaluator`, which can be instantiated in Python as shown to load the weights for model 8:
```
from model_evaluator import ModelEvaluator
me = ModelEvaluator(16, (120, 160, 3), "model_weights/8/checkpoints/weights.10.hdf5", "/path/to/validation/data") 
me.evaluate()
```
You can also call `model_evaluator.py` from the command line:
```
python model_evaluator.py \
    --batch_size 16 \
    --weights_path "model_weights/8/checkpoints/weights.10.hdf5" \
    --test_data_dir "/path/to/validation/data"
```
## [model_quantiser.py](../model_quantiser.py)
This file contains a class, `ModelQuantiser` which can be instantiated in Python as shown to quantise a given model in keras `.hdf5` format and save it as a `.tflite` file. 
```
from model_quantiser import ModelQuantiser
mq = ModelQuantiser(
        weights_file="/path/to/weights.hdf5",
        representative_dataset="/path/to/representative_dataset,
        representative_example_nr=100
    )
    # Quantise the model
    quantiser.quantise_model(output_path="/path/to/weights.tflite")
```
You can also call `model_quantiser.py` from the command line:
```
python model_quantiser.py \
  --weights_file /path/to/weights.hdf5 \
  --representative_dataset /path/to/representative_dataset \
  --representative_example_nr 100 \
  --output /path/to/weights.tflite
```
## [saliency_map_evaluator.py](../saliency_map_evaluator.py)
This file contains a class `SaliencyMapGenerator`, which can plot saliency maps alongside output confidence and the input image. It can be accessed in python as follows: 
```
from saliency_map_evaluator import SaliencyMapGenerator
smg = SaliencyMapGenerator(weights_file="model_weights/8/checkpoints/weights.10.hdf5")
smg.generate_saliency_map(input_image_path="input.png", output_path="saliency_plot.png")
```
You can also call `saliency_map_evaluator.py` from the command line:
```
python saliency_map_evaluator.py \
  --weights_file model_weights/8/checkpoints/weights.10.hdf5 \
  --input_image input.png \
  --output saliency_plot.png
```
## [generator.py](../generator.py)
This file contains a class `CustomDataGenerator`, which provides an iterable generator for Keras models, handling the preprocessing, augmentation, and batching of data for YOLO-format object detection annotations to binary classification ones. 

You can use `CustomDataGenerator` in Python as follows:
```
from generator import CustomDataGenerator
data_gen = CustomDataGenerator(
    data_directory="/path/to/dataset",
    batch_size=16,
    input_shape=(224, 224, 3),
    stop_training_flag={"stop": False},
    shuffle=True
)
X, y = data_gen[0]
```