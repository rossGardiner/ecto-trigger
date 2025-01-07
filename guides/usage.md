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

mt = ModelTrainer({"train_data_dir": "/hdd/bulk/projects/ecostack_to_vww/train1201", "val_data_dir": "/hdd/bulk/projects/ecostack_to_vww/train1201", "batch_size" : 16, "input_shape" : (96,96,1), "alpha": 0.1, "log_dir" : "logs", "model_type": "Mobnetv2", "epochs" : 2, "use_pretrained_weights" : False}) 

```

## [model_evaluator.py](../model_evaluator.py)
```
>>> from model_evaluator import ModelEvaluator
>>> me = ModelEvaluator(16, (120, 160, 3), "model_weights/8/checkpoints/weights.10.hdf5", "/hdd/bulk/projects/ecostack_to_vww/val1201") 
```
## [model_quantiser.py](../model_quantiser.py)

## [saliency_map_evaluator.py](../saliency_map_evaluator.py)