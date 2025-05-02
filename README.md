# Ecto-Trigger
Ecto-Trigger helps you train and deploy lightweight deep learning models for detecting objects (for example, insects) in camera trap images or video streams. 

This code-base is a tool-kit designed to help support ecologists and other end users, especially those working on designing camera trap solutions for scenarios where traditional PIR sensors do not work well as triggers due to missed detections, high trigger latency or excessive false positives. Our example model weights have been developed to detect insects. Lightweight binary classifiers can be deployed locally on microcontrollers and continuously run to filter insect images in real-time. 

All documentation is served via our [web-page](www.google.com), we have also made this accessible as a [pdf-format vignette](www.google.com). This all supports our paper: [Towards scalable insect monitoring: Ultra-lightweight CNNs as on-device triggers for insect camera traps](www.google.com), which contains more advanced technical details and background for our approach. 

## What is it? 

Ecto-Trigger trains **binary classifiers** (e.g. yes/no models) that declare whether an input image contains an object of interest. These models are:
1. **Lightweight** - meaning they can run on low-cost computers which can be deployed in the field, e.g. microcontrollers or Raspberry Pi
2. **Deep Learning-based** - use a convolutional neural network model, MobileNetv2 which is trained to detect the object of interest
3. **Deployable** - an output `.tflite` file can be produced executing models on field devices. 

## Code overview: What's included? 

| File | Purpose |
|------|---------|
| `model_loader.py` | Build and load deep learning models (for training use) |
| `model_trainer.py` | Train a model using your labeled image dataset |
| `model_evaluator.py` | Evaluate model performance on test data |
| `model_quantiser.py` | Convert trained models to `.tflite` for edge devices |
| `saliency_map_evaluator.py` | Visualize what the model focuses on |
| `tflite_model_runner.py` | Run `.tflite` models on Raspberry Pi (no TensorFlow needed) |

To find out more about how to use every file, check the [usage guidance](guides/usage.md), which includes a description of all the ins and outs. 


### 2. Prepare Your Dataset

Organize your dataset in YOLO-format for training like this:

```
your_data_train/
├── img001.jpg
├── img001.txt
├── img002.jpg
├── img002.txt
├──...
your_data_val/
├── img001.jpg
├── img001.txt
├── img002.jpg
├── img002.txt
├──...
```
Each .txt file should contain the YOLO-style annotation for its corresponding image (for more information see [here](https://roboflow.com/formats/yolo-darknet-txt)):

For images with insects: each line in the .txt file should contain a bounding box in this format:
```
0 x_center y_center width height
```

For images without insects: the .txt file will be empty (zero length).

### 3. Train a Model

```bash
python model_trainer.py     --train_data_dir "/path/to/train"     --val_data_dir "/path/to/val"     --batch_size 16     --input_shape "(120, 160, 3)"     --alpha 0.35     --epochs 20     --log_dir "logs"
```

### 4. Quantize the Model

```bash
python model_quantiser.py   --weights_file model_weights/your_model.hdf5   --representative_dataset /path/to/sample_data   --representative_example_nr 100   --output model_weights/your_model.tflite
```

### 5. Deploy It

See the [Deployment Guide](guides/deployment.md) to run your model on:

- Raspberry Pi
- ESP32-S3 Microcontroller

---

## More Guides

- [Usage Guide](guides/usage.md)
- [Deployment Guide](guides/deployment.md)
- [FAQs](guides/faqs.md)
- 

---



# Contributing

Ecto-Trigger is open-source, we encourage others to contribute. Please do so by making a [pull request](https://github.com/ross-jg/ecto-trigger/pulls). 

# License 

Distributed under the GPL-3.0, see [LICENSE](LICENSE) for more information.

# Citation
```
tbc
```



