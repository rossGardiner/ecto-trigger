# Introduction

## What? 
Ecto-Trigger is software which aims to help others develop very lightweight "tinyML" AI models for deployment onto microcontrollers or other edge computing devices. In this case models perform for binary classification tasks, specifically for detecting the presence or absence of objects in images. This libarary uses MobileNetv2, which is particularly suited for scenarios where computational efficiency is critical, such as real-time image filtering on resource-constrained devices. 

## Who is this for?

Firmware developers and AI Engineers working in Ecology:
    - We provide (more technical description. )

Ecology-focused individuals working on technologies:
    - hopefully this makes things easier because we've provided documentation

Others:
    - Ecto-trigger can be adapted for many tasks


Overall, Ecto-Trigger is designed for researchers, developers, and hobbyists working across the fields of computer vision, embedded systems, and environmental monitoring. It is particularly useful for those looking to deploy AI models on microcontrollers for tasks like wildlife monitoring, insect detection, or other lightweight image classification applications.

## Why use Ecto-Trigger?

Ecto-Trigger provides a complete pipeline for training, evaluating, quantising, and deploying lightweight binary classification models. It simplifies the process of creating models that can run efficiently on devices with limited computational resources. Key features include:

- **Model Training**: Train MobileNetV2-based binary classifiers with custom datasets.
- **Evaluation**: Evaluate model performance and visualize results using saliency maps.
- **Quantisation**: Convert models to TFLite format with INT8 precision for deployment on microcontrollers.
- **Deployment Guidance**: Detailed instructions for deploying models on platforms like Raspberry Pi and ESP32-S3.
- **Pre-trained Models**: Access to pre-trained models for insect detection, as described in our paper.

## Features Overview

Ecto-Trigger is composed of several modular files, each serving a specific purpose in the pipeline:

1. **`model_loader.py`**: Provides utilities for creating, loading, and managing Keras and TFLite models.
2. **`model_trainer.py`**: Handles model training with custom data generators and callbacks.
3. **`model_evaluator.py`**: Evaluates trained models on test datasets and computes metrics like accuracy.
4. **`model_quantiser.py`**: Quantizes models to TFLite format for efficient deployment.
5. **`saliency_map_evaluator.py`**: Generates saliency maps to visualize model predictions and highlight important regions in input images.
6. **`generator.py`**: Implements a custom data generator for preprocessing, augmentation, and batching of training data.

## Common Bugs and How to Address Them

1. **Dataset Format Issues**: Ensure your dataset is in YOLO annotation format. Incorrect formatting can cause errors during data loading. Refer to the [FAQs](about.md) for details.
2. **Class Imbalance**: Imbalanced datasets can lead to poor model performance. Balance your dataset with equal parts of positive and negative samples.
3. **Memory Errors During Training**: If you encounter memory issues, reduce the batch size or input image resolution.
4. **Quantization Errors**: Ensure the representative dataset used for quantization matches the input shape of the model.
5. **Deployment Issues on Microcontrollers**: Verify that the model fits within the memory constraints of the target device. Enable PSRAM on ESP32-S3 if needed.

## Getting Started

To get started with Ecto-Trigger, follow these steps:

1. **Setup**: Install the required dependencies as described in the [Package Install Guide](packages.md).
2. **Train a Model**: Use `model_trainer.py` to train a model with your dataset. Refer to the [Usage Guide](usage.md) for detailed instructions.
3. **Evaluate the Model**: Use `model_evaluator.py` to evaluate the trained model and visualize its performance.
4. **Quantize the Model**: Use `model_quantiser.py` to convert the model to TFLite format for deployment.
5. **Deploy the Model**: Follow the [Deployment Guide](deployment.md) to deploy the quantized model on your target device.

## Future Directions

We aim to expand Ecto-Trigger by adding support for additional model architectures, improving deployment workflows, and providing more pre-trained models for various applications. Contributions from the community are welcome—please see the [Contributing](../README.md#contributing) section for details.

For a complete walkthrough of the pipeline and additional resources, refer to the [Usage Guide](usage.md) and [FAQs](about.md).
