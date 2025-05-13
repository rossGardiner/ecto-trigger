# Ecto-Trigger

Ecto-Trigger is a toolkit designed to help ecologists develop lightweight AI models which can automate species detection in camera trap images. It is especially useful for scenarios where traditional motion sensors aren't reliable, for example, detecting insects as they lack the body-heat required to trigger conventional PIR sensors. 

This project explains Ecto-Trigger, its use cases, how to use it and how to install it. Frequently asked questions are given below to provide initial insights and further chapters explain specific aspects:

- [Usage Guide](guides/usage.md) gives detailed information on each of the files within Ecto-Trigger and how you can use them from within a Python programme and from the command line. 
- [Deployment Guide](guides/deployment.md) gives information about how to use Ecto-Trigger in the field, on devices which form camera traps. 


All documentation is served via our [web-page](www.google.com), we have also made this accessible as a [pdf-format vignette](www.google.com). This all supports our paper: [Towards scalable insect monitoring: Ultra-lightweight CNNs as on-device triggers for insect camera traps](www.google.com), which contains more advanced technical details and background for our approach. 

## FAQs
### What is Ecto-Trigger and why would I use it?

Ecto-Trigger is a free and open-source tool-kit that helps you train and use small, efficient AI models to detect the presence (or absence) of a specific object (such as an insect) in camera trap images, and can be used on the camera trap device itself.

It is built for use by ecologists, especially those in a fieldwork scenario where traditional camera trap triggers (such as passive infrared, PIR, sensors) don't work well. For example: 

- PIR sensors can miss cold-blooded, small or fast-moving animals, such as insects
- They many often trigger unnecessarily from heat or movement not attributed to the target animal, such as rustling leaves

You would use Ecto-Trigger in a scenario where these limations are signficant, for example, you may have limited storage, bandwidth or human resources, so you cannot afford to continuously record video or time-lapse footage. You may also wish to use low-powered microcontrollers to create your camera trap as these require less energy and so can be easier to scale up or use in environments where solar power is not easy to access. 

Ecto-Trigger uses a computer vision based approach, relying on the contents of the image itself - not motion or heat - to decide whether to keep or discard each captured image. These models act a trigger system themselves, which uses only optical information from a camera. 


### How does it work? 

The core idea is that a simple, non-computationally expensive model can be trained to answer one specific question about an image which was just captured: "Does this image, which I'm seeing right now, contain the object I care about?"

The model uses a compact convolutional neural network architecture called MobileNetv2 to answer this. It has been trained on example images (some containing the target object, some not). The model weights we provide with this code-base are trained to detect insects, to help with the development of insect camera traps, but you can train your own variations easily using our tools. These models are engineered in such a way that once trained, they can run on small, affordable devices such as ESP32s3 microcontrollers, to filter a stream of images in real-time while consuming very little energy. These are also compatible with more capable computing platforms such as the Raspberry Pi. 

An example pipeline for how Ecto-Trigger models could be built into code running on a camera trap itself is provided below, further information is given in our paper:

[assets/pipeline.png]()

### Do I need to know AI to use Ecto-Trigger?

No, this code-base is designed to be beginner-friendly. To run the code and produce models you do not need to understand advanced concepts in deep learning, but some understanding of Python programming and data-wrangling may be useful to make it easier to get started especially if you wish to train your own models. 

### Code overview: What's included in the toolkit? 

| File | Purpose |
|------|---------|
| `model_loader.py` | Build and load deep learning models (for training use) |
| `model_trainer.py` | Train a model using your labeled image dataset |
| `model_evaluator.py` | Evaluate model performance on test data |
| `model_quantiser.py` | Convert trained models to `.tflite` for edge devices |
| `saliency_map_evaluator.py` | Visualize what the model focuses on |
| `tflite_model_runner.py` | Run `.tflite` models on Raspberry Pi (no TensorFlow needed) |

To find out more about how to use every file, check the [usage guidance](guides/usage.md), which includes a description of all the ins and outs. 

### How can I get started?

The first thing you need to do is install all the necessary packages for Ecto-Trigger to run using Python. We recommend using a virtual environment to keep things neater, and have added all base requirements to a text file for convenience. To do this, you can use the instructions below:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Once completed, the packages within `requirements.txt` will be installed. If you want to check this, you can use:

```bash
pip list
```

Which will print out all the packages installed in your virtual environment to the terminal window. 

We have provided further installation details, which are accessible via our [install guidance](guides/packages.md) page.


# Contributing

This work is distributed under the GPL-3.0, see [LICENSE](LICENSE), which means it is open-source, free to use, and free to modify and distribute. We encourage others to contribute to help grow the code-base and add any features they may need or want. To do this, you can make a [pull request](<redacted>). 

 

# Citation
```
tbc
```



