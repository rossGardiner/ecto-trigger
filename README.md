# Ecto-Trigger
This repository contains accompanying code, documentation and guides supporting our paper: [Towards Scalable Insect Monitoring: Ultra-Lightweight CNNs as On-Device Triggers for Insect Camera Traps](https://arxiv.org/abs/2411.14467). Ecto-Trigger is software to produce lightweight binary classification models to detect images containing objects of interest, this is intended to be a lightweight alternative to object detection models, for scenarios where the location of the object is not required but computationally efficient models are important. 

Ecto-Trigger has been developed for insect camera traps, where insects (ectotherms) cannot activate traditional PIR camera trap triggers. Lightweight binary classifiers can be deployed locally on microcontrollers and continuously run to filter insect images in real-time. 

This repository contains information to build, deploy and evaluate your own binary classifiers, as well as the existing model weights from our paper for recognising images containing insects.

# Guides

Several markdown pages have been produced for user guidance. These are tabulated below:

1. [FAQs](guides/about.md)
2. [Setup Instructions](guides/packages.md)
3. [Usage](guides/usage.md)
4. [Deployment](guides/deployment.md)

# Documentation

Documentation for our code-base is served [here](https://ross-jg.github.io/ecto-trigger/html/). For a full list of classes, see the [Classes page](https://ross-jg.github.io/ecto-trigger/html/annotated.html). 


# Contributing

Ecto-Trigger is open-source, we encourage others to contribute. Please do so by making a [pull request](https://github.com/ross-jg/ecto-trigger/pulls). 

# License 

Distributed under the GPL-3.0, see [LICENSE](LICENSE) for more information.



