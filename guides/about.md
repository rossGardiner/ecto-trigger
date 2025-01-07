# What is it? 

Ecto-Trigger is a code-base which supports the development of lightweight MobileNetv2 models for (prescence/abscence) binary image classification. This software supports our paper: [Towards Scalable Insect Monitoring: Ultra-Lightweight CNNs as On-Device Triggers for Insect Camera Traps](https://arxiv.org/abs/2411.14467) (currently pre-print), where we use this as a method to detect insect ectotherms in natural images. 

Here you will find code for training, testing and deployment of binary classifier models, including saliency map analysis tools and quantisation toolkits. Our pre-trained weights for insect detection are also available with a range of model sizes. This is discussed in more depth in the paper. 

# How can I train my own model?

You can train your own model by following these steps:

1. Dataset
The code-base expects data in YOLO annotation format, see [here](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format). Ecto-Trigger models are binary classifiers and contain no method of dealing with class imbalance at train time, so its important to balance the dataset to contain roughly equal parts empty images and images containing the object(s) of interest. The datasets used for training/evaluation of the models presented in our paper are available at [zendodo](todo).

2. Model Instantiation/Training
You can instantiate a model and train it by following the [usage guidance](./usage.md). 

3. Evaluation 
This is facilitated via several scripts, saliency maps and accuracy computation can be computed following the [guidance](./usage.md).

4. Quantisation
To quantise models, making them suitable for inference on microcontrollers, you can use our [quantisation tool](./usage.md). 

5. Deployment
For guidance on how to deploy a quantised model, see [here]().


## Performance

We tested Ecto-Trigger on a challenging use-case, finding  images which contain insects. 

## How can I cite Ecto-Trigger?

If you find Ecto-Trigger helpful, please cite our paper:

```
tbc
```




