# Package Install Guide

 It is reccomended to install these dependencies in a virtual environment, this is facilitated by `python`: 

```
$ python3 -m venv venv
$ source venv/bin/activate
```

Packages can thus be installed using `pip install <package>`.

Install the following requirements for instantiating and running the model.

```
keras==2.4.1
tensorflow==2.4.1
opencv-python==4.6.0
```

Install these requirements for model training

```
imgaug==0.4.0
numpy==1.23.5
```

Alternatively, to install all the required packages, use our requirements file:

```
pip install -r requirements.txt
```

To use the file `tflite_model_runner.py`, you must install the tflite runtime, this can be useful on devices where you just want to execute the model and don't nessicarily want to install the whole tensorflow package. 

```
pip install tflite_runtime
```