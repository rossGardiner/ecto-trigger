# Deployment

Here you will find information on how to deploy models you have trained using Ecto-Trigger. 

<details open> <summary>Raspberry Pi</summary>

To execute the models on Raspberry Pi systems, you can choose to use the Tensorflow or TFLite runtime (reccomended). To use the TFLite runtime, check out the guidance [here](https://ai.google.dev/edge/litert/microcontrollers/python). Basic steps:

(On RPi)
```
python3 -m pip install tflite-runtime
```

Using Python
```
from PIL import Image
import tflite_runtime.interpreter as tflite

def load_model_and_predict(image_path, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(image_path).resize((width, height))

    input_data = np.expand_dims(img, axis=0)
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)
    print(1 if prediction > 0.5 else 0)
```

</details>

<details open>
  <summary>ESP32-S3</summary>
  World!

So far, we have found success deploying Ecto-Trigger to ESP32-S3 devices using the [ESP-NN library](https://github.com/espressif/esp-nn) to accelerate inference time for TFLite models. We hope to provide a library for this in the future. For now, here are basic steps we took to enable this. 

### Development Environment

First you must set up a development environment which allows you to compile and run code for Espressive devices. Follow the [guidance from Espressive](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/index.html) to get the esp idf. Details are given below for Ubuntu users:

Dependencies:

```
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```
Download idf
```
mkdir -p ~/esp
cd ~/esp
git clone -b v5.4 --recursive https://github.com/espressif/esp-idf.git
```
Install idf
```
cd ~/esp/esp-idf
./install.sh esp32s3

```
### Using the idf

First, set environment variabes:
```
. $HOME/esp/esp-idf/export.sh
```
This ensures the `IDF_PATH` environment variable is set, allowing you to use `idf.py`. You must do this each time you open a new shell. 

### Person detection
Espressive already provide an example which runs a person detection neural network in thier code, we can modify this to run our own models. 

Get the example code:
```
cd ~/esp
cp -r $IDF_PATH/examples/get-started/person_detection .
```

Configure the project: 
```
cd ~/esp/hello_world
idf.py set-target esp32s3
idf.py menuconfig
```
A menu will now appear, allowing you to control the configuration of the ESP32-S3. It is important to enable PSRAM here as the Ecto-Trigger models are larger than can be allocated in the standard amount of SRAM on ESP32-S3. In the future, it would be interesting to reduce model sizes such that this isn't required. To enable the PSRAM, follow instructions [here](https://docs.espressif.com/projects/esp-idf/en/release-v4.4/esp32s3/api-guides/flash_psram_config.html): "Enable the CONFIG_ESP32S3_SPIRAM_SUPPORT under Component config / ESP32-S3-Specific menu.". 

Next, give the project a build and flash it to the plugged in board to ensure there are no issues:
```
idf.py build
idf.py flash
```

### Modification
All being well, we can modify the `person_detection` code for use with Ecto-Trigger models. First, parse the quantised `.tflite` models into C data arrays.
```
xxd -i /path/to/model.tflite > model.cc
```
Make a C header file for this. 


</details>




