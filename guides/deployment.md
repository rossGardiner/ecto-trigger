# Deployment

Here you will find information on how to deploy models you have trained using Ecto-Trigger.  

<details> <summary>Raspberry Pi</summary>

To execute the models on Raspberry Pi systems, you can choose to use the Tensorflow or TFLite runtime (reccomended). To use the TFLite runtime, check out the guidance [here](https://ai.google.dev/edge/litert/microcontrollers/python). Basic steps:

(On RPi)
```
python3 -m pip install tflite-runtime
```

Using Python, you can execute a quantised inference: 
```
import numpy as np
from tflite_model_runner import TFLiteModelRunner

q_model = TFLiteModelRunner.load_tflite_model("model_weights/8/quant/8_int8.tflite")

input_image_array = np.random.uniform(0, 255, size=(q_model.get_input_details()[0]["shape"][1:])).astype(np.uint8)
input_image_array = np.expand_dims(input_image_array, axis=0)

q_model.set_tensor(q_model.get_input_details()[0]["index"], input_image_array)
q_model.invoke()

output = q_model.get_tensor(q_model.get_output_details()[0]["index"])
print(output[0]) # remember that the output will be in confidence range 0-255
```

</details>

<details>
  <summary>ESP32-S3</summary>

So far, we have found success deploying Ecto-Trigger to ESP32-S3 devices using the [ESP-NN library](https://github.com/espressif/esp-nn) to accelerate inference time for TFLite models. This is a much more involved process and requires quite a lot of background research and debugging. We hope to provide a library for this in the future to make things easier. For now, here are basic steps we took to enable this. 

\### Development Environment

First you must set up a development environment which allows you to compile and run code for Espressive devices. Follow the [guidance from Espressive](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/index.html) to get the esp idf. Details are given below for Ubuntu users:

Dependencies:

```
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```
Download
```
mkdir -p ~/esp
cd ~/esp
git clone -b v5.4 --recursive https://github.com/espressif/esp-idf.git
```
Install 
```
cd ~/esp/esp-idf
./install.sh esp32s3

```
\### Using the idf

First, set environment variabes:
```
. $HOME/esp/esp-idf/export.sh
```
This ensures the `IDF_PATH` environment variable is set, allowing you to use `idf.py`. You must do this each time you open a new shell. 

\### Person detection
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
You can also check the output to see runtime errors or programme outputs
```
idf.py monitor
```

\### Modification
All being well, we can modify the `person_detection` code for use with Ecto-Trigger models. First, parse the quantised `.tflite` models into C data arrays.
```
xxd -i /path/to/model.tflite > model.cc
```
You also have to make a C header file for this. To make things easier, we provide these header files and C data arrays for our trained models in the [model_weights](../model_weights/) directory (e.g. `model_weights/8/quant/8_model_data.cc` or `.h`). 

Add these files to the `main` directory, all the modifications will be in here. 

Finally, replace the `main_functions.cc` file with our modified one provided [here](./esp32s3), and do the same for `CMakeLists.txt`. This concludes the modifications, the programme will now execute one of the Eco-Trigger models. 

You can build the code with the model that you want using:
```
idf.py build -DMODEL=<number>
idf.py flash
idf.py monitor
```
Finally, setting up the camera compatibility will be different depending on the development board configuration. You may have to look at the pinout and modify `app_camera_esp.h`, we have provided ours for reference. The following config worked for us:
```
#define CAMERA_MODULE_NAME "ESP-S3-EYE"
#define CAMERA_PIN_PWDN -1
#define CAMERA_PIN_RESET 18 //-1

#define CAMERA_PIN_VSYNC 6
#define CAMERA_PIN_HREF 7
#define CAMERA_PIN_PCLK 13
#define CAMERA_PIN_XCLK 14

#define CAMERA_PIN_SIOD 4
#define CAMERA_PIN_SIOC 5

#define CAMERA_PIN_D0 11
#define CAMERA_PIN_D1 9
#define CAMERA_PIN_D2 8
#define CAMERA_PIN_D3 10
#define CAMERA_PIN_D4 12
#define CAMERA_PIN_D5 17 //18
#define CAMERA_PIN_D6 16 //17
#define CAMERA_PIN_D7 15 //16
```
</details>




