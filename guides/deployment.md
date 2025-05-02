# Deployment Guide for Ecto-Trigger

This guide explains how to run a trained Ecto-Trigger model on real devices in the field. We provide instructions for two supported platforms: 

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

Deploying models onto microcontroller platforms is a little more complicated, as these don't usually support python, so we have to compile code from scratch to execute on each device. This can be quite a complicated process and might require some engineering knowledge. To make things easy, we have provided an example project which uses our models on ESP32s3 chipset with the Platformio extension for VSCode. 

We have made a separate repository for this, which includes full guidance and further details. 

[<Redacted for review>]()

</details>

If you have not yet trained or quantised a model first, use one of ours developed for insect detection, or follow the our usage [guidance](guides/usage.md) to train your own. 




