#This file has been created with iterative consultation from the ChatGPT LLM, version 4o
import os
import numpy as np
import argparse
print("imported os, numpy and argparse")

import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, DepthwiseConv2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model
print("imported tensorflow")

import pathlib
import cv2 
import math
print("imported math")
import pandas as pd
print("imported pandas")




def preprocess(image, label, input_shape, MODEL):
    #image = tf.image.resize(image, (input_shape[0], input_shape[1]))

    # If the input shape indicates grayscale (w, h, 1), convert the image to grayscale
    image = tf.image.resize(image, (input_shape[0], input_shape[1]))

    # If the input shape indicates grayscale (w, h, 1), convert the image to grayscale
    if input_shape[2] == 1:
        image = tf.image.rgb_to_grayscale(image)


    if MODEL == "Mobnet":
        image = tf.keras.applications.mobilenet.preprocess_input(image)
    if MODEL == "Mobnetv2":
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label



#data_directory = '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/fomo_scratch/inat2017_yolo_val'
data_directory = ["/jmain02/home/J2AD013/dxa01/rxg80-dxa01/fomo_scratch/inat2017_yolo_val", 
                    "/jmain02/home/J2AD013/dxa01/rxg80-dxa01/fomo_scratch/inat2017_yolo_val_plantae",
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/insect_detect/dataset_240x135', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/insect_detect/dataset_val480x270', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/insect_detect/dataset_val960x540', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/insect_detect/dataset_1920x1080', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ecostack/dataset_240x135', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ecostack/dataset_480x270', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ecostack/dataset_960x540', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ecostack/data/val1201',
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/insect_detect/train/images',
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ami-traps/ami_traps/camera_trap_images/dataset_256x135', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ami-traps/ami_traps/camera_trap_images/dataset_512x270', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ami-traps/ami_traps/camera_trap_images/dataset_1024x540', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ami-traps/ami_traps/camera_trap_images/dataset_2048x1080', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ami-traps/ami_traps/camera_trap_images/images',
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/pollinator_detect/valColor', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/pollinator_detect/dataset_240x135',
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/pollinator_detect/dataset_480x270', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/pollinator_detect/dataset_960x540', 
                    '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/binary-classifer-insect-model/balanced_binary_insect_set_val2']
#data_directory =  '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/ecostack/dataset_A'
print(data_directory)

def make_model(LOG_DIR, USE_PRETRAIN):
    # Paths to your training and validation data directories
    

    # Stop training flag
    stop_training_flag = {'stop': False}    

   
    weights=os.path.join(LOG_DIR + "/checkpoints", "weights.10.hdf5") if not USE_PRETRAIN else os.path.join(LOG_DIR, "pretrain/checkpoints/weights.100.hdf5")
    #print(MODEL)
    print(weights)
    model = load_model(weights)
    new_model = Sequential()
    for layer in model.layers:
        if not isinstance(layer, Dropout):
            new_model.add(layer)
        else:
            print(f"removed dropout layer: {layer}")

    model = new_model
    # model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    # model = Sequential([
    #     model,
    #     GlobalAveragePooling2D(),
    #     Dropout(0.0),
    #     Dense(1, activation="sigmoid")
    # ])
    model.summary()
    print(model.input)
    #exit(0)
    # if MODEL=="Mobnet":
    #     base_model = MobileNet(input_shape=INPUT_SHAPE, include_top=False, alpha=ALPHA, weights=None, classes=10000)
    # elif MODEL=="Mobnetv2":
    #     # Load the MobileNetV2 model with pre-trained weights
    #     base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, alpha=ALPHA, weights=None, classes=10000)
    
    # base_model.load(weights)#= tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    # predictions = Dense(1, activation="sigmoid")(x)
    # model = tf.keras.Model(inputs = base_model.input, outputs=predictions)
    # print(len(model.layers))
    # # model = Sequential([
    # #     base_model,
    # #     GlobalAveragePooling2D(),
    # #     Dropout(0.5),
    # #     Dense(1, activation="sigmoid")
    # # ])
    # model.load(weights)
    # remove the inat top classifiying 10000 classes
    #model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

    
    # add dropout and pooling layers for regularisation and add a new binary classification head
    # if DROPOUT:
    # model = Sequential([
    #     model,
    #     GlobalAveragePooling2D(),
    #     Dropout(0.5),  # Add dropout with 50% rate
    #     Dense(1, activation='sigmoid')
    # ])
    # else:
    # m = Sequential()
    # for layer in model.layers:
    #     print(layer)
    #     m.add(layer)
    # model.add(GlobalAveragePooling2D)
    # model.add(Dense(1, activation="sigmoid"))
    # model = Sequential([
    #     model,
    #     GlobalAveragePooling2D(),
    #     Dense(1, activation='sigmoid')
    # ])
    

   

    # compile new model configuration 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.load_weights(weights)
    return model

from generator import CustomDataGenerator

def make_quant_model(model, INPUT_SHAPE):
    
    dg = CustomDataGenerator(data_directory=['/jmain02/home/J2AD013/dxa01/rxg80-dxa01/binary-classifer-insect-model/balanced_binary_insect_set'], 
                                batch_size=500, input_shape=INPUT_SHAPE, stop_training_flag=False)
    def representative_dataset_gen():
        for i in range(500):
            print(f"gen {i} \r")
            # Get sample input data as a numpy array in a method of your choosing.
            img = dg.read_img(dg.image_paths[i])
            print(f"shape {img.shape}")

            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            yield [img.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    print("quant model created!")
    #check interpreter loads
    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)
    
    return tflite_quant_model


def parse_id(file_id):
    #returns the average area of bounding boxes in an image
    image = cv2.imread(file_id + ".jpg")
    annotation = file_id + ".txt"
    imsize = image.shape[:2]
    areas = []
    classes = []
    if os.path.exists(annotation) and os.path.getsize(annotation) > 0:
        for line in open(annotation):
            area, class_id, _ = parse_yolo_bbox_line(line, imsize[0], imsize[1])
            areas.append(area)
            classes.append(class_id)
        return sum(areas) / len(areas), classes
    else:
        return -1.0, classes 



def parse_yolo_bbox_line(line, img_width, img_height):
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    # Convert normalized coordinates to pixel values
    x_center_pixel = x_center * img_width
    y_center_pixel = y_center * img_height
    width_pixel = width * img_width
    height_pixel = height * img_height

    # Calculate the bounding box area in pixel values
    area = math.sqrt(width_pixel * height_pixel) / math.sqrt(img_width * img_height)

    return area, class_id, (x_center_pixel, y_center_pixel, width_pixel, height_pixel)

def get_path_without_last_extension(file_path):
    # Split the file path into root and extension
    root, ext = os.path.splitext(file_path)
    
    # Check if there's another extension left to remove
    # while ext:
    #     root, ext = os.path.splitext(root)
    
    return root

from glob import glob

def eval(model, quant_model, input_size, filename="out.csv", csv=None, MODEL="Mobnet"):
    if not csv:
        if isinstance(data_directory, list):
            filenames = []
            for path in data_directory:
                filenames += glob(os.path.join(path, "*.jpg"))
        else:
            filenames = glob(os.path.join(data_directory, "*.jpg"))
        sizes = []
        classes_list = []
        for file in filenames:
            bbox_size, classes = parse_id(get_path_without_last_extension(file))
            sizes.append(bbox_size)
            classes_list.append(classes)
        # Create a DataFrame
        df = pd.DataFrame({'filename': filenames, 'size': sizes, 'classes_list': classes_list})
        #df = df.sort_values(by='size')
        
        df.to_csv('test.csv')
    else:
        df = pd.read_csv("test.csv")
    # Determine the bins
    
    
    # Add a column to DataFrame with bin assignment
    each_prediction = np.zeros((len(df["filename"].tolist())))
    df['predictions'] = each_prediction
    df['predictions_quant'] = each_prediction
    interpreter = tf.lite.Interpreter(model_content=quant_model)
    interpreter.allocate_tensors()
    print(input_size[:2])
    for i, a in enumerate(df["filename"].tolist()):
        img, lab = open_img(a, input_shape, MODEL)
        preds = run_inference(model, [img])
        df.loc[i, 'predictions'] = np.max(preds)
        df.loc[i, 'predictions_quant'] = run_inference_quant(interpreter, img)
        
    df.to_csv(filename, index=False)
        
def open_img(filename, input_shape, MODEL):
    im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (input_shape[1], input_shape[0]))
    #print(im.shape)
    #exit(0)
    #im = cv2.transpose(im)
    if input_shape[2] == 1:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = np.expand_dims(im, axis=-1) 
    im = tf.keras.preprocessing.image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    
    #im = np.ndarray.astype(im, np.uint8)
    if MODEL == "Mobnet":
        im = tf.keras.applications.mobilenet.preprocess_input(im)
    if MODEL == "Mobnetv2":
        im = tf.keras.applications.mobilenet_v2.preprocess_input(im)
    #im = tf.convert_to_tensor(im)
    

    return im, 1







# Function to run inference on the frame
def run_inference(model, img_array):
    preds = model.predict(img_array)
    return preds

def run_inference_quant(interpreter, img_array):
    normed = convert_to_uint8(arr=img_array)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], normed)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    return output[0][0]

def convert_to_uint8(arr):
    # Step 1: Normalize the values from [-1.0, 1.0] to [0, 1]
    normalized_arr = (arr + 1) / 2

    # Step 2: Scale the values from [0, 1] to [0, 255]
    scaled_arr = normalized_arr * 255

    # Step 3: Convert the array to uint8 type
    uint8_arr = scaled_arr.astype(np.uint8)

    return uint8_arr











def parse_tuple(input_string):
    """
    Helper function to parse a tuple from a string.
    Example: "1080,1080,3" -> (1080, 1080, 3)
    """
    return tuple(map(int, input_string.strip('()').split(',')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with custom parameters.')
    parser.add_argument('--DROPOUT', type=bool, default=True, help='Whether to use dropout in the model', action=argparse.BooleanOptionalAction)
    parser.add_argument('--BATCH_SZ', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--INPUT_SHAPE', type=str, default="(96, 96, 1)", help='Input shape for the model')
    parser.add_argument('--ALPHA', type=float, default=0.2, help='Alpha parameter for MobileNetV2')
    parser.add_argument('--LOG_DIR', type=str, default='logs/5/', help='Directory for logging and checkpoints')
    parser.add_argument('--MODEL', type=str, default='Mobnetv2', help='Model type to use')
    parser.add_argument('--IMAGENET', type=bool, default=True, help='Use pretrained imagenet weights', action=argparse.BooleanOptionalAction)
    parser.add_argument('--USE_PRETRAIN', type=bool, default=True, help="Use a pretrained path and dont bother with inat pretraining epochs", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    input_shape = parse_tuple(args.INPUT_SHAPE)
    #model = make_model(args.DROPOUT, args.BATCH_SZ, input_shape, args.ALPHA, args.LOG_DIR, args.MODEL, args.USE_PRETRAIN)
    model = make_model(args.LOG_DIR, args.USE_PRETRAIN)
    quant_save_name = "eval26082024/" + args.LOG_DIR.split("/")[1] + "_int8.tflite"
    p = pathlib.Path(quant_save_name)

    if not p.exists():
        quant_model = make_quant_model(model, input_shape)
        p.write_bytes(quant_model)
    else:
        quant_model = p.read_bytes()
    
    
    filename = "eval26082024/" + args.LOG_DIR.split("/")[1] + ".csv"
    print(filename)
    eval(model, quant_model, input_shape, filename=filename, csv=None, MODEL=args.MODEL)

    
