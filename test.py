#This file has been created with iterative consultation from the ChatGPT LLM, version 4o

import numpy as np
import argparse
from tensorflow.keras.models import load_model
from generator import CustomDataGenerator

def evaluate_model(BATCH_SZ, INPUT_SHAPE, MODEL, WEIGHTS_PATH, VAL_DATA_DIRECTORY):
    # Stop training flag
    stop_training_flag = {'stop': False}

    # Create the data generator for validation
    val_generator = CustomDataGenerator(
        data_directory=VAL_DATA_DIRECTORY,
        batch_size=BATCH_SZ,
        input_shape=INPUT_SHAPE,
        stop_training_flag=stop_training_flag,
        shuffle=False  # No need to shuffle during validation
    )

    print(f"Model: {MODEL}")
    #print(MODEL)
    print(WEIGHTS_PATH)
    model = load_model(WEIGHTS_PATH)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_generator)

    return val_loss, val_accuracy

def parse_tuple(input_string):
    """
    Helper function to parse a tuple from a string.
    Example: "(96, 96, 1)" -> (96, 96, 1)
    """
    return tuple(map(int, input_string.strip('()').split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model with custom parameters.')
    parser.add_argument('--BATCH_SZ', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--INPUT_SHAPE', type=str, default="(1080, 1080, 3)", help='Input shape for the model')
    parser.add_argument('--ALPHA', type=float, default=0.2, help='Alpha parameter for MobileNetV2')
    parser.add_argument('--WEIGHTS_PATH', type=str, default='logs/1/checkpoints/weights.epoch01.hdf5', help='Path for trained weights checkpoint')
    parser.add_argument('--MODEL', type=str, default='Mobnetv2', help='Model type to use')
    parser.add_argument('--VAL_DATA_DIRECTORY', type=str, default='/jmain02/home/J2AD013/dxa01/rxg80-dxa01/binary-classifer-insect-model/balanced_binary_insect_set_val2', help='val data')

    args = parser.parse_args()
    
    input_shape = parse_tuple(args.INPUT_SHAPE)
    
    result = evaluate_model(args.BATCH_SZ, input_shape, args.MODEL, args.WEIGHTS_PATH, args.VAL_DATA_DIRECTORY)
    print(f"Validation Loss: {result[0]}, Validation Accuracy: {result[1]}")
