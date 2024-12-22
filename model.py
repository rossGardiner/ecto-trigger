#This file has been created with iterative consultation from the ChatGPT LLM, version 4o

import os
import numpy as np
import argparse
import tensorflow as tf

from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow_datasets as tfds
from generator import CustomDataGenerator
from callbacks import SaveWeightsCallback
import cv2

def preprocess(image, label, input_shape, MODEL):
    image = tf.image.resize(image, (input_shape[0], input_shape[1]))

    # If the input shape indicates grayscale (w, h, 1), convert the image to grayscale
    if input_shape[2] == 1:
        image = tf.image.rgb_to_grayscale(image)

    if MODEL == "Mobnet":
        image = tf.keras.applications.mobilenet.preprocess_input(image)
    if MODEL == "Mobnetv2":
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def save_first_image(dataset, filename):
    for image_batch, label_batch in dataset.take(1):
        first_image = image_batch[0].numpy()
        first_label = label_batch[0].numpy()
        
        # Save the image to a file
        image = (first_image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # Print the label
        print("Label:", first_label)
        break


def train_model(DROPOUT, BATCH_SZ, INPUT_SHAPE, ALPHA, LOG_DIR, MODEL, USE_PRETRAIN):
    # Paths to your training and validation data directories
    #train_data_directory = ['/jmain02/home/J2AD013/dxa01/rxg80-dxa01/fomo_scratch/inat2017_yolo',
    #                        '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/fomo_scratch/inat2017_yolo_train_plantae']
    train_data_directory = ['/jmain02/home/J2AD013/dxa01/rxg80-dxa01/binary-classifer-insect-model/balanced_binary_insect_set2']
    validation_data_directory = ['/jmain02/home/J2AD013/dxa01/rxg80-dxa01/binary-classifer-insect-model/balanced_binary_insect_set_val2']
    #['/jmain02/home/J2AD013/dxa01/rxg80-dxa01/fomo_scratch/inat2017_yolo_val',
    #                            # '/jmain02/home/J2AD013/dxa01/rxg80-dxa01/fomo_scratch/inat2017_yolo_val_plantae', 
    #                            #]

    # Stop training flag
    stop_training_flag = {'stop': False}

    # Create the data generators
    train_generator = CustomDataGenerator(
        data_directory=train_data_directory,
        batch_size=BATCH_SZ,
        input_shape=INPUT_SHAPE,
        stop_training_flag=stop_training_flag,
        shuffle=True
    )

    validation_generator = CustomDataGenerator(
        data_directory=validation_data_directory,
        batch_size=BATCH_SZ,
        input_shape=INPUT_SHAPE,
        stop_training_flag=stop_training_flag,
        shuffle=False
    )

    dataset_name = 'i_naturalist2021'
    (ds_train, ds_validation), ds_info = tfds.load(
        dataset_name,
        split=['mini', 'val'],
        with_info=True,
        as_supervised=True
    )

    ds_train = ds_train.map(lambda image, label: preprocess(image, label, INPUT_SHAPE, MODEL=MODEL)).shuffle(1000).batch(BATCH_SZ).prefetch(tf.data.experimental.AUTOTUNE)
    ds_validation = ds_validation.map(lambda image, label: preprocess(image, label, INPUT_SHAPE, MODEL=MODEL)).batch(BATCH_SZ).prefetch(tf.data.experimental.AUTOTUNE)
    
    save_first_image(ds_train, "train1.png")

    weights=None if not USE_PRETRAIN else os.path.join(LOG_DIR, "pretrain/checkpoints/weights.100.hdf5")
    print(MODEL)
    print(weights)
    if MODEL=="Mobnet":
        base_model = MobileNet(input_shape=INPUT_SHAPE, include_top=True, alpha=ALPHA, weights=weights, classes=ds_info.features['label'].num_classes)
    elif MODEL=="Mobnetv2":
        # Load the MobileNetV2 model with pre-trained weights
        base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=True, alpha=ALPHA, weights=weights, classes=ds_info.features['label'].num_classes)
    elif MODEL=="Mobnetv3Small":
        dropout_rate=0.0
        if DROPOUT:
            dropout_rate = 0.5
            DROPOUT=False
        base_model = MobileNetV3Small(input_shape=INPUT_SHAPE, include_top=False, alpha=ALPHA, weights=weights, dropout_rate=dropout_rate)#, include_preprocessing=False)
    elif MODEL=="Mobnetv3Large":
        dropout_rate=0.0
        if DROPOUT:
            dropout_rate = 0.5
            DROPOUT=False
        base_model = MobileNetV3Large(input_shape=INPUT_SHAPE, include_top=False, alpha=ALPHA, weights=weights, dropout_rate=dropout_rate)#, include_preprocessing=False)
    # Freeze the base model
    base_model.trainable = True
    model=base_model

    if not USE_PRETRAIN:
        # Pretrain on inat21 mini set 
        # model = tf.keras.Sequential([
        # base_model,
        #     tf.keras.layers.GlobalAveragePooling2D(),
        #     tf.keras.layers.Dense(ds_info.features['label'].num_classes, activation='softmax')
        # ])

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        callbacks = [
                ModelCheckpoint(
                    filepath=os.path.join(LOG_DIR, 'checkpoints/weights.{epoch:02d}-{loss:.3f}.hdf5'),
                    save_weights_only=False,
                    monitor='loss',  # or 'val_accuracy' if you are monitoring accuracy
                    mode='min',  # or 'max' if you are monitoring accuracy
                    save_freq=500,
                    verbose=1,
                    save_best_only=True
                ),
                ModelCheckpoint(
                    filepath=os.path.join(LOG_DIR, 'checkpoints/weights.{epoch:02d}.hdf5'),
                    save_weights_only=False,
                    monitor='loss',  # or 'val_accuracy' if you are monitoring accuracy
                    mode='min',  # or 'max' if you are monitoring accuracy
                    save_freq="epoch",
                    verbose=1,
                    save_best_only=False
                ),
                TensorBoard(
                    log_dir=LOG_DIR,
                    write_graph=False,
                    write_images=True,
                    update_freq=500,  # update every n batches
                    profile_batch=0,  # Disable profiling
                    embeddings_freq=0  # Disable embedding visualization
                )]

        # Pretraining 
        history = model.fit(
            ds_train, 
            epochs=100,
            validation_data=ds_validation,
            callbacks=[ModelCheckpoint(
                    filepath=os.path.join(os.path.join(LOG_DIR, "pretrain"), 'checkpoints/weights.{epoch:02d}-{loss:.3f}.hdf5'),
                    save_weights_only=False,
                    monitor='loss',  # or 'val_accuracy' if you are monitoring accuracy
                    mode='min',  # or 'max' if you are monitoring accuracy
                    save_freq=500,
                    verbose=1,
                    save_best_only=True
                ),
                ModelCheckpoint(
                    filepath=os.path.join(os.path.join(LOG_DIR, "pretrain"), 'checkpoints/weights.{epoch:02d}.hdf5'),
                    save_weights_only=False,
                    monitor='loss',  # or 'val_accuracy' if you are monitoring accuracy
                    mode='min',  # or 'max' if you are monitoring accuracy
                    save_freq="epoch",
                    verbose=1,
                    save_best_only=False
                ),
                TensorBoard(
                    log_dir=os.path.join(LOG_DIR, "pretrain"),
                    write_graph=False,
                    write_images=True,
                    update_freq=500,  # update every n batches
                    profile_batch=0,  # Disable profiling
                    embeddings_freq=0  # Disable embedding visualization
                )]
        )
    # remove the inat top classifiying 10000 classes
    model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

    model.summary()
    # add dropout and pooling layers for regularisation and add a new binary classification head
    if DROPOUT:
        model = Sequential([
            model,
            GlobalAveragePooling2D(),
            Dropout(0.5),  # Add dropout with 50% rate
            Dense(1, activation='sigmoid')
        ])
    else:
        model = Sequential([
            model,
            GlobalAveragePooling2D(),
            Dense(1, activation='sigmoid')
        ])

    # add quantisation aware training (TFMOT)
    #model = tfmot.quantization.keras.quantize_model(model)

    # compile new model configuration 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[ModelCheckpoint(
                filepath=os.path.join(LOG_DIR, 'checkpoints/weights.{epoch:02d}-{loss:.3f}.hdf5'),
                save_weights_only=False,
                monitor='loss',  # or 'val_accuracy' if you are monitoring accuracy
                mode='min',  # or 'max' if you are monitoring accuracy
                save_freq=500,
                verbose=1,
                save_best_only=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(LOG_DIR, 'checkpoints/weights.{epoch:02d}.hdf5'),
                save_weights_only=False,
                monitor='loss',  # or 'val_accuracy' if you are monitoring accuracy
                mode='min',  # or 'max' if you are monitoring accuracy
                save_freq="epoch",
                verbose=1,
                save_best_only=False
            ),
            TensorBoard(
                log_dir=LOG_DIR,
                write_graph=False,
                write_images=True,
                update_freq=500,  # update every n batches
                profile_batch=0,  # Disable profiling
                embeddings_freq=0  # Disable embedding visualization
            )]
    )
def parse_tuple(input_string):
    """
    Helper function to parse a tuple from a string.
    Example: "1080,1080,3" -> (1080, 1080, 3)
    """
    return tuple(map(int, input_string.strip('()').split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with custom parameters.')
    parser.add_argument('--DROPOUT', type=bool, default=True, help='Whether to use dropout in the model', action=argparse.BooleanOptionalAction)
    parser.add_argument('--BATCH_SZ', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--INPUT_SHAPE', type=str, default="(1080, 1080, 3)", help='Input shape for the model')
    parser.add_argument('--ALPHA', type=float, default=0.1, help='Alpha parameter for MobileNetV2')
    parser.add_argument('--LOG_DIR', type=str, default='logs/1/', help='Directory for logging and checkpoints')
    parser.add_argument('--MODEL', type=str, default='Mobnet', help='Model type to use')
    parser.add_argument('--IMAGENET', type=bool, default=True, help='Use pretrained imagenet weights', action=argparse.BooleanOptionalAction)
    parser.add_argument('--USE_PRETRAIN', type=bool, default=True, help="Use a pretrained path and dont bother with inat pretraining epochs",  action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    input_shape = parse_tuple(args.INPUT_SHAPE)
    train_model(args.DROPOUT, args.BATCH_SZ, input_shape, args.ALPHA, args.LOG_DIR, args.MODEL, args.USE_PRETRAIN)
