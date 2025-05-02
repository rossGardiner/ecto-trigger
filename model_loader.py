#This file has been created with iterative consultation from the ChatGPT LLM, version 4o
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import load_model
import pathlib

class ModelLoader:
    """A utility class to load Keras and TFLite models."""
    @staticmethod
    def create_model(input_shape, alpha, dropout_rate=0.5, freeze_base=False):
        """
        Create a binary classification model using MobileNetV2 as the base. This method uses the MobileNetv2 implementation from Keras and adds 

        Args:
            input_shape (tuple): Input shape for the model (height, width, channels).
            alpha (float): Width multiplier for MobileNetV2.
            dropout_rate (float): Dropout rate for the dropout layer. Default is 0.5.
            freeze_base (bool): Whether or not to make the base model trainable. 
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        try:
            print(f"Creating MobileNetV2 model with input_shape={input_shape}, alpha={alpha}")
            
            # Load MobileNetV2 as the base model
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,  # Exclude the top layers to add custom head
                alpha=alpha,
                weights=None  # Start with random weights for training
            )

            # Freeze the base model to retain feature extraction during training
            base_model.trainable = (not freeze_base)

            # Create the binary classification head
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dropout(dropout_rate),  # Regularization
                Dense(1, activation='sigmoid')  # Binary classification
            ])

            # Compile the model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            print("Model created successfully!")
            model.summary()
            return model
        except Exception as e:
            print(f"Failed to create MobileNetV2 model: {e}")
            raise



    @staticmethod
    def load_keras_model(weights_path: str):
        """
        Load a trained Keras model from an HDF5 file and print its summary.
        
        Args:
            weights_path (str): Path to the .hdf5 weights file.

        Returns:
            tf.keras.Model: Loaded Keras model.
        """
        try:
            print(f"Loading Keras model from: {weights_path}")
            model = load_model(weights_path)
            print("Model loaded successfully!")
            print("Model Summary:")
            model.summary()
            return model
        except Exception as e:
            print(f"Failed to load Keras model: {e}")
            raise

    