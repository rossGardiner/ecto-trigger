#This file has been created with iterative consultation from the ChatGPT LLM, version 4o
import argparse
import tensorflow as tf
import numpy as np
import os
from model_loader import ModelLoader  
from generator import CustomDataGenerator


class ModelQuantiser:
    """
    A class to handle the quantization of a Keras model to TFLite with INT8 precision.

    Attributes:
        model (tf.keras.Model): The model to be quantised.
        representative_dataset (str): Path to the representative dataset for quantization.
    """

    def __init__(self, weights_file, representative_dataset, representative_example_nr):
        """
        Initialize the ModelQuantiser.

        Args:
            weights_file (str): Path to the Keras model weights file.
            representative_dataset (str): Directory containing the representative dataset.
        """
        self.weights_file = weights_file
        self.representative_dataset = representative_dataset
        self.representative_example_nr = representative_example_nr

        self.model = self._load_model()
        self.input_shape = self.model.input_shape[1:] 

        print(f"Preparing representative dataset from: {self.representative_dataset}")

        self.data_generator = CustomDataGenerator(
            data_directory=[self.representative_dataset],
            batch_size=1,
            input_shape=self.input_shape,
            stop_training_flag=False,
            shuffle=True
        )

    def _load_model(self):
        """
        Load the Keras model from the weights file.

        Returns:
            tf.keras.Model: Loaded model.
        """
        print(f"Loading model from: {self.weights_file}")
        model = ModelLoader.load_keras_model(self.weights_file)
        print("Model loaded successfully!")
        return model

    

    def quantise_model(self, output_path):
        """
        Quantise the model to TFLite format with INT8 precision.

        Args:
            output_path (str): Path to save the quantised TFLite model.
        """
        print("Starting model quantization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Define representative_dataset_gen function
        def representative_dataset_gen():
            """
            Generator for representative dataset samples.

            Yields:
                list: A batch of representative samples for quantization.
            """
            for i in range(self.representative_example_nr):
                img, _ = self.data_generator[i]
                print(f"Generating sample {i + 1}/{self.representative_example_nr}")
                yield [img.astype(np.float32)]

        # Set quantisation options
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = representative_dataset_gen

        # Convert the model
        tflite_quant_model = converter.convert()
        print("Model quantisation completed!")

        # Save the quantised model
        with open(output_path, 'wb') as f:
            f.write(tflite_quant_model)
        print(f"Quantised model saved to: {output_path}")

        # Test the quantised model
        self._test_quantised_model(tflite_quant_model)

    def _test_quantised_model(self, tflite_quant_model):
        """
        Test the quantised TFLite model by loading it into a TFLite interpreter.

        Args:
            tflite_quant_model (bytes): The quantised TFLite model.
        """
        print("Testing the quantised model...")
        interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
        interpreter.allocate_tensors()

        input_type = interpreter.get_input_details()[0]['dtype']
        output_type = interpreter.get_output_details()[0]['dtype']

        print(f"TFLite model input type: {input_type}")
        print(f"TFLite model output type: {output_type}")
        print("Quantised model test completed!")


def main():
    """
    Main function to handle command-line arguments and quantise the model.
    """
    parser = argparse.ArgumentParser(description="Quantise a Keras model to TFLite format with INT8 precision.")
    parser.add_argument('--weights_file', type=str, required=True, help="Path to the Keras model weights file.")
    parser.add_argument('--representative_dataset', type=str, required=True, help="Path to the representative dataset.")
    parser.add_argument('--representative_example_nr', type=int, required=False, default=500, help="Number of examples to take from the representative dataset." )
    parser.add_argument('--output', type=str, required=True, help="Path to save the quantised TFLite model.")
    args = parser.parse_args()

    # Initialize the ModelQuantiser
    quantiser = ModelQuantiser(
        weights_file=args.weights_file,
        representative_dataset=args.representative_dataset,
        representative_example_nr=args.representative_example_nr
    )

    # Quantise the model
    quantiser.quantise_model(output_path=args.output)


if __name__ == "__main__":
    main()
