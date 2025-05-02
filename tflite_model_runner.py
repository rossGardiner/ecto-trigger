
import pathlib
import numpy as np
import tflite_runtime.interpreter as tflite

class TFLiteModelRunner:
    @staticmethod
    def load_tflite_model(tflite_path: str):
        """
        Load a TFLite model for inference and print input/output details.

        Args:
            tflite_path (str): Path to the .tflite file.

        Returns:
            tf.lite.Interpreter: TFLite interpreter with the model loaded.
        """
        try:
            print(f"Loading TFLite model from: {tflite_path}")
            tflite_model = pathlib.Path(tflite_path).read_bytes()
            interpreter = tflite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            print("TFLite model loaded successfully!")
            
            # Print input and output details
            input_details, output_details = TFLiteModelRunner.get_tflite_input_output_details(interpreter)
            print("\nTFLite Model Input Details:")
            for detail in input_details:
                print(detail)
            print("\nTFLite Model Output Details:")
            for detail in output_details:
                print(detail)

            return interpreter
        except Exception as e:
            print(f"Failed to load TFLite model: {e}")
            raise

    @staticmethod
    def get_tflite_input_output_details(interpreter):
        """
        Get input and output details of a loaded TFLite model.

        Args:
            interpreter (tf.lite.Interpreter): Loaded TFLite model interpreter.

        Returns:
            tuple: (input_details, output_details)
        """
        try:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            return input_details, output_details
        
        except Exception as e:
            print(f"Failed to fetch TFLite model details: {e}")
            raise
