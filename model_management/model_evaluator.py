import argparse
import numpy as np
from tensorflow.keras.models import load_model
from generator import CustomDataGenerator


class ModelEvaluator:
    """
    A class to handle model evaluation.

    Attributes:
        batch_size (int): Batch size for evaluation.
        weights_path (str): Path to the trained model weights.
        val_data_dir (str): Directory containing validation data.
    """

    def __init__(self, batch_size, weights_path, val_data_dir):
        """
        Initialize the ModelEvaluator with given parameters.

        Args:
            batch_size (int): Batch size for evaluation.
            weights_path (str): Path to the trained model weights.
            val_data_dir (str): Directory containing validation data.
        """
        self.batch_size = batch_size
        self.weights_path = weights_path
        self.val_data_dir = val_data_dir
        self.stop_training_flag = {'stop': False}

    def load_model(self):
        """
        Load the Keras model from the specified weights path.

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        print(f"Loading model from: {self.weights_path}")
        model = load_model(self.weights_path)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Model loaded successfully!")
        return model


    def create_data_generator(self, input_shape):
        """
        Create the data generator for validation.

        Args:
            input_shape (tuple): Input shape of the model.

        Returns:
            CustomDataGenerator: Instance of the custom data generator.
        """
        print(f"Creating validation data generator for directory: {self.val_data_dir}")
        return CustomDataGenerator(
            data_directory=self.val_data_dir,
            batch_size=self.batch_size,
            input_shape=input_shape,
            stop_training_flag=self.stop_training_flag,
            shuffle=False
        )

    def evaluate(self):
        """
        Evaluate the model on the validation data.

        Returns:
            tuple: Validation loss and accuracy.
        """
        model = self.load_model()
        input_shape = model.input_shape[1:]
        val_generator = self.create_data_generator(input_shape)
        print("Starting evaluation...")
        val_loss, val_accuracy = model.evaluate(val_generator)
        print("Evaluation complete!")
        return val_loss, val_accuracy


def main():
    """
    Main function for parsing arguments and running the evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate a MobileNetV2 model with custom parameters.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--weights_path', type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument('--val_data_dir', type=str, required=True, help="Directory containing validation data.")

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        batch_size=args.batch_size,
        weights_path=args.weights_path,
        val_data_dir=args.val_data_dir
    )

    val_loss, val_accuracy = evaluator.evaluate()
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
