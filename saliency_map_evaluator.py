import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from model_loader import ModelLoader  


class SaliencyMapGenerator:
    """
    A class to generate a saliency map for an input image using a trained model. 
    """

    def __init__(self, weights_file):
        """
        Initialize the SaliencyMapGenerator.

        Args:
            weights_file (str): Path to the trained model weights file.
            input_shape (tuple): Input shape of the model.
        """
        self.weights_file = weights_file
        self.model = self._load_model()
        self.input_shape = self.model.input_shape[1:] 

    def _load_model(self):
        """
        Load the model using the provided weights file.

        Returns:
            tf.keras.Model: Loaded Keras model.
        """
        print(f"Loading model from: {self.weights_file}")
        model = ModelLoader.load_keras_model(self.weights_file)
        print("Model loaded successfully!")
        return model

    @staticmethod
    def _preprocess_image(image_path, input_shape):
        """
        Preprocess the input image.

        Args:
            image_path (str): Path to the input image.
            input_shape (tuple): Shape to resize the image to.

        Returns:
            np.ndarray: Preprocessed image array.
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_shape[1], input_shape[0]))
        if input_shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1) 
        
        image = np.ndarray.astype(image, np.uint8)
        image = np.expand_dims(image, axis=0)
        #image = tf.keras.preprocessing.image.img_to_array(image)
        #
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image

    def generate_saliency_map(self, input_image_path, output_path):
        """
        Generate and save a saliency map for the input image. Saliency map is 

        Args:
            input_image_path (str): Path to the input image.
            output_path (str): Path to save the saliency map.
        """
        img_array = self._preprocess_image(input_image_path, self.input_shape)
        input_tensor = tf.convert_to_tensor(img_array)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.model(input_tensor)
            top_class = np.argmax(predictions[0])
            top_class_pred = predictions[:, top_class]

        grads = tape.gradient(top_class_pred, input_tensor)
        normalized_grads = tf.maximum(0.0, tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.keras.backend.epsilon()))
        saliency_map = np.max(normalized_grads.numpy(), axis=-1).reshape(self.input_shape[:2])
        #saliency_map = saliency_map / saliency_map.max()
        
        # Save and display the saliency map
        plt.figure(figsize=(10, 10))
        plt.imshow(saliency_map, cmap='hot')
        plt.axis('off')
        plt.title("Saliency Map")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saliency map saved to: {output_path}")


def main():
    """
    Main function to parse arguments and generate the saliency map.
    """
    parser = argparse.ArgumentParser(description="Generate a saliency map for an input image using a trained model.")
    parser.add_argument('--weights_file', type=str, required=True, help="Path to the trained model weights file.")
    parser.add_argument('--input_image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the generated saliency map.")
    args = parser.parse_args()

    # Parse input shape

    # Initialize the SaliencyMapGenerator
    generator = SaliencyMapGenerator(weights_file=args.weights_file)

    # Generate the saliency map
    generator.generate_saliency_map(input_image_path=args.input_image, output_path=args.output)


if __name__ == "__main__":
    main()
