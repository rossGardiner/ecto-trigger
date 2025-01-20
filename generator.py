#This file has been created with iterative consultation from the ChatGPT LLM, version 4o

from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os
import imgaug.augmenters as iaa


def get_augmenter(input_size=(224, 224)):
    """
    Create an image augmenter using the imgaug library.

    Args:
        input_size (tuple): Desired output size of the images (height, width).

    Returns:
        iaa.Sequential: Augmentation pipeline.
    """
    augmenter = iaa.Sequential([
        iaa.Resize({"height": input_size[0], "width": input_size[1]}),
        iaa.Fliplr(0.5),  # Horizontally flip images with a liklihood ratio of 50%
    ])
    return augmenter


class CustomDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for Keras models.

    This generator deals with YOLO-format object detection data annotations and loads them into a Keras Sequence suitable for training binary (presence/absence) mobilenet_v2 models. 
    This is done by: 1) loading images and their corresponding labels, 2) resizing them to fit model input shape, 3) application of pre-processing and image augmentation.
    Preprocessing is facilitated by the mobilenet_v2 preprocess_input function, accessed through Keras and augmentation makes use of the imgaug PyPI package.
    If the input shape contains only 1 colour channel, then preprocessing loads the images as greyscale using opencv function cvtColor.
    Each Generator yields batches for training or evaluation, batch size is selected at instantation. 

    CustomDataGenerator is iterable, the method __get_item__() can be broken out for debugging purposes by setting the stop_training_flag.

    Attributes:
        data_directory (str or list): Directory or list of directories containing images and labels.
        batch_size (int): Number of samples per batch.
        input_shape (tuple): Shape of the input images (height, width, channels).
        stop_training_flag (dict): Dictionary containing a 'stop' flag for early stopping.
        shuffle (bool): Whether to shuffle the dataset at the end of each epoch.
    """

    def __init__(self, data_directory, batch_size, input_shape, stop_training_flag={"stop":False}, shuffle=True):
        """
        Initialize the data generator.

        Args:
            data_directory (str or list): Directory or list of directories with image and label files.
            batch_size (int): Number of samples per batch.
            input_shape (tuple): Shape of the input images (height, width, channels).
            stop_training_flag (dict): Dictionary with a 'stop' key to signal early stopping, a useful way to stop the generator hanging for test runs.
            shuffle (bool): Whether to shuffle data after each epoch. Defaults to True.
        """
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.input_shape = input_shape[:2]
        self.nr_channels = input_shape[2]
        self.shuffle = shuffle
        self.stop_training_flag = stop_training_flag
        self.image_paths, self.label_paths = self.load_image_and_label_paths()
        self.indexes = np.arange(len(self.image_paths))
        self.augmenter = get_augmenter(input_shape[:2])
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_image_and_label_paths(self):
        """
        Load paths to image and label files.

        Returns:
            tuple: Lists of image paths and corresponding label paths.
        """
        image_paths = []
        label_paths = []

        if isinstance(self.data_directory, list):
            for directory in self.data_directory:
                for filename in os.listdir(directory):
                    if filename.endswith('.jpg'):
                        image_paths.append(os.path.join(directory, filename))
                        root, _ = os.path.splitext(os.path.join(directory, filename))
                        label_paths.append(root + '.txt')
        else:
            for filename in os.listdir(self.data_directory):
                if filename.endswith('.jpg'):
                    image_paths.append(os.path.join(self.data_directory, filename))
                    root, _ = os.path.splitext(os.path.join(self.data_directory, filename))
                    label_paths.append(root + '.txt')

        return image_paths, label_paths

    def __len__(self):
        """
        Compute the number of batches per epoch.

        Returns:
            int: Number of batches.
        """
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def read_img(self, path):
        """
        Read and preprocess an image.

        Args:
            path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image.
        """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.nr_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)

        image = np.ndarray.astype(image, np.uint8)

        # Apply augmentation
        image = self.augmenter(image=image)

        # Preprocess for MobileNetV2
        image = preprocess_input(image)
        return image

    def __getitem__(self, index):
        """
        Generate one batch of data. This method will raise a StopIteration Exception if self.stop_training_flag is set.

        Args:
            index (int): Index of the batch.

        Returns:
            tuple: Batch of images (X) and labels (y).
        """
        if self.stop_training_flag and self.stop_training_flag['stop']:
            raise StopIteration()

        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((len(batch_indexes), *self.input_shape, self.nr_channels))
        y = np.zeros((len(batch_indexes), 1))  # Binary classification labels

        for i, batch_index in enumerate(batch_indexes):
            X[i] = self.read_img(self.image_paths[batch_index])
            y[i] = self.read_binary_label(self.label_paths[batch_index])

        return X, y

    def read_binary_label(self, label_path):
        """
        Read a binary label from a label file.

        Args:
            label_path (str): Path to the label file.

        Returns:
            float: Binary label (1.0 for presence, 0.0 for absence).
        """
        with open(label_path, 'r') as file:
            content = file.read().strip()
            return 1.0 if content else 0.0

    def on_epoch_end(self):
        """
        Shuffle the dataset at the end of each epoch, if enabled.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
