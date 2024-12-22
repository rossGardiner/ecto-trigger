from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os
import imgaug.augmenters as iaa


def get_augmenter(input_size=(224, 224)):
    augmenter = iaa.Sequential([
        iaa.Resize({"height": input_size[0], "width": input_size[1]}),
        iaa.Fliplr(0.5)#,  # Horizontally flip 50% of the images
        # iaa.Affine(
        #     rotate=(-30, 30),  # Rotate images by -30 to +30 degrees
        #     scale=(0.8, 1.2)  # Scale images to 80-120% of their size
        # ),
        #iaa.Crop(percent=(0, 0.1))  # Crop images by 0-10% of their height/width
    ])
    return augmenter


class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, data_directory, batch_size, input_shape, stop_training_flag, shuffle=True):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.input_shape = input_shape[:2]
        self.nr_channels = input_shape[2] 
        self.shuffle = shuffle
        self.image_paths, self.label_paths = self.load_image_and_label_paths()
        self.indexes = np.arange(len(self.image_paths))
        self.augmenter = get_augmenter(input_shape[:2])
        self.stop_training_flag = stop_training_flag
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_image_and_label_paths(self):
        image_paths = []
        label_paths = []
        if isinstance(self.data_directory, list):
            for directory in self.data_directory:
                for filename in os.listdir(directory):
                    if filename.endswith('.jpg'):
                        image_paths.append(os.path.join(directory, filename))
                        root, ext = os.path.splitext(os.path.join(directory, filename))
                        label_paths.append(os.path.join(directory, root + '.txt'))
        else:
            for filename in os.listdir(self.data_directory):
                if filename.endswith('.jpg'):
                    image_paths.append(os.path.join(self.data_directory, filename))
                    root, ext = os.path.splitext(os.path.join(self.data_directory, filename))
                    label_paths.append(os.path.join(self.data_directory, root + '.txt'))
        return image_paths, label_paths

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def read_img(self, path):
        arg = cv2.IMREAD_COLOR
        image = cv2.imread(path, arg)
        # if image.ndim == 2:  # If image is (w, h), add a channel dimension
        #     image = np.expand_dims(image, axis=-1)
        # if image.shape[-1] == 1:  # If image is grayscale (w, h, 1), convert to (w, h, 3)
        #     image = np.repeat(image, 3, axis=-1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.nr_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1) 
            
        # convert to uint8
        image = np.ndarray.astype(image, np.uint8)

        # Augment the image
        image = self.augmenter(image=image)
        image = preprocess_input(image)  # Scale to MobileNet input range [-1.0, 1.0]
        return image


    def __getitem__(self, index):
        if self.stop_training_flag and self.stop_training_flag['stop']:
            raise StopIteration()
        
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((len(batch_indexes), *self.input_shape, self.nr_channels))
        y = np.zeros((len(batch_indexes), 1))  # Binary classification labels

        for i, batch_index in enumerate(batch_indexes):
            image = self.read_img(self.image_paths[batch_index])
            
            # Assign image to batch
            X[i] = image
            
            # Read and process label for binary classification
            label = self.read_binary_label(self.label_paths[batch_index])
            y[i] = label

        return X, y

    def read_binary_label(self, label_path):
        # Assuming the presence of a bounding box indicates the presence of an instance
        with open(label_path, 'r') as file:
            content = file.read().strip()
            if content:  # If there's content, label is 1 (instance present)
                return 1.0
            else:  # If no content, label is 0 (no instance)
                return 0.0

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
