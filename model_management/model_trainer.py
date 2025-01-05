import argparse
import tensorflow as tf
import os
from model_loader import ModelLoader
from generator import CustomDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

class ModelTrainer:
    """
    A class to handle the training of a model using a custom data generator.

    Attributes:
        model (tf.keras.Model): The model to be trained.
        train_generator (CustomDataGenerator): Data generator for training data.
        val_generator (CustomDataGenerator): Data generator for validation data.
    """

    def __init__(self, config):
        """
        Initialize the ModelTrainer.

        Args:
            config (dict): Configuration dictionary with training parameters.
        """
        self.config = config
        self.model = self._load_model()
        self.train_generator = self._create_data_generator(config['train_data_dir'])
        self.val_generator = self._create_data_generator(config['val_data_dir'], shuffle=False)

    def _load_model(self):
        """
        Load the model using ModelLoader.

        Returns:
            tf.keras.Model: Compiled model.
        """
        if self.config['use_pretrained_weights']:
            print("Error: Pretrained weights cannot be loaded for training. Use a fresh model.")
            exit(1)

        model = ModelLoader.create_model(
            input_shape=self.config['input_shape'],
            alpha=self.config['alpha']        )
        print("Model created successfully!")
        return model

    def _create_data_generator(self, data_directory, shuffle=True):
        """
        Create an instance of CustomDataGenerator.

        Args:
            data_directory (str or list): Directory containing the dataset.
            shuffle (bool): Whether to shuffle data. Defaults to True.

        Returns:
            CustomDataGenerator: Initialized data generator.
        """
        print(f"Creating data generator for directory: {data_directory}")
        return CustomDataGenerator(
            data_directory=data_directory,
            batch_size=self.config['batch_size'],
            input_shape=self.config['input_shape'],
            stop_training_flag={'stop': False},
            shuffle=shuffle
        )

    def train(self):
        """
        Train the model using the data generators.
        """
        callbacks=[ModelCheckpoint(
                filepath=os.path.join(self.config["log_dir"], 'checkpoints/weights.{epoch:02d}-{loss:.3f}.hdf5'),
                save_weights_only=False,
                monitor='loss', 
                mode='min',  
                save_freq=500,
                verbose=1,
                save_best_only=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.config["log_dir"], 'checkpoints/weights.{epoch:02d}.hdf5'),
                save_weights_only=False,
                monitor='loss',  
                mode='min',  
                save_freq="epoch",
                verbose=1,
                save_best_only=False
            ),
            TensorBoard(
                log_dir=self.config["log_dir"],
                write_graph=False,
                write_images=True,
                update_freq=500,  # update every n batches
                profile_batch=0,  # Disable profiling
                embeddings_freq=0  # Disable embedding visualization
            )]

        print("Starting training...")
        history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=self.config['epochs'],
            callbacks=callbacks
        )
        print("Training complete!")
        return history


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a model using a custom data generator.")
    parser.add_argument('--train_data_dir', type=str, required=True, help="Path to the training data directory.")
    parser.add_argument('--val_data_dir', type=str, required=True, help="Path to the validation data directory.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--input_shape', type=str, default="(96, 96, 3)", help="Input shape for the model (e.g., '(96, 96, 3)').")
    parser.add_argument('--alpha', type=float, default=0.2, help="Alpha parameter for MobileNet variants.")
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory for saving logs and checkpoints.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument('--use_pretrained_weights', action='store_true', help="Attempt to use pretrained weights (Not supported for training).")
    return parser.parse_args()


def main():
    """
    Main function for training a model.
    """
    args = parse_args()

    # Convert input shape from string to tuple
    input_shape = tuple(map(int, args.input_shape.strip('()').split(',')))

    config = {
        'train_data_dir': args.train_data_dir,
        'val_data_dir': args.val_data_dir,
        'batch_size': args.batch_size,
        'input_shape': input_shape,
        'alpha': args.alpha,
        'log_dir': args.log_dir,
        'epochs': args.epochs,
        'use_pretrained_weights': args.use_pretrained_weights
    }

    # Initialize and train the model
    trainer = ModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
