#This file has been created with iterative consultation from the ChatGPT LLM, version 4o
import os 
from tensorflow.keras.callbacks import Callback

class SaveWeightsCallback(Callback):
    def __init__(self, save_dir, save_format="tf"):
        super(SaveWeightsCallback, self).__init__()
        self.save_dir = save_dir
        self.save_format = save_format
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def on_epoch_end(self, epoch, logs=None):
        model_save_path = os.path.join(self.save_dir, f'epoch_{epoch + 1}')
        self.model.save_weights(model_save_path, save_format=self.save_format)
        # Save the model configuration separately
        model_config_path = os.path.join(self.save_dir, f'epoch_{epoch + 1}_config.json')
        with open(model_config_path, 'w') as f:
            f.write(self.model.to_json())