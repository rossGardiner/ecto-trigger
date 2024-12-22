#This file has been created with iterative consultation from the ChatGPT LLM, version 4o
import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from pathlib import Path
import re
import math
from sklearn.metrics import auc
import cv2

def preprocess(image, input_shape, MODEL):
    image = tf.image.resize(image, (input_shape[0], input_shape[1]))
    if input_shape[2] == 1:
        image = tf.image.rgb_to_grayscale(image)

    if MODEL == "Mobnet":
        image = tf.keras.applications.mobilenet.preprocess_input(image)
    if MODEL == "Mobnetv2":
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def open_img(filename, input_shape, MODEL):
    im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (input_shape[1], input_shape[0]))
    if input_shape[2] == 1:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = np.expand_dims(im, axis=-1)
    im = tf.keras.preprocessing.image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    if MODEL == "Mobnet":
        im = tf.keras.applications.mobilenet.preprocess_input(im)
    if MODEL == "Mobnetv2":
        im = tf.keras.applications.mobilenet_v2.preprocess_input(im)
    return im

def parse_yolo_bbox_line(line, img_width, img_height):
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    x_center_pixel = x_center * img_width
    y_center_pixel = y_center * img_height
    width_pixel = width * img_width
    height_pixel = height * img_height

    return class_id, (x_center_pixel, y_center_pixel, width_pixel, height_pixel)

def get_bounding_boxes(file_id, input_shape):
    annotation = file_id + ".txt"
    bounding_boxes = []
    if os.path.exists(annotation) and os.path.getsize(annotation) > 0:
        for line in open(annotation):
            _, bbox = parse_yolo_bbox_line(line, input_shape[1], input_shape[0])  
            bounding_boxes.append(bbox)
    return bounding_boxes

def calculate_saliency_map(model, img_array, input_shape):
    input_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        top_class = np.argmax(predictions[0])
        top_class_pred = predictions[:, top_class]

    grads = tape.gradient(top_class_pred, input_tensor)
    normalized_grads = tf.maximum(0.0, tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.keras.backend.epsilon()))
    saliency_map = np.max(normalized_grads.numpy(), axis=-1).reshape(input_shape[:2])
    saliency_map = saliency_map / saliency_map.max()
    return saliency_map

def measure_saliency_inside_bbox(saliency_map, bounding_box, thresholds, img_shape):
    x_center, y_center, width, height = bounding_box
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    bbox_saliency = saliency_map[y_min:y_max, x_min:x_max]
    results = {}
    iou_results = {}
    top_pixel_inside_bbox = False

    top_pixel = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    if x_min <= top_pixel[1] <= x_max and y_min <= top_pixel[0] <= y_max:
        top_pixel_inside_bbox = True
    
    img_area = img_shape[0] * img_shape[1]
    bbox_relative_size = math.sqrt(width * height) / math.sqrt(img_area)

    for threshold in thresholds:
        binary_saliency = (saliency_map >= threshold).astype(np.uint8)
        binary_bbox_saliency = (bbox_saliency >= threshold).astype(np.uint8)
        bbox_area = width * height
        salient_area = binary_bbox_saliency.sum()
        all_salient_pixels = binary_saliency.sum()
        if salient_area > 0 and all_salient_pixels > 0:
            results[threshold] = (salient_area / all_salient_pixels)
        else:
            results[threshold] = 0.0

        # IoU computation
        bbox_mask = np.zeros_like(saliency_map, dtype=bool)
        bbox_mask[y_min:y_max, x_min:x_max] = True
        intersection = np.logical_and(binary_saliency, bbox_mask).sum()
        union = np.logical_or(binary_saliency, bbox_mask).sum()
        iou_results[threshold] = intersection / union if union > 0 else 0

    return results, iou_results, top_pixel_inside_bbox

def process_dataset(model, data_directory, thresholds, input_shape, MODEL, output_dir, verbose):
    filenames = glob(os.path.join(data_directory, "*.jpg"))
    saliency_scores = {threshold: [] for threshold in thresholds}
    iou_scores = {threshold: [] for threshold in thresholds}
    top_pixel_inside_bbox_count = 0
    total_bboxes = 0

    for filename in filenames:
        img_array = open_img(filename, input_shape, MODEL)
        saliency_map = calculate_saliency_map(model, img_array, input_shape)
        file_id = os.path.splitext(filename)[0]
        bounding_boxes = get_bounding_boxes(file_id, input_shape)
        if len(bounding_boxes) == 0:
            continue
        for bbox in bounding_boxes:
            total_bboxes += 1
            saliency_result, iou_result, top_pixel_inside_bbox = measure_saliency_inside_bbox(saliency_map, bbox, thresholds, input_shape)
            if top_pixel_inside_bbox:
                top_pixel_inside_bbox_count += 1
            for threshold, score in saliency_result.items():
                saliency_scores[threshold].append(score)
            for threshold, iou in iou_result.items():
                iou_scores[threshold].append(iou)

        if verbose:
            plt.figure(figsize=(10, 10))
            plt.imshow(saliency_map, cmap='hot')
            cv2.imwrite(os.path.join(output_dir, f'{os.path.basename(file_id)}_saliency_map_raw.png'), saliency_map * 255)
            plt.colorbar()
            plt.title(f'Saliency Map: {os.path.basename(filename)}')
            output_file = os.path.join(output_dir, f'{os.path.basename(file_id)}_saliency_map.png')
            plt.savefig(output_file)
            plt.close()

    top_pixel_inside_bbox_proportion = top_pixel_inside_bbox_count / total_bboxes if total_bboxes > 0 else 0
    return saliency_scores, iou_scores, top_pixel_inside_bbox_proportion

def plot_results(saliency_scores, iou_scores, top_pixel_inside_bbox_proportion, output_path):
    thresholds = list(saliency_scores.keys())
    mean_saliency_scores = [np.mean(saliency_scores[threshold]) for threshold in thresholds]
    mean_iou_scores = [np.mean(iou_scores[threshold]) for threshold in thresholds]

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))

    ax[0].plot(thresholds, mean_saliency_scores)
    ax[0].set_xlabel('Threshold')
    ax[0].set_ylabel('Mean Portion of Salient Pixels Inside Bounding Box')
    ax[0].set_title('Saliency Map Alignment with Bounding Boxes')

    ax[1].plot(thresholds, mean_iou_scores)
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('Mean IoU')
    ax[1].set_title('Mean IoU Scores for Different Thresholds')

    ax[2].bar(['Top Pixel Inside BBox'], [top_pixel_inside_bbox_proportion])
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel('Proportion')
    ax[2].set_title('Proportion of Top Salient Pixel Inside Bounding Box')

    plt.savefig(output_path, format='pdf')
    plt.close()

    # Save data to CSV
    data = {
        'Threshold': thresholds,
        'Mean Saliency Score': mean_saliency_scores,
        'Mean IoU Score': mean_iou_scores,
        'Top Pixel Inside BBox Proportion': [top_pixel_inside_bbox_proportion] * len(thresholds)
    }
    df = pd.DataFrame(data)
    csv_output_path = output_path.replace('.pdf', '.csv')
    df.to_csv(csv_output_path, index=False)

    # Compute AUC for the first two plots
    auc_saliency = auc(thresholds, mean_saliency_scores)
    auc_iou = auc(thresholds, mean_iou_scores)
    
    return auc_saliency, auc_iou, top_pixel_inside_bbox_proportion

def get_weights_filenames(checkpoints_dir):
    pattern = re.compile(r'weights\.(\d+)\.hdf5$')
    weights_files = []

    for filename in os.listdir(checkpoints_dir):
        if pattern.match(filename):
            weights_files.append(filename)

    weights_files.sort(key=lambda x: int(pattern.search(x).group(1)))
    return weights_files

def main():
    parser = argparse.ArgumentParser(description='Evaluate saliency map alignment with bounding boxes.')
    parser.add_argument('--LOG_DIR', type=str, default='logs/5/', help='Directory for logging and checkpoints')
    parser.add_argument('--USE_PRETRAIN', type=bool, default=True, help="Use a pretrained path and dont bother with inat pretraining epochs", action=argparse.BooleanOptionalAction)
    parser.add_argument('--INPUT_SHAPE', type=str, default="(96, 96, 1)", help='Input shape for the model')
    parser.add_argument('--MODEL', type=str, default='Mobnetv2', help='Model type to use')
    parser.add_argument('--DATA_DIR', type=str, required=True, help='Directory of images and YOLO annotations')
    parser.add_argument('--OUTPUT_DIR', type=str, default='eval_results', help='Directory to save evaluation results')
    parser.add_argument('--VERBOSE', type=bool, default=False, help='Save saliency maps for each image', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    model_nr = args.LOG_DIR.split('/')[1]

    input_shape = tuple(map(int, args.INPUT_SHAPE.strip('()').split(',')))
    thresholds = np.linspace(0.00, 1, num=101)

    # Create output directory if it doesn't exist
    Path(args.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Get list of weights files
    weights_files = get_weights_filenames(os.path.join(args.LOG_DIR, "checkpoints"))

    auc_saliency_list = []
    auc_iou_list = []
    top_pixel_proportion_list = []
    epochs = []

    for weight_file in weights_files:
        epoch = int(re.search(r'weights\.(\d+)\.hdf5$', weight_file).group(1))
        epochs.append(epoch)
        weights_path = os.path.join(args.LOG_DIR, "checkpoints", weight_file)

        # Load model
        model = load_model(weights_path)

        # Process dataset
        saliency_scores, iou_scores, top_pixel_inside_bbox_proportion = process_dataset(model, args.DATA_DIR, thresholds, input_shape, args.MODEL, args.OUTPUT_DIR, args.VERBOSE)

        # Plot and save results
        output_path = os.path.join(f"{args.OUTPUT_DIR}/{model_nr}")
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f'saliency_alignment_{epoch}.pdf')
        auc_saliency, auc_iou, top_pixel_proportion = plot_results(saliency_scores, iou_scores, top_pixel_inside_bbox_proportion, output_path)

        # Store AUC and proportion values
        auc_saliency_list.append(auc_saliency)
        auc_iou_list.append(auc_iou)
        top_pixel_proportion_list.append(top_pixel_proportion)

    # Plot AUC and top pixel proportion over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, auc_saliency_list, label='AUC Saliency Alignment')
    plt.plot(epochs, auc_iou_list, label='AUC IoU')
    plt.plot(epochs, top_pixel_proportion_list, label='Top Pixel Inside BBox Proportion')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('AUC and Top Pixel Proportion Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(args.OUTPUT_DIR, f'{model_nr}_auc_top_pixel_over_epochs.pdf'))
    plt.close()

    # Save AUC and top pixel proportion data to CSV
    summary_data = {
        'Epoch': epochs,
        'AUC Saliency Alignment': auc_saliency_list,
        'AUC IoU': auc_iou_list,
        'Top Pixel Inside BBox Proportion': top_pixel_proportion_list
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(args.OUTPUT_DIR, f'{model_nr}_auc_top_pixel_over_epochs.csv')
    summary_df.to_csv(summary_csv_path, index=False)

if __name__ == "__main__":
    main()
