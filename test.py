import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as Func
import time
import meccano 
import encoder
import os
from datetime import timedelta
import csv
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, top_k_accuracy_score
import config as cfg

def center_crop_all_frames(frames, img_crop):
    crop_transform = transforms.CenterCrop(size=img_crop)
    cropped_frames = [crop_transform(frame) for frame in frames]
    return cropped_frames

def test_model(model, test_loader, num_classes):
    model.eval()
    
    test_start = time.time()

    all_preds = []
    all_labels = []
    
    top1_correct = 0
    top5_correct = 0
    total = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels, vid_path, clip_start, clip_end in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            probs = Func.softmax(outputs, dim=1)

            _, top1_prediction = torch.max(probs.data, 1)
            top1_correct += (top1_prediction == labels).sum().item()

            _, top5_prediction = torch.topk(probs, k=5, dim=1)
            top5_correct += torch.any(top5_prediction == labels.view(-1, 1), dim=1).sum().item()

            total += labels.size(0)

            correct_predictions = (top1_prediction == labels)
            for i in range(len(labels)):
                class_correct[labels[i]] += correct_predictions[i].item()
                class_total[labels[i]] += 1

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(top1_prediction.cpu().numpy()) 

    top1_accuracy = 100.0 * top1_correct / total
    top5_accuracy = 100.0 * top5_correct / total
    print('++++++++++++++++++++++++++++++')
    print(f'Test Top-1 Accuracy: {top1_accuracy:.2f}%, Test Top-5 Accuracy: {top5_accuracy:.2f}%, Time: {str(timedelta(seconds=time.time() - test_start))}')
    print('++++++++++++++++++++++++++++++')

    print(f"Average Class Precision (sklearn): {(100.0 * precision_score(all_labels, all_preds, average='weighted')):.2f}%")
    print(f"Average Class Recall (sklearn): {(100.0 * recall_score(all_labels, all_preds, average='weighted')):.2f}%")
    print(f"Average F1-score (sklearn): {(100.0 * f1_score(all_labels, all_preds, average='weighted')):.2f}%")
    print(f"Top-1 Accuracy (sklearn): {(100.0 * accuracy_score(all_labels, all_preds)):.2f}%")

if __name__ == '__main__':
    data_dir = cfg.data_dir
    wt_file = cfg.test_wt_file
    modality = cfg.test_modality
    
    batch_size = cfg.test_batch_size
    num_frames = cfg.test_num_frames
    num_workers = cfg.test_num_workers

    num_classes = cfg.num_classes
    step_size = cfg.test_step_size

    img_mean = cfg.img_mean
    img_std = cfg.img_std
    img_resize = cfg.test_img_resize
    img_crop = cfg.test_img_crop

    test_transform = transforms.Compose([
        transforms.Lambda(lambda frames: [F.resize(frame, img_resize) for frame in frames]),
        transforms.Lambda(lambda frames: center_crop_all_frames(frames, img_crop)),
        transforms.Lambda(lambda frames: [F.to_tensor(frame) for frame in frames]),
        transforms.Lambda(lambda frames: [F.normalize(frame, mean = img_mean, std = img_std) for frame in frames]),
    ])
    
    # Create datasets
    test_dataset = meccano.Meccano(data_dir, 'Test', modality, step_size, num_frames, test_transform)
      
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
    # Initialize the model
    model = encoder.SSModel(num_classes)
    model = model.cuda() 

    # Loading weights
    print('==> Loading weights: ', wt_file)
    checkpoint = torch.load(wt_file)
    model.load_state_dict(checkpoint['model'])

    # Evaluate the model on the test set
    test_model(model, test_loader, num_classes)
