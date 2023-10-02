import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as Func
import time
import meccano_mm as meccano
import encoder
import os
from datetime import timedelta
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, top_k_accuracy_score
import config as cfg

def center_crop_all_frames(frames, img_crop):
    crop_transform = transforms.CenterCrop(size=img_crop)
    cropped_frames = [crop_transform(frame) for frame in frames]
    return cropped_frames

def test_model(model_1, model_2, test_loader, num_classes):
    model_1.eval()
    model_2.eval()
    
    test_start = time.time()

    all_preds = []
    all_labels = []
    
    top1_correct = 0
    top5_correct = 0
    total = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs_1, inputs_2, labels, vid_path, clip_start, clip_end in test_loader:
            inputs_1 = inputs_1.cuda()
            inputs_2 = inputs_2.cuda()
            labels = labels.cuda()
            outputs_1 = model_1(inputs_1)
            outputs_2 = model_2(inputs_2)
            probs_1 = Func.softmax(outputs_1, dim=1)
            probs_2 = Func.softmax(outputs_2, dim=1)

            outputs = (probs_1 + probs_2) / 2.0

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
    wt_file_1 = cfg.test_wt_file_1
    wt_file_2 = cfg.test_wt_file_2
    modality_1 = cfg.test_modality_1
    modality_2 = cfg.test_modality_2
     
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
    test_dataset = meccano.Meccano(data_dir, 'Test', modality_1, modality_2, step_size, num_frames, test_transform)
      
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
    # Initialize the model
    model_1 = encoder.SSModel(num_classes)
    model_1 = model_1.cuda() 

    model_2 = encoder.SSModel(num_classes)
    model_2 = model_2.cuda() 

    # Loading weights
    print('==> Loading weights: ', wt_file_1)
    checkpoint_1 = torch.load(wt_file_1)
    model_1.load_state_dict(checkpoint_1['model'])

    print('==> Loading weights: ', wt_file_2)
    checkpoint_2 = torch.load(wt_file_2)
    model_2.load_state_dict(checkpoint_2['model'])

    # Evaluate the model on the test set
    test_model(model_1, model_2, test_loader, num_classes)
