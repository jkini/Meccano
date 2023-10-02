import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn as nn
import torch.nn.functional as Func
import time
import meccano 
import encoder
import os
from datetime import timedelta
from collections import OrderedDict
from focal_loss.focal_loss import FocalLoss
import config as cfg

#F.resize sets shortest edge the assigned value until and unless specified
def resize_long_edge(frames, img_resize):
    width, height = frames[0].size
    
    if width > height:
        new_width = img_resize
        new_height = int(img_resize * height / width)
    else:
        new_width = int(img_resize * width / height)
        new_height = img_resize
        
    resized_frames = [F.resize(frame, (new_height, new_width)) for frame in frames]
    return resized_frames

def random_resized_crop_all_frames(frames, img_crop):
    crop_transform = transforms.RandomResizedCrop(size=img_crop, scale=img_crop_scale, ratio=img_crop_ratio)
    crop_params = crop_transform.get_params(frames[0], scale=img_crop_scale, ratio=img_crop_ratio)
    cropped_frames = [F.resized_crop(frame, *crop_params, size=img_crop) for frame in frames]
    return cropped_frames

def center_crop_all_frames(frames, img_crop):
    crop_transform = transforms.CenterCrop(size=img_crop)
    cropped_frames = [crop_transform(frame) for frame in frames]
    return cropped_frames

def linear_annealing(current_epoch, total_epochs, initial_gamma, final_gamma):
    return initial_gamma + (final_gamma - initial_gamma) * (current_epoch / total_epochs)

def exponential_annealing(current_epoch, total_epochs, initial_gamma, final_gamma):
    return initial_gamma * (final_gamma / initial_gamma) ** (current_epoch / total_epochs)

def train_model(model, train_loader, val_loader, initial_gamma, final_gamma, criterion, optimizer, scheduler, num_epochs, weights_dir, run_id):
    model.train()
    
    train_start = time.time()
    
    best_score = 0
    
    for epoch in range(num_epochs):
        epoch_train_start = time.time()
        running_loss = 0.0
        current_gamma = exponential_annealing(epoch, num_epochs, initial_gamma, final_gamma)
        criterion.gamma = current_gamma
        print('criterion.gamma: ', criterion.gamma)
        for inputs, labels, vid_path, clip_start, clip_end in train_loader:
            batch_start = time.time()
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            probs = Func.softmax(outputs, dim=1)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Time : {str(timedelta(seconds=time.time() - epoch_train_start))}, Training Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        running_loss_val = 0.0
        epoch_val_start = time.time()
        with torch.no_grad():
            for inputs_val, labels_val, vid_path_val, clip_start_val, clip_end_val in val_loader:
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                outputs_val = model(inputs_val)
                _, predicted_val = torch.max(outputs_val.data, 1)
                probs_val = Func.softmax(outputs_val, dim=1)
                loss_val = criterion(probs_val, labels_val)
                
                running_loss_val += loss_val.item() * inputs_val.size(0)

                total += labels_val.size(0)
                correct += (predicted_val == labels_val).sum().item()

        val_loss = running_loss_val / len(val_loader.dataset)
        accuracy_val = 100.0 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Time: {str(timedelta(seconds=time.time() - epoch_val_start))}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy_val:.2f}%')
    
        if accuracy_val > best_score:
            print('++++++++++++++++++++++++++++++')
            print(f'Epoch {epoch + 1} is the best model till now for {run_id}!')
            print('++++++++++++++++++++++++++++++')
            state = {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }            
            save_file = os.path.join(weights_dir, '{}/model_best_{}.pth'.format(run_id, epoch + 1))
            torch.save(state, save_file)
            best_score = accuracy_val
            
        elif (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs:
            print('==> Saving...')
            state = {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }            
            save_file = os.path.join(weights_dir, '{}/model_{}.pth'.format(run_id, epoch + 1))
            torch.save(state, save_file)
     
    print(f'Finished Training ---> Time: {str(timedelta(seconds=time.time() - train_start))}')

if __name__ == '__main__':
    data_dir = cfg.data_dir
    weights_dir = cfg.train_weights_dir
    ss_wt_file = cfg.train_ss_wt_file
    modality = cfg.train_modality

    print('++++++++++++++++++++++++++++++')
    run_id = cfg.train_run_id
    print(f'Run ID: {run_id}') 
    print('++++++++++++++++++++++++++++++')

    batch_size = cfg.train_batch_size
    num_frames = cfg.train_num_frames
    num_epochs = cfg.train_num_epochs
    num_workers = cfg.train_num_workers

    num_classes = cfg.num_classes
    step_size = cfg.train_step_size

    img_mean = cfg.img_mean
    img_std = cfg.img_std
    img_resize = cfg.train_img_resize
    img_crop = cfg.train_img_crop
    img_crop_scale = cfg.train_img_crop_scale
    img_crop_ratio = cfg.train_img_crop_ratio

    lr = cfg.train_lr
    betas = cfg.train_betas
    weight_decay = cfg.train_weight_decay

    # Focal loss
    initial_gamma = cfg.train_initial_gamma
    final_gamma = cfg.train_final_gamma
    
    # Resize image long edge to 256 -> aspect ratio same
    # Scale, then crop randomly maintaining aspect ratio, then resize to 224 (causes blurring)
    train_transform = transforms.Compose([
        transforms.Lambda(lambda frames: resize_long_edge(frames, img_resize)),
        transforms.Lambda(lambda frames: random_resized_crop_all_frames(frames, img_crop)),
        transforms.Lambda(lambda frames: [F.to_tensor(frame) for frame in frames]),
        transforms.Lambda(lambda frames: [F.normalize(frame, mean = img_mean, std = img_std) for frame in frames]),
    ])

    # Resize image short edge to 256-> aspect ratio same
    # Centercrop to 224
    val_transform = transforms.Compose([
        transforms.Lambda(lambda frames: [F.resize(frame, img_resize) for frame in frames]),
        transforms.Lambda(lambda frames: center_crop_all_frames(frames, img_crop)),
        transforms.Lambda(lambda frames: [F.to_tensor(frame) for frame in frames]),
        transforms.Lambda(lambda frames: [F.normalize(frame, mean = img_mean, std = img_std) for frame in frames]),
    ])

    # Create datasets
    train_dataset = meccano.Meccano(data_dir, 'Train', modality, step_size, num_frames, train_transform)
    val_dataset = meccano.Meccano(data_dir, 'Val', modality, step_size, num_frames, val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

    # Initialize the model, loss function, and optimizer)
    model = encoder.SSModel(num_classes)
    model = model.cuda() 
    criterion = FocalLoss(initial_gamma)
    criterion = criterion.cuda()
    optimizer = optim.AdamW(model.parameters(), lr, betas, weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs)

    print('==> Loading SomethingSomthing weights..')
    checkpoint = torch.load(ss_wt_file)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'backbone' in k:
            name = 'swin3d_b_ss.' + k[9:]
            new_state_dict[name] = v 
    model.load_state_dict(new_state_dict, strict = False)

    # Train the model
    train_model(model, train_loader, val_loader, initial_gamma, final_gamma, criterion, optimizer, scheduler, num_epochs, weights_dir, run_id)