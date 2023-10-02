############## Dataset ############## 

# Dataset location
data_dir = '/datasets/MECCANO' 
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
num_classes = 61

############## Testing ############## 

# Testing individual modalities using test.py file
test_wt_file = '/projects/MECCANO/weights/RGB/model_best_15.pth' # '/projects/MECCANO/weights/Depth/model_best_15.pth'
test_modality = 'rgb_frames' # 'depth_frames'

# Testing with fused modalities using test_mm.py file
test_wt_file_1 = '/projects/MECCANO/weights/RGB/model_best_15.pth'
test_wt_file_2 = '/projects/MECCANO/weights/Depth/model_best_15.pth'
test_modality_1 = 'rgb_frames'
test_modality_2 = 'depth_frames'

# Parameters
test_batch_size = 16
test_num_frames = 16
test_num_workers = 8

test_step_size = 1 # skip number of frames in a clip

test_img_resize = (224) # new_width for long edge i.e. max edge, maintaining aspect ratio
test_img_crop = (224, 224) # not maintaining aspect ratio

############## Training ############## 

# Update Run ID for every train run 
train_run_id = 1 

# Trainig individual modalities using train.py file
train_modality = 'rgb_frames' # 'depth_frames'

# Location to store new weights 
train_weights_dir = '/projects/MECCANO/weights'
# Location to access SWIN-B backbone weights  
train_ss_wt_file = '/projects/MECCANO/weights/swin_base_patch244_window1677_sthv2.pth'

# Parameters
train_batch_size = 8
train_num_frames = 16
train_num_epochs = 20
train_num_workers = 8

train_step_size = 1

train_img_resize = (256) # new_width for long edge i.e. max edge, maintaining aspect ratio
train_img_crop = (224, 224) # not maintaining aspect ratio
train_img_crop_scale = (0.5, 1.0)
train_img_crop_ratio = (3/4, 4/3)

train_lr = 3e-4
train_betas=(0.9, 0.999)
train_weight_decay = 0.05

# Focal loss
train_initial_gamma = 2
train_final_gamma = 0.1

