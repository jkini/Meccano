import csv
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
            
class Meccano(Dataset):
    def __init__(self, data_dir, mode, modality, step_size, num_frames, transform):
        self.data_dir = data_dir
        self.mode = mode
        self.modality = modality
        self.transform = transform
        self.step_size = step_size
        self.num_frames = num_frames
        
        print('Constructing MECCANO {}...'.format(mode))
        self._construct_loader()

    def _construct_loader(self):
        path_to_file = os.path.join(
            self.data_dir, 'action_annotations/MECCANO_{}_actions.csv'.format(self.mode.lower())
        )
        
        self._path_to_videos = []
        self._labels = []
        self._frame_start = []
        self._frame_end = []
        with open(path_to_file, 'r') as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()[1:]):
                video_path, action_label, action_noun, frame_start, frame_end  = path_label.split(',')
                self._path_to_videos.append(video_path)
                self._frame_start.append(frame_start)
                self._frame_end.append(frame_end)
                self._labels.append(int(action_label))

    def __getitem__(self, index):        
        frames_desc = []
        frame_count = int(self._frame_start[index][:-4])         
        while(frame_count <= int(self._frame_end[index][:-4])):            
            name_frame = str(frame_count)
            if(len(name_frame) == 4): #add a prefix 0
                name_frame = '0'+name_frame
            elif(len(name_frame) == 3): #add two prefix 0
                name_frame = '00'+name_frame
            elif(len(name_frame) == 2): #add three prefix 0
                name_frame = '000'+name_frame
            elif(len(name_frame) == 1): #add four prefix 0
                name_frame = '0000'+name_frame
                
            image_path = self.data_dir+'/'+self.modality+'/'+self.mode+'/'+self._path_to_videos[index]+'/'+name_frame+'.jpg'
            frames_desc.append(image_path)
            frame_count+=1
        frame_indexes = self.temporal_sampling(int(self._frame_start[index][:-4]), int(self._frame_end[index][:-4]), self.step_size, self.num_frames)
        frames_desc = np.take(frames_desc, frame_indexes, 0)
        
        frames = [Image.open(frame_path) for frame_path in frames_desc]
        aug_frames = self.transform(frames)
        
        out_frames = torch.stack(aug_frames) 
        out_frames = out_frames.permute(1, 0, 2, 3) 

        label = self._labels[index]
        path = self._path_to_videos[index]
        start = self._frame_start[index]
        end = self._frame_end[index]
        return out_frames, label, path, start, end
    
    def __len__(self):
        return len(self._path_to_videos)

    def temporal_sampling(self, start_idx, end_idx, step_size, num_samples):      
        # Calculate the total number of integers we can fit with the given step
        max_possible_elements = (end_idx - start_idx) // step_size + 1

        # Calculate the number of times we need to repeat the last integer
        num_repeats = max(0, num_samples - max_possible_elements)

        # Select random start_idx if step_size allows, without repeating the end_idx
        if num_repeats == 0:
            start_idx = np.random.randint(start_idx , end_idx - ((num_samples - 1) * step_size) + 1)
            max_possible_elements = (end_idx - start_idx) // step_size + 1
            num_repeats = max(0, num_samples - max_possible_elements)

        # Create an array with the first 'num_samples - num_repeats' integers with the specified step
        frame_ids = np.arange(start_idx, start_idx + (num_samples - num_repeats) * step_size, step_size)

        # If num_repeats is greater than 0, append the last integer num_repeats times
        if num_repeats > 0:
            last_integer = frame_ids[-1]
            frame_ids = np.append(frame_ids, np.full(num_repeats, last_integer))

        index = frame_ids - start_idx
        return index
