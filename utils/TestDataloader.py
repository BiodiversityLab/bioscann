from torch.utils.data import DataLoader,Dataset
import torch
import os
from tifffile import imread
import torchvision.transforms as transforms
import pdb
import numpy as np

class ImageTestDataset(Dataset):
    def __init__(self,input_image_folder, output_image_folder, test_feature_image_folder, resize):
        self.input_image_folder = input_image_folder
        self.output_image_folder = output_image_folder
        self.test_feature_image_folder = test_feature_image_folder
        self.resize = resize
        extensions = tuple(['.jpg','.jpeg','.png','.tif','.tiff'])
        self.input_images = []
        self.output_images = []
        self.loss_masks = []
        self.test_feature_images = []
        self.test_feature_images_strings = []
        
        
        temp_output_files = []
       # feature_names = []


        for root, dirs, filenames in os.walk(self.test_feature_image_folder):
            for f in filenames:
                if f.endswith(extensions):
                    self.test_feature_images.append(os.path.join(root,f))

        for root, dirs, filenames in os.walk(self.output_image_folder):
            for f in filenames:
                if f.endswith(extensions):
                    temp_output_files.append(f)

        for root, dirs, filenames in os.walk(self.input_image_folder):
            for f in filenames:
                if f.endswith(extensions) and f in temp_output_files:
                    begin_name = f.split('.')[0]
                   # feature_names.append(begin_name)
                    self.input_images.append(f)
                    self.output_images.append(f)
                    self.loss_masks.append(f.split('.')[0]+'_mask.tiff')
                    sample_test_feature_images_list = [os.path.join(self.test_feature_image_folder,test_feature_image.replace('\\','/').split('/')[-1]) for test_feature_image in self.test_feature_images if test_feature_image.replace('\\','/').split('/')[-1].startswith(begin_name+'-')]
                    sample_test_feature_images = ' '.join(sample_test_feature_images_list)
                    self.test_feature_images_strings.append(sample_test_feature_images)

        
        self.input_images_length = self.__len__()
        self.number_of_features = len(sample_test_feature_images_list)
        self.test_feature_names = np.unique(np.array([images_name.split('-')[-1].split('.')[0] for images_name in self.test_feature_images]))
        


    def __len__(self):
        return len(self.input_images)


    def __getitem__(self, index):
       
        image = imread(os.path.join(self.input_image_folder, self.input_images[index])).astype('uint8')

        image = self.resize(image)

        output_image = imread(os.path.join(self.output_image_folder, self.output_images[index])).astype('uint8')

        output_image = self.resize(output_image)

        loss_mask_image = imread(os.path.join(self.input_image_folder, self.loss_masks[index])).astype('uint8')
        loss_mask_image = self.resize(loss_mask_image)

        #sample_test_feature_images_path = [os.path.join(self.test_feature_image_folder,test_feature_image) for test_feature_image in self.test_feature_images[index]]
        sample_test_feature_images_path = self.test_feature_images_strings[index]
        sample = { 'image': image, 'output': output_image, 'test_feature_images': sample_test_feature_images_path, 'loss_mask': loss_mask_image, 'image_name': os.path.join(self.input_image_folder, self.input_images[index])}

        return sample



class PixelClassifierTestDataloader:

    def __init__(self, indata, outdata, testdata, image_size, batch_size=48):
        self.resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
        ])

        self.test=ImageTestDataset(indata,outdata,testdata, self.resize)

        self.dataloader = DataLoader(
            self.test, 
            batch_size=batch_size,
            shuffle=False
        )

