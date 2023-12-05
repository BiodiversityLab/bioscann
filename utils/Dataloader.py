import os
from torch.utils.data import DataLoader,Dataset
import random
from tifffile import imread
import cv2
import csv

import math
import torchvision.transforms as transforms
import torch


class ImageDataset(Dataset):
    def __init__(self,input_image_folder, output_image_folder, transform, resize, cutmix=False, loss_mask=False):
        self.input_image_folder = input_image_folder
        self.output_image_folder = output_image_folder
        self.transform = transform
        self.resize = resize
        extensions = tuple(['.jpg','.jpeg','.png','.tif','.tiff'])
        self.input_images = []
        self.output_images = []
        self.loss_masks = []
        self.loss_mask = loss_mask
        
        temp_output_files = []
        for root, dirs, filenames in os.walk(self.output_image_folder):
            for f in filenames:
                if f.endswith(extensions):
                    temp_output_files.append(f)

        for root, dirs, filenames in os.walk(self.input_image_folder):
            for f in filenames:
                if f.endswith(extensions):
                    if loss_mask and os.path.isfile(os.path.join(root,f.split('.')[0]+'_mask.tiff')) and f in temp_output_files:
                        self.input_images.append(f)
                        self.output_images.append(f)
                        self.loss_masks.append(f.split('.')[0]+'_mask.tiff')
                    elif not loss_mask and f in temp_output_files:
                        self.input_images.append(f)
                        self.output_images.append(f)
        
        self.input_images_length = self.__len__()
        if cutmix:
            self.cutmix = True
            self.width, self.height = self.__init_cut_mix()
        else:
            self.cutmix = False
        


    def __init_cut_mix(self):
        image = imread(os.path.join(self.input_image_folder, self.input_images[0])).astype('uint8')
        width = len(image[1])
        height = len(image[0])
        return width, height

        

    def __get_cut_mix_image(self,cuts, index, image_folder, images):
   
        crop_h = int(self.height/cuts)
        crop_w = int(self.width/cuts)
        new_image = imread(os.path.join(image_folder, images[index])).astype('uint8')
        multi_dim = False
        if len(new_image.shape) > 2:
            multi_dim = True

        i = 1
        for x in range(cuts):
            for y in range(cuts):
                image2 = imread(os.path.join(image_folder, images[(index+i)%self.input_images_length])).astype('uint8')

                #every even image should be cropped at random position
                if i % 2 == 0:
                    offset_x = random.randint(0,self.width-self.width/cuts)
                    offset_y = random.randint(0,self.height-self.height/cuts)
                else:
                    offset_x = int((self.width/cuts)*x)
                    offset_y = int((self.height/cuts)*y)

                if multi_dim:
                    new_image[offset_y:offset_y+crop_h,offset_x:offset_x+crop_w,:] = image2[int(self.height/2-(self.height/cuts)/2):int(self.height/2+(self.height/cuts)/2),int(self.width/2-(self.width/cuts)/2):int(self.width/2+(self.width/cuts)/2),:]
                else:
                    new_image[offset_y:offset_y+crop_h,offset_x:offset_x+crop_w] = image2[int(self.height/2-(self.height/cuts)/2):int(self.height/2+(self.height/cuts)/2),int(self.width/2-(self.width/cuts)/2):int(self.width/2+(self.width/cuts)/2)]
                i += 1
        return new_image


    def __len__(self):
        return len(self.input_images)


    def __getitem__(self, index):
        seed = random.randint(1,999)
       # image = cv2.imread(os.path.join(self.input_image_folder, self.input_images[index]))
        #image = imread(os.path.join(self.input_image_folder, self.input_images[index])).astype('uint8')

        if self.cutmix:
            image = self.__get_cut_mix_image(cuts=4, index=index, image_folder=self.input_image_folder, images=self.input_images)
        else:
            image = imread(os.path.join(self.input_image_folder, self.input_images[index])).astype('uint8')
            #if len(image.shape) > 2:
            #    image = image.transpose(1,2,0)


        torch.manual_seed(seed)
        image = self.transform(image)
       # output_image = cv2.imread(os.path.join(self.output_image_folder, self.output_images[index]), cv2.IMREAD_UNCHANGED)
       # output_image = imread(os.path.join(self.output_image_folder, self.output_images[index])).astype('uint8')
        if self.cutmix:
            output_image = self.__get_cut_mix_image(cuts=4, index=index, image_folder=self.output_image_folder, images=self.output_images)
        else:
            output_image = imread(os.path.join(self.output_image_folder, self.output_images[index])).astype('uint8')
        torch.manual_seed(seed)
        output_image = self.resize(output_image)

        if self.loss_mask:
            loss_mask_image = imread(os.path.join(self.input_image_folder, self.loss_masks[index])).astype('uint8')
            torch.manual_seed(seed)
            loss_mask_image = self.resize(loss_mask_image)

        if self.loss_mask:
            sample = { 'image': image, 'output': output_image, 'loss_mask': loss_mask_image, 'image_name': os.path.join(self.input_image_folder, self.input_images[index])}
        elif not self.loss_mask:
            sample = { 'image': image, 'output': output_image, 'image_name': os.path.join(self.input_image_folder, self.input_images[index])}
        return sample


class PixelClassifierDataloader:

    def __init__(self, indata, outdata, image_size, cutmix=False, batch_size=48, val_indata='',val_outdata='', loss_mask=False):
        self.train_transform = transforms.Compose([
                        #transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Resize(image_size),
                        #transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.RandomVerticalFlip(p=0.5),
                        #transforms.RandomRotation(degrees=180),
                        #transforms.RandomAffine(degrees=180,translate=(0.0,0.1)),
                        #transforms.ColorJitter(brightness=(0.8,1.2)),
                        #transforms.ToTensor(),
                        #transforms.Resize((64 ,64)),
                        ])

        self.resize = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(image_size),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(degrees=180),
            #transforms.RandomAffine(degrees=180,translate=(0.1,0.4)),
            #transforms.ToTensor(),
            #transforms.Resize((64 ,64)),
        ])
        if val_indata=='' and val_outdata=='':
            self.train_dataset=ImageDataset(indata,outdata,self.train_transform, self.resize,cutmix=cutmix, loss_mask=loss_mask)
            val_split, train_split = int(math.ceil(self.train_dataset.__len__()*0.2)), int(math.floor(self.train_dataset.__len__()*0.8))
            val, train = torch.utils.data.random_split(self.train_dataset,[val_split,train_split])
        else:
            train=ImageDataset(indata,outdata,self.train_transform, self.resize,cutmix=cutmix, loss_mask=loss_mask)
            val=ImageDataset(val_indata,val_outdata,self.train_transform, self.resize,cutmix=cutmix, loss_mask=loss_mask)

        self.train_dataloader = DataLoader(
            train, 
            batch_size=batch_size,
            shuffle=True
        )

        self.val_dataloader = DataLoader(
            val, 
            batch_size=batch_size,
            shuffle=True
        )
