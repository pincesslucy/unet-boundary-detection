import os
import scipy.io
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms
import torch

class CustomDataset(Dataset):
    def __init__(self, data_dir, type, transform=None):
        self.image_dir = data_dir + 'images/' + type
        self.label_dir = data_dir + 'ground_truth/' + type
        self.transform = transform
        
        self.image_paths = [os.path.join(self.image_dir, file_name) for file_name in os.listdir(self.image_dir)]
        self.label_paths = [os.path.join(self.label_dir, file_name) for file_name in os.listdir(self.label_dir)]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path).convert("RGB")
        label_mat = scipy.io.loadmat(label_path)  
        label = label_mat['groundTruth'][0][0][0][0][1]
        #if size of label is 321x481, resize it to 481x321
        if label.shape[0] == 481:
            image = image.transpose(Image.ROTATE_90)
            label = label.transpose()
        
        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return image, label

# transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5), (0.5))])
# dataset = CustomDataset(data_dir='./BSDS500/', type='train', transform=transform)

# import matplotlib.pyplot as plt
# image, label = dataset[30]
# label = label.squeeze().numpy()
# image = image.squeeze().numpy()

# plt.subplot(1, 2, 1)
# plt.hist(label.flatten())

# plt.subplot(1, 2, 2)
# plt.hist(image.flatten())
# plt.show()

# label = np.where(label>0, 255, 0)
# label = np.uint8(label)
# label = Image.fromarray(label)
# label.show()

# image = image*0.5 + 0.5
# image = image*255
# image = np.uint8(image)
# image = Image.fromarray(image)
# image.show()