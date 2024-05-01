import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
from model import UNet
import scipy.io
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = UNet()
model.load_state_dict(torch.load('./save/model_epoch_20.pth'))
model.to(device)

# transform
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Resize((321, 481)),
                                transforms.Normalize((0.5), (0.5))])
#input
img = 'image.jpg'
img_name = img.split('.')[0]
image = Image.open('./test_input/' + img)
image = transform(image).unsqueeze(0).to(device)

# #label
# file = './BSDS500/ground_truth/test/196027.mat'
# data = scipy.io.loadmat(file)
# label = data['groundTruth'][0][0][0][0][1]

# inference
with torch.no_grad():
    model.eval()
    output = model(image)


output = output.squeeze().cpu().numpy()
output = np.where(output>-2.8, 255, 0)
plt.hist(output.flatten())
plt.show()
plt.imshow(output, cmap='gray')
plt.show()
# plt.imshow(label, cmap='gray')
plt.show()
plt.imsave(f'./test_output/{img_name}_output.png', output, cmap='gray')
