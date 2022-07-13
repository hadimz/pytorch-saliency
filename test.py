import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from saliency_eval import get_pretrained_saliency_fn

sal_fn = get_pretrained_saliency_fn(cuda=False)

mean=[[[[0.485]], [[0.456]], [[0.406]]]]
std=[[[0.229]], [[0.224]], [[0.225]]]
img = cv2.imread('test.jpg')
img = np.flip(img, 2)
img = np.moveaxis(img, 2, 0)
img = img - mean
img = img / std
img = torch.tensor(img)
# get the saliency map (see get_pretrained_saliency_fn doc for details)
sal_map = sal_fn(img, 203)
sal_img = sal_map.squeeze().detach().numpy()
cv2.imwrite('terrier.jpg',sal_img)

print(np.min(sal_img))
print(np.max(sal_img))