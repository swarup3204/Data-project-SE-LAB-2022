from PIL import Image
import torch
import torch.nn as nn
import torchvision
from matplotlib import cm
from matplotlib.patches import Rectangle
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
#from scipy.misc import imread,imresize

# Class id to name mapping
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# Class definition for the model
class InstanceSegmentationModel(object):
    '''
        The blackbox image segmentation model (MaskRCNN).
        Given an image as numpy array (3, H, W), it generates the segmentation masks.
    '''

    # __init__ function
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    # function for calling the mask-rcnn model
    def __call__(self, input):
        '''
            Arguments:
                input (numpy array): A (3, H, W) array of numbers in [0, 1] representing the image.
            Returns:
                pred_boxes (list): list of bounding boxes, [[x1 y1 x2 y2], ..] where (x1, y1) are the coordinates of the top left corner
                                    and (x2, y2) are the coordinates of the bottom right corner.
                pred_masks (list): list of the segmentation masks for each of the objects detected.
                pred_class (list): list of predicted classes.
                pred_score (list): list of the probability (confidence) of prediction of each of the bounding boxes.
            Tip:
                You can print the outputs to get better clarity :)
        '''

        input_tensor = torch.from_numpy(input)
        input_tensor = input_tensor.type(torch.FloatTensor)
        input_tensor = input_tensor.unsqueeze(0)
        predictions = self.model(input_tensor)
        #print('hi')
        #print(predictions) #uncomment this if you want to know about the output structure.

        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in
                      list(predictions[0]['labels'].numpy())]  # Prediction classes
        pred_masks = list(predictions[0]['masks'].detach().numpy())  # Prediction masks
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in
                      list(predictions[0]['boxes'].detach().numpy())]  # Bounding boxes
        pred_score = list(predictions[0]['scores'].detach().numpy())  # Prediction scores

        return pred_boxes, pred_masks, pred_class, pred_score
"""
im1=Image.open("/home/swarup/3.jpg")
#im1.show()
im1.convert('RGB')
imArray=np.asarray(im1)
imArray1=imArray.transpose(2,0,1)
imArray2=np.divide(imArray1,255)
print(imArray2)
#im1.convert("RGB")

x=InstanceSegmentationModel()
pb,pm,pc,ps=x(imArray2)
print('ok')
print(pc)
print('ok')
print(pm)
print('ok')
print(pb)
print('ok')
print(ps)
print('hi')
#plt.imshow(imArray)
#fig,ax=plt.subplots()
#ax.imshow(np.transpose(pm[0],(1,2,0)), cmap='hot_r',alpha=1)
 # I would add interpolation='none'
#Afterwards, you can easily overlay the segmentation by doing:

#plt.imshow(np.transpose(pm[1],(1,2,0)), cmap='hot_r',alpha=0.6) # interpolation='none'
#plt.imshow(np.transpose(pm[2],(1,2,0)), cmap='hot_r',alpha=0.3)
#plt.imshow(np.transpose(pm[3],(1,2,0)), cmap='hot_r',alpha=0.2)
x=np.transpose(pm[0],(1,2,0))
im1=Image.fromarray(np.uint8(cm.gist_earth(x)*255))
#im2=Image.fromarray(pm[1])
#im3=Image.fromarray(pm[2])
im0=Image.open("./data/imgs/3.jpg")
Image.Image.paste(im0, im1)#, (50, 125))
im1.save("./data/imgs/hi.jpg")
#plt.show()
"""
"""
ax=plt.gca()
b1=pb[0]
h=b1[1][1]-b1[0][1]
w=b1[1][0]-b1[0][0]
rect = Rectangle(b1[0], h, w, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
ax.text(b1[0][0],b1[0][1], 'mouse')
plt.savefig('segmented.png')
print('bye')
"""