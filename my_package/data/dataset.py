# Imports
import json
import numpy as np
from PIL import Image

class Dataset(object):
    '''
        A class for the dataset that will return data items as per the given index
    '''

    def __init__(self, annotation_file, transforms=None):
        '''
            Arguments:
            annotation_file: path to the annotation file
            transforms: list of transforms (class instances)
                        For instance, [<class 'RandomCrop'>, <class 'Rotate'>]
        '''
        self.annotation_file=annotation_file
        self.transforms=transforms

    def __len__(self):
        '''
            return the number of data points in the dataset
        '''

        with open(self.annotation_file, 'r') as json_file:
            json_list = list(json_file)
        return len(json_list)

    def __getitem__(self, idx):
        '''
            return the dataset element for the index: "idx"
            Arguments:
                idx: index of the data element.
            Returns: A dictionary with:
                image: image (in the form of a numpy array) (shape: (3, H, W))
                gt_png_ann: the segmentation annotation image (in the form of a numpy array) (shape: (1, H, W))
                gt_bboxes: N X 5 array where N is the number of bounding boxes, each
                            consisting of [class, x1, y1, x2, y2]
                            x1 and x2 lie between 0 and width of the image,
                            y1 and y2 lie between 0 and height of the image.
            You need to do the following,
            1. Extract the correct annotation using the idx provided.
            2. Read the image, png segmentation and convert it into a numpy array (wont be necessary
                with some libraries). The shape of the arrays would be (3, H, W) and (1, H, W), respectively.
            3. Scale the values in the arrays to be with [0, 1].
            4. Perform the desired transformations on the image.
            5. Return the dictionary of the transformed image and annotations as specified.
        '''
        with open(self.annotation_file, 'r') as json_file:
            json_list = list(json_file)
        result=[]
        for json_str in json_list:
            result.append(json.loads(json_str))
        dict = result[idx]
        print(dict)
        img_fn = dict["img_fn"]
        img = Image.open("/home/swarup/PycharmProjects/pythonProject/my_package/data/" + img_fn)
        imgArray = np.asarray(img)
        #print(imgArray.shape)
        imgArray = np.transpose(imgArray,(2, 0, 1))
        imgArray = np.divide(imgArray, 255)
        png_ann_fn=dict["png_ann_fn"]
        pngArray = np.asarray(png_ann_fn)
        pngArray=np.transpose(pngArray,(2,0,1))
        pngArray=np.divide(pngArray,255)
        bboxes=dict["bboxes"]


        return imgArray


