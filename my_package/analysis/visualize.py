# Imports

# from my_package.model import *
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from my_package.model import InstanceSegmentationModel


def plot_visualization(image, predictions, output):  # Write the required arguments

    # The function should plot the predicted segmentation maps and the bounding boxes on the images and save them.
    # Tip: keep the dimensions of the output image less than 800 to avoid RAM crashes.
    pred_boxes, pred_masks, pred_class, pred_score = predictions

    idx1 = 0
    idx2 = 1
    idx3 = 2
    mx = 0
    for i in range(len(pred_score)):
        if pred_score[i] > mx:
            mx = pred_score[i]
            idx1 = i

    p1 = pred_masks[idx1]
    """
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            if p1[0][i][j]>0.4:
                image[0][i][j] = image[0][i][j] * 0.4+0.6
                image[1][i][j] = image[0][i][j] * 0.4
                image[2][i][j] = image[0][i][j] * 0.4
    """
    for i, e1 in enumerate(image[0]):
        for j, e2 in enumerate(image[0][i]):
            if p1[0][i][j] > 0.4:
                image[0][i][j] = image[0][i][j] * 0.4 + 0.6
                image[1][i][j] = image[0][i][j] * 0.4
                image[2][i][j] = image[0][i][j] * 0.4

    if len(pred_score) > 1:
        mx = 0
        for i in range(len(pred_score)):
            if pred_score[i] > mx and i != idx1:
                mx = pred_score[i]
                idx2 = i
        p2 = pred_masks[idx2]
        for i, e1 in enumerate(image[0]):
            for j, e2 in enumerate(image[0][i]):
                if p2[0][i][j] > 0.4:
                    image[0][i][j] = image[0][i][j] * 0.4
                    image[1][i][j] = image[0][i][j] * 0.4 + 0.6
                    image[2][i][j] = image[0][i][j] * 0.4
        if len(pred_score) > 2:
            mx = 0
            for i in range(len(pred_score)):
                if pred_score[i] > mx and (i != idx1 and i != idx2):
                    mx = pred_score[i]
                    idx3 = i
            p3 = pred_masks[idx3]
            for i, e1 in enumerate(image[0]):
                for j, e2 in enumerate(image[0][i]):
                    if p3[0][i][j] > 0.4:
                        image[0][i][j] = image[0][i][j] * 0.4
                        image[1][i][j] = image[0][i][j] * 0.4
                        image[2][i][j] = image[0][i][j] * 0.4 + 0.6
    # image1=np.transpose(image,(1,2,0))
    plt.clf()
    # plt.show()
    ax = plt.gca()
    ax.axis('off')
    figure = plt.gcf();
    w = image.shape[2]
    h = image.shape[1]
    print(w)
    print(h)
    #figure.set_size_inches(w, h)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    b = pred_boxes[idx1]
    h = b[1][1] - b[0][1]
    w = b[1][0] - b[0][0]
    rect = Rectangle(b[0], w, h, linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
    ax.plot(1, 3, "-", label=pred_class[idx1]+' '+str(pred_score[idx1]), color='green')
    # ax.legend()
    if len(pred_masks) > 1:
        b = pred_boxes[idx2]
        h = b[1][1] - b[0][1]
        w = b[1][0] - b[0][0]
        rect = Rectangle(b[0], w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.plot(1, 7, "-", label=pred_class[idx2]+' '+str(pred_score[idx2]), color='red')
        # ax.legend()
        if len(pred_masks) > 2:
            b = pred_boxes[idx3]
            h = b[1][1] - b[0][1]
            w = b[1][0] - b[0][0]
            rect = Rectangle(b[0], w, h, linewidth=2, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)
            ax.plot(1, 11, "-", label=pred_class[idx3]+' '+str(pred_score[idx3]), color='orange')
            # ax.legend()
    ax.legend()
    # plt.show()
    plt.savefig(output,dpi=100)
    # plt.imshow(np.transpose(pred_masks[idx1],(1,2,0)), cmap='hot_r',alpha=0.6) # interpolation='none'
    # plt.imshow(np.transpose(pred_masks[idx2],(1,2,0)), cmap='hot_r',alpha=0.3)
    # plt.imshow(np.transpose(pred_masks[idx3],(1,2,0)), cmap='hot_r',alpha=0.2)


output = "/home/swarup/result.png"
image = Image.open("/home/swarup/PycharmProjects/pythonProject/my_package/data/imgs/5.jpg")
arr = np.asarray(image)
arr = np.transpose(arr, (2, 0, 1))
arr = np.divide(arr, 255)
seg = InstanceSegmentationModel()
plot_visualization(arr, seg(arr), output)
print('done')
