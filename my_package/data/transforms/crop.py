from PIL import Image
from random import randrange
import math

class CropImage(object):
    def __init__(self, shape, crop_type='center'):
        # Write your code here
        self.shape = shape
        self.crop_type = crop_type

    def __call__(self, image):

        # Write your code here
        if self.crop_type == 'center':
            width, height = image.size  # Get dimensions

            left = math.floor((width - self.shape[0]) / 2.0)
            top = math.floor((height - self.shape[1]) / 2.0)
            right = math.floor((width + self.shape[0]) / 2.0)
            bottom = math.floor((height + self.shape[1]) / 2.0)

            # Crop the center of the image
            im = image.crop((left, top, right, bottom))
        elif self.crop_type == 'random':

            x,y=image.size

            x1 = randrange(0, x - self.shape[0])
            y1 = randrange(0, y - self.shape[1])
            im=image.crop((x1, y1, x1 + self.shape[0], y1 + self.shape[1]))

        return im

"""
im=Image.open("/home/swarup/PycharmProjects/pythonProject/my_package/data/imgs/0.jpg")
x=CropImage((420,260),'center')
image=x(im)
image.save("/home/swarup/result.jpg")
print('done')
"""