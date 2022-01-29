# Imports
#from PIL import Image


class RotateImage(object):

    def __init__(self, degrees):

        self.degrees=degrees

    def __call__(self, sample):

        im = sample.rotate(self.degrees)
        #im.save('X.png)
        #can be used to check like this
        return im
        # Write your code here
"""
im=Image.open("/home/swarup/PycharmProjects/pythonProject/my_package/data/imgs/0.jpg")
x=RotateImage(81)
image=x(im)
image.save("/home/swarup/result.jpg")
print('done')
"""