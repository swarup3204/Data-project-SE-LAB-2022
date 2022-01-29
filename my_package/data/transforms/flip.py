# Imports
from PIL import Image


class FlipImage(object):
    def __init__(self, flip_type='horizontal'):
        # Write your code here
        self.flip_type = flip_type

    def __call__(self, image):
        # Write your code here
        if self.flip_type == 'horizontal':
            im = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.flip_type == 'vertical':
            im = image.transpose(Image.FLIP_TOP_BOTTOM)

        return im

"""
im1=Image.open("/home/swarup/Desktop/ss.png")
im1.show()
#im1.convert("RGB")


x=FlipImage('vertical')
im2=x.__call__(im1)
im2.save("/home/swarup/Desktop/new_ss.png")
im2.show()
"""