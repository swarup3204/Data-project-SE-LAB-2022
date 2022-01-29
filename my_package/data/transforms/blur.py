# Imports
from PIL import Image, ImageFilter


class BlurImage(object):
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        im = image.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return im


"""
im1=Image.open("/home/swarup/Desktop/ss.png")
im1.show()
#im1.convert("RGB")


x=BlurImage(3)
im2=x.__call__(im1)
im2.save("/home/swarup/Desktop/new_ss.png")
im2.show()
"""