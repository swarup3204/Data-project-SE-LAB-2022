#Imports
from PIL import Image


class RescaleImage(object):
    '''
        Rescales the image to a given size.
    '''

    def __init__(self, output_size):
        '''
            Arguments:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        '''

        # Write your code here
        self.output_size=output_size

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)
            Returns:
            image (numpy array or PIL image)
            Note: You do not need to resize the bounding boxes. ONLY RESIZE THE IMAGE.
        '''

        # Write your code here
        if type(self.output_size)==tuple:
            return image.resize(self.output_size)
        elif type(self.output_size)==int:
            basewidth=self.output_size
            if image.shape[0] < image.shape[1]:
                wpercent=(basewidth/float(image.size[0]))
                hsize=int(float(image.size[1])*float(wpercent))
                img=image.resize((basewidth,hsize),Image.ANTIALIAS)
                return img
            else:
                wpercent=(basewidth/float(image.size[1]))
                wsize=int(float(image.size[0])*float(wpercent))
                img=image.resize((wsize,basewidth),Image.ANTIALIAS)
                return img

im=Image.open("/home/swarup/PycharmProjects/pythonProject/my_package/data/imgs/0.jpg")
x=RescaleImage((1000,260))
image=x(im)
image.save("/home/swarup/result.jpg")
print('done')
