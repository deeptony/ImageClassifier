from PIL import Image
from math import floor
import numpy as np
import torch 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #opening image
    pil_image = Image.open(image)
    #resizing shortest side to 256, keeping aspect ratio
    w, h = pil_image.size
    size = 256, 256
    if w > h:
        ratio = float(w) / float(h)
        newheight = ratio * size[0]
        pil_image.resize((size[0], int(floor(newheight))), Image.ANTIALIAS)
    else:
        ratio = float(h) / float(w)
        newwidth = ratio * size[1]
        print(newwidth)
        pil_image.resize((int(floor(newwidth), size[1])), Image.ANTIALIAS)
  
    # center crop the image
    x = 224
    cropped_image = pil_image.crop((w//2 - x//2, h//2 - x//2, w//2 + x//2, h//2 + x//2))
    print(cropped_image.size)
    
    #convert image to numpy array. Divide by 255 as is the largest value (to obtain floats 0-1)
    np_image = np.array(cropped_image) / 255
    print(np_image.shape)

    #normalizing image
    mu = np.array([0.485, 0.456, 0.406])
    sigma = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - mu) / sigma
    #reorder dimensions by transposing
    transposed = normalized_image.transpose((2, 0, 1))
    #convert to tensor
    processed_image = torch.from_numpy(transposed)
    print(processed_image.shape)
    return processed_image.type(torch.DoubleTensor)