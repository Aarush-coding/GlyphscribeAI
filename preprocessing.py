'''
PREPROCESSING:
Using the PIL image processing library 

creating functions :
- preprocess_fom_canvas
- crop_image
- to tensor
- get preview
- from pil

'''
from PIL import Image, ImageOps, ImageChops
import torchvision.transforms as transforms
import torch

def to_tensor(img):
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(img)

def center_digit(img, padding = 50):
    bg = Image.new('L', img.size, 0)          
    diff = ImageChops.difference(img, bg)      
    box = diff.getbbox()                       
    
    if box:
        img = img.crop(box)
    img = crop_to_digit(img)
    longest_side = max(img.size)
    square = Image.new('L',(longest_side, longest_side), 0)
    longest_side = max(img.size) + padding 
    get_offset = ((longest_side - img.size[0])//2,
                  (longest_side - img.size[1])//2 - 22)
    square.paste(img,get_offset)
    return square

# only for images with white backgournd
def crop_to_digit(img):
    bg = Image.new(img.mode, img.size, 255)
    diff = ImageChops.difference(img, bg)
    box = diff.getbbox()
    if box:
        img = img.crop(box)
    return img

# for images with white background, crop tightly around the digit
def preprocess_from_pil(img):

    img = img.convert('L')          
    img = crop_to_digit(img)        
    img = ImageOps.invert(img)      
    img = img.resize((28, 28), Image.BILINEAR)  
    tensor = to_tensor(img)
    return tensor.unsqueeze(0)     

# When canvas implimentted, this will be used to preprocess the canvas image directly without cropping
def preprocess_from_canvas(img):
 
    img = img.convert('L')  
    img = center_digit(img)        
    img = img.resize((28, 28), Image.BILINEAR)
    tensor = to_tensor(img)
    return tensor.unsqueeze(0)

# returns the iimage that the model sees after preprocessing
def get_preview_image(img, source="photo"):

    if source == "photo":
        img = img.convert('L')
        img = crop_to_digit(img)
        img = ImageOps.invert(img)
        img = img.resize((28, 28), Image.BILINEAR)
    else:  # if using the cavas no need to crop 
        img = img.convert('L')
        img = center_digit(img)  
        img = img.resize((28, 28), Image.BILINEAR)
    return img




