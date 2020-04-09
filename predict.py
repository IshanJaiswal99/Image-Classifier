import numpy as np
import torch
import PIL
import argparse
import json
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from math import ceil


def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)

    model = models.vgg16(pretrained=True)
    for param in model.parameters(): param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    scheduler = scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.1,last_epoch=-1)
    return model, optimizer, scheduler


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = PIL.Image.open(image)
    image.thumbnail([256,256], PIL.Image.LANCZOS)
    
    width, height = image.size 
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    image = image.crop((left,top,right,bottom))
    
    np_image = np.array(image) / 255
    
    np_image = ((np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2,0,1)

    return np_image

def predict(image_path, model, topk,cuda,cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0)
    if cuda:
        device =   torch.device("cuda:0")
        image = image.cuda().float()
    else:
        device =  torch.device("cpu")
    #image = image.to(device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        #output = torch.nn.functional.softmax(output, dim=1)
        prob, idxs = torch.topk(output, topk,)
        probs = prob[0].tolist()
        idxs = np.array(idxs)
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in idxs[0]]
        
        # map the class name with collected topk classes
        names = []
        for cls in classes:
            names.append(cat_to_name[str(cls)])
        
    print_result(probs, names)
    
    
def print_result(probs, flowers):
    
    #print("Best Likly Flower is: {}".format(flowers[0]))
    #print(probs)
    for i in range (len(flowers)):
        print("Rank {}:".format(i+1),
            "Flower: {} , Probability: {}".format(flowers[i], probs[i]))


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name')
    parser.add_argument('input', type=str, help='Image path')
    parser.add_argument('checkpoint', type=str, help='Models checkpoint for inference')
    parser.add_argument('--top_k', type=int, help='Return top k most likely classes. Default value 5')
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names from a json file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')

    # Parse and read arguments and assign them to variables if exists 
    args, _ = parser.parse_known_args()

    image_path = args.input
    checkpoint = args.checkpoint

    top_k = 5
    if args.top_k:
        top_k = args.top_k

    category_names = 'cat_to_name.json'
    if args.category_names:
        category_names = args.category_names

    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("Warning! GPU flag was set however no GPU is available in the machine")
    
    
    with open(category_names, 'r') as f:
        	cat_to_name = json.load(f)
            
    model_loaded, optimizer_loaded, scheduler_loaded = load_checkpoint(checkpoint)
    #image = process_image()
    predict(image_path, model_loaded, top_k,cuda,cat_to_name)
    

            
if __name__ == '__main__':
    main()
          
