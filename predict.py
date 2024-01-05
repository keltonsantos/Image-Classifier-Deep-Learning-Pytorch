import argparse
from torchvision import models
from torch import nn
import torch
from PIL import Image
import json
import numpy as np

# load the model checkpoint
checkpoint_path = 'checkpoint.pth'
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# process the image 
image_path = 'flowers/valid/100/image_07895.jpg' #This is an example I used to test the model
def process_image(image_path):
    image = Image.open(image_path)

    # Resize and crop the image
    size = 256
    width, height = image.size
    aspect_ratio = width / height if width > height else height / width

    if width > height:
        new_height = int(size / aspect_ratio)
        image = image.resize((size, new_height))
    else:
        new_width = int(size / aspect_ratio)
        image = image.resize((new_width, size))

    # Crop the center
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    image = image.crop((left, top, right, bottom))

    # Normalize the image
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - mean) / std

    # Transpose the color channel
    processed_image = normalized_image.transpose((2, 0, 1))

    return torch.from_numpy(processed_image).float()

# Function to predict the class for an input image
def predict(image_path, model, topk, category_names, gpu):
    # Process the image
    processed_image = process_image(image_path)

    # Add batch dimension
    processed_image = processed_image.unsqueeze(0)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Predict the class probabilities
    with torch.no_grad():
        processed_image = processed_image.to(device)
        log_ps = model(processed_image)
        ps = torch.exp(log_ps)

    # Get the top k probabilities and indices
    top_ps, top_indices = ps.topk(topk, dim=1)

    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]

    return top_ps[0].tolist(), top_classes

# Main function
def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image with a trained network.')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('checkpoint', help='Path to the model checkpoint')
    parser.add_argument('--top_k', dest='top_k', type=int, default=20, help='Return top K most likely classes')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Load the model checkpoint
    model = load_checkpoint(args.checkpoint)

    # Predict the class for the input image
    top_probs, top_classes = predict(args.image_path, model, args.top_k, args.category_names, args.gpu)

    # Print the results
    print("Top probabilities:", top_probs)
    print("Top classes:", top_classes)

if __name__ == "__main__":
    main()