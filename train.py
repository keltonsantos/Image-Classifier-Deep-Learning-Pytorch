import argparse
from torchvision import models
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import json
import torch

# load and preprocess the data

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define data transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets with ImageFolder
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = ImageFolder(valid_dir, transform=val_test_transforms)
    test_dataset = ImageFolder(test_dir, transform=val_test_transforms)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=32,
                                               shuffle = False)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle = False)

    # Extract class to index mapping
    class_to_idx = train_dataset.class_to_idx

    return train_loader, valid_loader, test_loader, class_to_idx

    # build and train the model
def build_and_train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Load and preprocess the data
    train_loader, valid_loader, _, class_to_idx = load_data(data_dir)

    # Use the specified architecture or default to VGG16
    if arch == 'VGG':
        model = models.vgg16(pretrained=True)
        input_size = 25088  # Update this with the correct input size for VGG16
    elif arch == 'Densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024  # Update this with the correct input size for Densenet121
    else:
        raise ValueError("Invalid architecture. Choose 'VGG' and 'Densenet'.")

    # Freeze parameters to avoid backpropagation through them
    for param in model.parameters():
        param.requires_grad = False

    # Define the classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, len(class_to_idx)),
        nn.LogSoftmax(dim=1)
    )

    # Replace the model classifier with the new one
    model.classifier = classifier

    # Specify the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)

    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate the model
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                valid_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

    # Save the model checkpoint
    checkpoint = {
        'arch': arch,
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'classifier': classifier,
        'epochs': epochs
    }
    torch.save(checkpoint, save_dir)

# Main function
def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir', help='Set directory with training data')
    parser.add_argument('--save_dir', dest='save_dir', default='checkpoint.pth', help='Set directory to save checkpoints')
    parser.add_argument('--arch', dest='arch', default='vgg16', help='Choose architecture (default: vgg16)')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01, help='Set learning rate')
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=256, help='Set number of hidden units')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='Set number of epochs')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    build_and_train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

if __name__ == "__main__":
    main()