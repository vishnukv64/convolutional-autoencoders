# link to the dataset = https://drive.google.com/u/0/uc?id=1VT-8w1rTT2GCE5IE5zFJPMzv7bqca-Ri&export=download

import torch
import torch.nn as nn
from torch.optim import Adam
from model import VGGEncoder, Decoder
from torch.utils.data import DataLoader
from dataset import CatKingdom
from utils import train, predict, find_similar_images
import torchvision.transforms as transforms
import glob
from sklearn.model_selection import train_test_split

# Hyper parameters and configs


if __name__ == "__main__":
    root_images_path = "dataset/*"
    num_epochs = 20
    image_size = 256
    batch_size = 32
    lr = 0.01
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    images = glob.glob(root_images_path)
    train_images, test_images = train_test_split(images, test_size=0.1, random_state=42)

    transformation = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std=std),
        ]
    )

    # Defining the Training and testing dataset and dataloaders

    train_dataset = CatKingdom(train_images, transform=transformation)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_dataset = CatKingdom(test_images, transform=transformation)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Initializing the decoder and the encoder with initializing to CUDA

    encoder = VGGEncoder()
    decoder = Decoder()

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print("CUDA is not available.  Training on CPU ...")
        device = "cpu"
    else:
        print("CUDA is available!  Training on GPU ...")
        device = "cuda"

    if train_on_gpu:
        encoder.cuda()
        decoder.cuda()

    # defining the loss and the optimizer

    criterion = nn.MSELoss()
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = Adam(autoencoder_params, lr=lr)

    # train
    loss = train(
        train_dataloader, batch_size, num_epochs, criterion, optimizer, encoder, decoder
    )
    print(loss)

    # prediction
    predict(encoder, decoder, test_dataloader, device)

    # similiarity
    for i in [4, 8, 12, 15, 20, 25]:  # random images in the batch
        find_similar_images(i, encoder, test_dataloader, device)
