from tqdm import tqdm
import torch
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import numpy as np


def train(
    dataloader, bs, epochs, criterion, optimizer, encoder, decoder, train_on_gpu=True
):
    n_epochs = epochs
    train_loss = None

    for epoch in range(1, n_epochs + 1):
        # keep track of training and validation loss
        train_loss = 0.0
        ###################
        # train the model #
        ###################
        # model by default is set to train
        # model.train()

        for batch_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = None
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                images = data.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            encoder_op = encoder(images)
            decoder_op = decoder(encoder_op)
            # calculate the batch loss
            loss = criterion(decoder_op, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * images.size(0)

        training_loss = train_loss / len(dataloader.sampler)
        print(" Epoch %d, loss: %.16f" % (epoch, training_loss))

    torch.save(encoder.state_dict(), "encoder.pt")
    torch.save(decoder.state_dict(), "decoder.pt")

    return train_loss


def predict(encoder, decoder, test_dataloader, device):
    encoder.eval()  # eval mode
    decoder.eval()  # eval mode
    dataiter = iter(test_dataloader)
    images = dataiter.next()
    images = images.to(device)  # since model is in cuda, we need to load images to cuda
    output = encoder(images)
    output = decoder(output)

    # prep images for display
    images = images.cpu().numpy()  # bring back again from cuda to cpu to plot

    # use detach when it's an output that requires_grad
    output = output.cpu()  # need to load from cuda to cpu
    output = output.detach().numpy()  # and then detach it

    n = 10
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # originals
        ax = plt.subplot(2, n, i + 1)
        ax.axis("off")
        ax.set_title("Original")
        out = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * out + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)

        # outputs
        ax = plt.subplot(2, n, i + 1 + n)
        ax.axis("off")
        ax.set_title("outputs")
        out = np.transpose(output[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * out + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)


# Function to find similar images to the one supplied by idx
def find_similar_images(idx, model, test_dataloader, device):
    n = 10
    model.eval()
    dataiter = iter(test_dataloader)
    images = dataiter.next()
    images = images.to(device)  # since model is in cuda, we need to load images to cuda

    # this code block takes one predicted output and compares with the other images to find similarity
    output_idx = model(images[idx].unsqueeze(0))  # output of the index passed
    output_idx = output_idx.cpu()  # need to load from cuda to cpu
    output_idx = output_idx.detach().numpy()  # and then detach it
    output_idx = output_idx.reshape(output_idx.shape[0], -1)

    # takes all the predicted outputs of the images
    output = model(images)  # total output of images

    images_original = (
        images.cpu().numpy()
    )  # preserving the original images before reshaping to 1D
    images = images.cpu().numpy()  # bring back again from cuda to cpu to plot

    # Converting to 1D for finding cosine distance

    output = output.cpu()  # need to load from cuda to cpu
    output = (
        output.detach().numpy()
    )  # use detach when it's an output that requires_grad

    # convert images and outputs to 1D vector
    output = output.reshape(output.shape[0], -1)
    images = images.reshape(images.shape[0], -1)

    # calculate the cosine similarity from our reference image to all the other image codes
    similarities = [
        (cosine(output_idx[0], output[i][0]), i) for i in range(len(output))
    ]
    similarities.sort()
    similar_idxs = [i for _, i in similarities]

    # plot top 10 matches
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(similar_idxs[:n]):
        ax = plt.subplot(2, n, i + 1)
        ax.axis("off")
        if i == 0:
            ax.set_title("Original")
        else:
            ax.set_title(i)

        out = np.transpose(images_original[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * out + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)


def imshow(dataloader):
    fig = plt.figure(figsize=(24, 16))
    fig.tight_layout()
    images = next(iter(dataloader))

    for num, sample in enumerate(images[:10]):
        # classes = [sample[x] for x in classes]
        plt.subplot(4, 6, num + 1)
        plt.axis("off")
        sample = sample.cpu().numpy()
        out = np.transpose(sample, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * out + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
