import torch
import utils
import tqdm
import argparse
from torchvision.models import resnet50
from utils import crop_from_bottom, plot_tuple
from custom_datasets import PP2Dataset
from RawFeat import RawFeat
from torch.utils.data import DataLoader
from torchvision import transforms

# Transforms
transform = transforms.Compose([
    transforms.Lambda(lambda img: crop_from_bottom(img, 25)),
    transforms.Resize((224, 224), antialias=True)
])

# Dataset
pp2 = PP2Dataset('data/cleaned_votes.tsv', 'data/cleaned_locations.tsv',
                 'data/places.tsv', 'data/images', transform=transform)

# Visualization
# test = pp2[0]
# plot_tuple(test)

# Split
train, validation = torch.utils.data.random_split(pp2, [int(len(pp2)*0.75)+1, int(len(pp2)*0.25)])

# Dataloaders
train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation, batch_size=64, shuffle=True)

# Training loop

# Model
model = resnet50(weights='DEFAULT')
model = RawFeat(model)

# Optimizer (Temporary, paper does not specify which to use)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch(epoch_index, num_epochs):
    running_loss = 0.
    last_loss = 0.

    with tqdm.tqdm(total=len(train_dataloader), desc=f"Epoch {epoch_index}/{num_epochs}") as pbar:
        for batch_idx, batch in enumerate(train_dataloader):
            left_images_batch = batch[0]
            right_images_batch = batch[1]
            labels_batch = batch[2].unsqueeze(dim=1)

            optimizer.zero_grad()

            left_scores_batch = model.forward(left_images_batch, 64)
            right_scores_batch = model.forward(right_images_batch, 64)

            loss_batch = utils.loss(left_scores_batch, right_scores_batch, labels_batch, 1, 1)

            # gradients
            loss_batch.backward()
            # update weights
            optimizer.step()

            running_loss += loss_batch.item()
            last_loss = running_loss / (batch_idx + 1)  # loss per batch

            pbar.set_postfix(loss=last_loss)
            pbar.update(1)

        print(f'Epoch: {epoch_index}/{num_epochs}, Loss: {last_loss}')
    return last_loss


def train_model(num_epochs):
    for epoch_index in range(1, num_epochs + 1):
        train_one_epoch(epoch_index, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with specified epochs")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train the model")
    args = parser.parse_args()
    num_epochs = args.epochs
    train_model(num_epochs)