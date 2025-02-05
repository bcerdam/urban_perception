import torch
import utils
import argparse
import os
from torchvision.models import resnet50
from utils import crop_from_bottom, plot_tuple
from custom_datasets import PP2Dataset
from RawFeat import RawFeat
from torch.utils.data import DataLoader
from torchvision import transforms

# # CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Transforms
# transform = transforms.Compose([
#     transforms.Lambda(lambda img: crop_from_bottom(img, 25)),
#     transforms.Resize((224, 224), antialias=True)
# ])
#
# # Dataset
# pp2 = PP2Dataset('data/cleaned_votes.tsv', 'data/cleaned_locations.tsv',
#                  'data/places.tsv', 'data/images', transform=transform)
#
# # Visualization
# # test = pp2[0]
# # plot_tuple(test)
#
# # Split
# train, validation = torch.utils.data.random_split(pp2, [int(len(pp2)*0.75)+1, int(len(pp2)*0.25)])
#
# # Dataloaders
# train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
# validation_dataloader = DataLoader(validation, batch_size=64, shuffle=True)
#
# # Training loop
#
# # Model
# model = resnet50(weights='DEFAULT')
# model = RawFeat(model).to(device)
#
# # Optimizer (Temporary, paper does not specify which to use)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch(epoch_index, num_epochs, train_dataloader, device, optimizer, model):
    running_loss = 0.
    last_loss = 0.

    for batch_idx, batch in enumerate(train_dataloader):
        left_images_batch = batch[0].to(device)
        right_images_batch = batch[1].to(device)
        labels_batch = batch[2].unsqueeze(dim=1).to(device)

        optimizer.zero_grad()

        left_scores_batch = model.forward(left_images_batch, left_images_batch.shape[0])
        right_scores_batch = model.forward(right_images_batch, right_images_batch.shape[0])

        loss_batch = utils.loss(left_scores_batch, right_scores_batch, labels_batch, 1, 1, device)

        # gradients
        loss_batch.backward()
        # update weights
        optimizer.step()

        running_loss += loss_batch.item()
        last_loss = running_loss / (batch_idx + 1)  # loss per batch

        # Print loss every 1000 batches
        if batch_idx % 1000 == 0:
            avg_loss = running_loss / (batch_idx+1)  # Average loss so far
            print(f'Epoch: {epoch_index}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {avg_loss:.4f}')

    print(f'LAST EPOCH UPDATE: Epoch: {epoch_index}/{num_epochs}, Train Loss: {last_loss}')
    return last_loss

def validate_model(epoch_index, num_epochs, validation_dataloader, device, model):
    model.eval()
    total_loss = 0.
    num_batches = 0

    with torch.no_grad():
        for batch in validation_dataloader:
            left_images_batch = batch[0].to(device)
            right_images_batch = batch[1].to(device)
            labels_batch = batch[2].unsqueeze(dim=1).to(device)

            left_scores_batch = model.forward(left_images_batch, left_images_batch.shape[0])
            right_scores_batch = model.forward(right_images_batch, right_images_batch.shape[0])

            loss_batch = utils.loss(left_scores_batch, right_scores_batch, labels_batch, 1, 1, device)

            total_loss += loss_batch.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f'LAST EPOCH UPDATE: Epoch: {epoch_index}/{num_epochs}, Validation Loss: {avg_loss}')
    return avg_loss

def train_model(num_epochs, train_dataloader, validation_dataloader, device, optimizer, model):
    for epoch_index in range(1, num_epochs + 1):
        train_one_epoch(epoch_index, num_epochs, train_dataloader, device, optimizer, model)
        validate_model(epoch_index, num_epochs, validation_dataloader, device, model)

        os.makedirs('model_checkpoints', exist_ok=True)
        checkpoint_path = os.path.join('model_checkpoints', f"model_epoch_{epoch_index}.pth")
        torch.save(model.state_dict(), checkpoint_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with specified epochs")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train the model")
    args = parser.parse_args()
    num_epochs = args.epochs

    # CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_size = int(len(pp2) * 0.75)
    validation_size = len(pp2) - train_size
    train, validation = torch.utils.data.random_split(pp2, [train_size, validation_size])

    # Dataloaders
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=64, shuffle=True)

    # Training loop

    # Model
    model = resnet50(weights='DEFAULT')
    model = RawFeat(model).to(device)

    # Optimizer (Temporary, paper does not specify which to use)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(num_epochs, train_dataloader, validation_dataloader, device, optimizer, model)
