import torch
import utils
import argparse
import os
from torchvision.models import resnet50, resnet18
from utils import crop_from_bottom, plot_tuple
from custom_datasets import PP2Dataset
from RawFeat import RawFeat
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd

def train_one_epoch(epoch_index, num_epochs, train_dataloader, device, optimizer, model):
    running_loss = 0.
    last_loss = 0.
    similarity_threshold = 0.15  # Tolerance for label 0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_dataloader):
        left_images_batch = batch[0].to(device)
        right_images_batch = batch[1].to(device)
        labels_batch = batch[2].unsqueeze(dim=1).to(device)

        optimizer.zero_grad()

        left_scores_batch = model.forward(left_images_batch, left_images_batch.shape[0])
        right_scores_batch = model.forward(right_images_batch, right_images_batch.shape[0])

        loss_batch = utils.loss(left_scores_batch, right_scores_batch, labels_batch, 1, 0.15, device)

        # gradients
        loss_batch.backward()
        # update weights
        optimizer.step()

        running_loss += loss_batch.item()
        last_loss = running_loss / (batch_idx + 1)  # loss per batch

        left_scores = left_scores_batch.squeeze()
        right_scores = right_scores_batch.squeeze()
        predictions = torch.zeros_like(labels_batch.squeeze())
        predictions[left_scores > right_scores] = 1
        predictions[left_scores < right_scores] = -1
        predictions[torch.abs(left_scores - right_scores) < similarity_threshold] = 0

        correct_predictions += (predictions == labels_batch.squeeze()).sum().item()
        total_samples += labels_batch.squeeze().shape[0]

    accuracy = (correct_predictions / total_samples) * 100
    print(f'Epoch: {epoch_index}/{num_epochs}, Train Loss: {last_loss}, Train Accuracy: {accuracy:.2f}%')
    return last_loss

def validate_model(epoch_index, num_epochs, validation_dataloader, device, model):
    model.eval()
    running_loss = 0.

    correct_predictions = 0
    total_samples = 0
    similarity_threshold = 0.15  # Tolerance for label 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dataloader):
            left_images_batch = batch[0].to(device)
            right_images_batch = batch[1].to(device)
            labels_batch = batch[2].unsqueeze(dim=1).to(device)

            left_scores_batch = model.forward(left_images_batch, left_images_batch.shape[0])
            right_scores_batch = model.forward(right_images_batch, right_images_batch.shape[0])

            loss_batch = utils.loss(left_scores_batch, right_scores_batch, labels_batch, 1, 0.15, device)

            running_loss += loss_batch.item()
            last_loss = running_loss / (batch_idx + 1)

            left_scores = left_scores_batch.squeeze()
            right_scores = right_scores_batch.squeeze()
            predictions = torch.zeros_like(labels_batch.squeeze())

            predictions[left_scores > right_scores] = 1
            predictions[left_scores < right_scores] = -1
            predictions[torch.abs(left_scores - right_scores) < similarity_threshold] = 0

            correct_predictions += (predictions == labels_batch.squeeze()).sum().item()
            total_samples += labels_batch.squeeze().shape[0]

    accuracy = (correct_predictions / total_samples) * 100
    print(f'Epoch: {epoch_index}/{num_epochs}, Validation Loss: {last_loss}, Validation Accuracy: {accuracy:.2f}%')

    return last_loss

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
    NUM_EPOCHS = args.epochs

    # hp
    SAMPLE_SIZE = 25000
    locations_path = 'data/cleaned_locations.tsv'
    places_path = 'data/places.tsv'
    img_dir = 'data/images'

    # CUDA
    transform = transforms.Compose([
        transforms.Lambda(lambda img: crop_from_bottom(img, 25)),
        transforms.Resize((224, 224), antialias=True)
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    votes_df = (
        pd.read_csv('data/cleaned_votes.tsv', sep='\t')
        .query("study_id == '50a68a51fdc9f05596000002'")  # Filter by study_id
    )
    all_images = set(votes_df["left"]).union(set(votes_df["right"]))
    train_images, val_images = train_test_split(list(all_images), test_size=0.25, random_state=42)
    train_df = votes_df[votes_df["left"].isin(train_images) & votes_df["right"].isin(train_images)]
    val_df = votes_df[votes_df["left"].isin(val_images) & votes_df["right"].isin(val_images)]

    train_size = int(SAMPLE_SIZE * 0.75)
    validation_size = SAMPLE_SIZE - train_size

    pp2_train = PP2Dataset(train_df, locations_path, places_path, img_dir, train_size, transform=transform)
    pp2_validation = PP2Dataset(val_df, locations_path, places_path, img_dir, validation_size, transform=transform)

    # Dataloaders
    train_dataloader = DataLoader(pp2_train, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(pp2_validation, batch_size=64, shuffle=True)

    # Feature extractor
    # model = resnet50(weights='DEFAULT')
    # model = RawFeat(model).to(device)
    model = resnet18(weights='DEFAULT')
    model = RawFeat(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(NUM_EPOCHS, train_dataloader, validation_dataloader, device, optimizer, model)

'''
Local
'''

# NUM_EPOCHS = 1
# SAMPLE_SIZE = 100
# locations_path = 'data/cleaned_locations.tsv'
# places_path = 'data/places.tsv'
# img_dir = 'data/images'
#
#
# transform = transforms.Compose([
#         transforms.Lambda(lambda img: crop_from_bottom(img, 25)),
#         transforms.Resize((224, 224), antialias=True)
#     ])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# votes_df = (
#             pd.read_csv('data/cleaned_votes.tsv', sep='\t')
#             .query("study_id == '50a68a51fdc9f05596000002'")  # Filter by study_id
#             )
# all_images = set(votes_df["left"]).union(set(votes_df["right"]))
# train_images, val_images = train_test_split(list(all_images), test_size=0.25, random_state=42)
# train_df = votes_df[votes_df["left"].isin(train_images) & votes_df["right"].isin(train_images)]
# val_df = votes_df[votes_df["left"].isin(val_images) & votes_df["right"].isin(val_images)]
#
#
# train_size = int(SAMPLE_SIZE * 0.75)
# validation_size = SAMPLE_SIZE - train_size
#
# pp2_train = PP2Dataset(train_df, locations_path, places_path, img_dir, train_size, transform=transform)
# pp2_validation = PP2Dataset(val_df, locations_path, places_path, img_dir, validation_size, transform=transform)
#
# train_dataloader = DataLoader(pp2_train, batch_size=64, shuffle=True)
# validation_dataloader = DataLoader(pp2_validation, batch_size=64, shuffle=True)
#
# model = resnet18(weights='DEFAULT')
# model = RawFeat(model).to(device)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train_model(NUM_EPOCHS, train_dataloader, validation_dataloader, device, optimizer, model)


