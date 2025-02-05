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

    print(f'Epoch: {epoch_index}/{num_epochs}, Train Loss: {last_loss}')
    return last_loss

def validate_model(epoch_index, num_epochs, validation_dataloader, device, model):
    model.eval()
    total_loss = 0.
    num_batches = 0

    correct_predictions = 0
    total_samples = 0
    similarity_threshold = 0.05  # Tolerance for label 0

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

            left_scores = left_scores_batch.squeeze()
            right_scores = right_scores_batch.squeeze()
            predictions = torch.zeros_like(labels_batch.squeeze())
            predictions[left_scores > right_scores] = -1
            predictions[left_scores < right_scores] = 1
            predictions[torch.abs(left_scores - right_scores) < similarity_threshold] = 0

            correct_predictions += (predictions == labels_batch.squeeze()).sum().item()
            total_samples += labels_batch.squeeze().shape[0]

    avg_loss = total_loss / num_batches
    accuracy = (correct_predictions / total_samples) * 100
    print(f'Epoch: {epoch_index}/{num_epochs}, Validation Loss: {avg_loss}, Accuracy: {accuracy:.2f}%')

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
