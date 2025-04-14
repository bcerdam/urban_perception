import torch
import image_utils
import argparse
from utils import clear_directory
from torchvision.models import resnet50, resnet18
from pp2 import PP2Dataset
from raw_feat import RawFeat
from raw_feat_reg import RawFeatReg
from torch.utils.data import DataLoader
from train import train_one_epoch
from validation import validate_model


def train_model(num_epochs, train_dataloader, pp2_train, pp2_validation, device, optimizer, model, m_w, m_t, similarity_threshold, inference_model):
    for epoch_index in range(1, num_epochs + 1):
        checkpoint_path = train_one_epoch(epoch_index, train_dataloader, device, optimizer, model, m_w, m_t, similarity_threshold, pp2_train)
        validate_model(checkpoint_path, pp2_validation, m_w, m_t, similarity_threshold, inference_model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model with specified epochs")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train the model")
    parser.add_argument('--votes_sample_size', type=int, required=True, help="Number of votes samples")
    parser.add_argument('--model', type=str, required=True, help='RawFeat or RawFeatReg')
    args = parser.parse_args()

    clear_directory('model_checkpoints')
    clear_directory('status')

    NUM_EPOCHS = args.epochs
    VOTES_SAMPLE_SIZE = args.votes_sample_size
    MODEL = args.model

    # NUM_EPOCHS = 40
    # VOTES_SAMPLE_SIZE = 1000
    # MODEL = 'RawFeatReg'

    IMAGE_TEST_SIZE = 0.25
    TRAIN_SIZE = int(VOTES_SAMPLE_SIZE * 0.75)
    VALIDATION_SIZE = VOTES_SAMPLE_SIZE - TRAIN_SIZE
    BATCH_SIZE = 64

    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    M_W = 1
    M_T = 1
    SIMILARITY_THRESHOLD = 1

    LOCATIONS_PATH = 'data/cleaned_locations.tsv'
    PLACES_PATH = 'data/places.tsv'
    IMG_PATH = 'data/images'
    VOTES_PATH = 'data/cleaned_votes.tsv'

    # CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_df, val_df = image_utils.unique_images_votes_df(VOTES_PATH, IMAGE_TEST_SIZE)
    pp2_train = PP2Dataset(train_df, LOCATIONS_PATH, PLACES_PATH, IMG_PATH, TRAIN_SIZE, transform=image_utils.transform())
    pp2_validation = PP2Dataset(val_df, LOCATIONS_PATH, PLACES_PATH, IMG_PATH, VALIDATION_SIZE,
                                transform=image_utils.transform())

    # Dataloaders
    train_dataloader = DataLoader(pp2_train, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(pp2_validation, batch_size=BATCH_SIZE, shuffle=True)

    # model (feature extractor + FC)
    if MODEL == 'RawFeat':
        model = resnet50(weights='DEFAULT')
        model = RawFeat(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(NUM_EPOCHS, train_dataloader, pp2_train, pp2_validation, device, optimizer, model, M_W, M_T, SIMILARITY_THRESHOLD, MODEL)
    elif MODEL == 'RawFeatReg':
        model = resnet18(weights='DEFAULT')
        model = RawFeatReg(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(NUM_EPOCHS, train_dataloader, pp2_train, pp2_validation, device, optimizer, model, M_W, M_T, SIMILARITY_THRESHOLD, MODEL)


