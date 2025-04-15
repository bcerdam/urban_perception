import torch
from core_py.utils import image_utils
import argparse
import os
import datetime
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", module="torch._utils")
from core_py.utils.utils import clear_directory
from torchvision.models import resnet50, resnet18
from core_py.datasets.pp2 import PP2Dataset
from core_py.nets.raw_feat import RawFeat
from core_py.nets.raw_feat_reg import RawFeatReg
from torch.utils.data import DataLoader
from core_py.pipeline.train_pipeline import train_one_epoch
from core_py.pipeline.validation_pipeline import validate_model
from transformers import ViTModel
from core_py.nets.raw_vit import RawViT

execution_start_time = datetime.datetime.now()
time_based_subdir_name = execution_start_time.strftime("%d_%m_%H-%M-%S")
run_checkpoint_dir = os.path.join('model_checkpoints', time_based_subdir_name)
os.makedirs(run_checkpoint_dir, exist_ok=True)

def train_model(num_epochs, train_dataloader, pp2_train, pp2_validation, device, optimizer, model, m_w, m_t, similarity_threshold, inference_model, run_checkpoint_dir):
    for epoch_index in range(1, num_epochs + 1):
        checkpoint_path = train_one_epoch(epoch_index, train_dataloader, device, optimizer, model, m_w, m_t, similarity_threshold, pp2_train, run_checkpoint_dir)
        validate_model(checkpoint_path, pp2_validation, m_w, m_t, similarity_threshold, inference_model, run_checkpoint_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model with specified epochs")
    parser.add_argument('--model', type=str, required=False, help='RawFeat or RawFeatReg', default='RawFeat')
    parser.add_argument('--perception_attribute', type=str, required=False, help='Perception attribute to learn.', default='safer')
    parser.add_argument('--epochs', type=int, required=False, help="Number of epochs to train the model", default=5)
    parser.add_argument('--votes_sample_size', type=int, required=False, help="Number of votes samples", default=5000)
    parser.add_argument('--votes_train_size_percentage', type=float, required=False, help='% of train split on votes dataset', default=0.75)
    parser.add_argument('--image_test_size_percentage', type=float, required=False, help='% of test split on image dataset', default=0.25)
    parser.add_argument('--batch_size', type=int, required=False, help='Batch size of votes', default=64)
    parser.add_argument('--learning_rate', type=float, required=False, help='Learning rate for ADAM', default=0.001)
    parser.add_argument('--m_w', type=float, required=False, help='Hyperparameter that models the win margin', default=1)
    parser.add_argument('--m_t', type=float, required=False, help='Hyperparameter that models the tie margin', default=1)
    parser.add_argument('--similarity_threshold', type=float, required=False, help='Threshold for a tie between two scores', default=1)
    parser.add_argument('--locations_path', type=str, required=False, help="Path for the locations dataset", default='data/cleaned_locations.tsv')
    parser.add_argument('--places_path', type=str, required=False, help="Path for the places dataset", default='data/places.tsv')
    parser.add_argument('--images_path', type=str, required=False, help="Path for the images dataset", default='data/images')
    parser.add_argument('--votes_path', type=str, required=False, help="Path for the votes dataset", default='data/cleaned_votes.tsv')


    args = parser.parse_args()

    clear_directory('model_checkpoints')
    clear_directory('status')

    NUM_EPOCHS = args.epochs
    VOTES_SAMPLE_SIZE = args.votes_sample_size
    MODEL = args.model
    IMAGE_TEST_SIZE = args.image_test_size_percentage
    TRAIN_SIZE_PERCENTAGE = args.votes_train_size_percentage
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    M_W = args.m_w
    M_T = args.m_t
    SIMILARITY_THRESHOLD = args.similarity_threshold
    LOCATIONS_PATH = args.locations_path
    PLACES_PATH = args.places_path
    IMG_PATH = args.images_path
    VOTES_PATH = args.votes_path
    PERCEPTION_ATTRIBUTE = args.perception_attribute

    # NUM_EPOCHS = 40
    # VOTES_SAMPLE_SIZE = 1000
    # MODEL = 'RawFeatReg'

    # IMAGE_TEST_SIZE = 0.25
    # TRAIN_SIZE_PERCENTAGE = 0.75
    # BATCH_SIZE = 64

    # LEARNING_RATE = 0.001
    # WEIGHT_DECAY = 0.01
    # M_W = 1
    # M_T = 1
    # SIMILARITY_THRESHOLD = 1

    # LOCATIONS_PATH = 'data/cleaned_locations.tsv'
    # PLACES_PATH = 'data/places.tsv'
    # IMG_PATH = 'data/images'
    # VOTES_PATH = 'data/cleaned_votes.tsv'
    # PERCEPTION_ATTRIBUTE = 'safer'

    # Perception attributes
    pattributes_dict = ({'safer': '50a68a51fdc9f05596000002',
      'livelier': '50f62c41a84ea7c5fdd2e454',
      'more_boring': '50f62c68a84ea7c5fdd2e456',
      'wealthier': '50f62cb7a84ea7c5fdd2e458',
      'more_depressing': '50f62ccfa84ea7c5fdd2e459',
      'more_beautiful': '5217c351ad93a7d3e7b07a64'
      })

    PERCEPTION_ATTRIBUTE = pattributes_dict[PERCEPTION_ATTRIBUTE]

    # Split
    TRAIN_SIZE = int(VOTES_SAMPLE_SIZE * TRAIN_SIZE_PERCENTAGE)
    VALIDATION_SIZE = VOTES_SAMPLE_SIZE - TRAIN_SIZE

    # CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_df, val_df = image_utils.unique_images_votes_df(VOTES_PATH, IMAGE_TEST_SIZE, PERCEPTION_ATTRIBUTE)
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
        train_model(NUM_EPOCHS, train_dataloader, pp2_train, pp2_validation, device, optimizer, model, M_W, M_T, SIMILARITY_THRESHOLD, MODEL, run_checkpoint_dir)
    elif MODEL == 'RawViT':
        model = RawViT(hf_model_name='google/vit-base-patch16-224-in21k').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(NUM_EPOCHS, train_dataloader, pp2_train, pp2_validation, device, optimizer, model, M_W, M_T, SIMILARITY_THRESHOLD, MODEL, run_checkpoint_dir)
    elif MODEL == 'RawFeatReg':
        model = resnet18(weights='DEFAULT')
        model = RawFeatReg(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(NUM_EPOCHS, train_dataloader, pp2_train, pp2_validation, device, optimizer, model, M_W, M_T, SIMILARITY_THRESHOLD, MODEL)


