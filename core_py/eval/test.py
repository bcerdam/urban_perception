import torch
import torchvision.models as models
from core_py.utils import utils
from core_py.nets.raw_feat_reg import RawFeatRegInference
from core_py.datasets.pp2 import PP2Dataset
from core_py.utils.image_utils import unique_images_votes_df, transform
import plotly.graph_objects as go


# Feature extractor
# resnet50 = models.resnet50(weights='DEFAULT')
resnet18 = models.resnet18(weights='DEFAULT')

# Weights
weight_path = "data/model_epoch_2.pth"

# Model
# model = RawFeatInference(resnet50, weight_path)
model = RawFeatRegInference(resnet18, weight_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

SIMILARITY_THRESHOLD = 1
VOTES_SAMPLE_SIZE = 1000
IMAGE_TEST_SIZE = 0.25
TRAIN_SIZE = int(VOTES_SAMPLE_SIZE * 0.75)
VALIDATION_SIZE = VOTES_SAMPLE_SIZE - TRAIN_SIZE
LOCATIONS_PATH = '../../data/cleaned_locations.tsv'
PLACES_PATH = '../../data/places.tsv'
IMG_PATH = '../../data/images'
VOTES_PATH = '../../data/cleaned_votes.tsv'

# Datasets
train_df, val_df = unique_images_votes_df(VOTES_PATH, IMAGE_TEST_SIZE)
pp2_train = PP2Dataset(train_df, LOCATIONS_PATH, PLACES_PATH, IMG_PATH, TRAIN_SIZE, transform=transform())
pp2_validation = PP2Dataset(val_df, LOCATIONS_PATH, PLACES_PATH, IMG_PATH, VALIDATION_SIZE, transform=transform())


def plot_results(idx, dataset, save_path=None):
    vote = dataset[idx]


    left_image_tensor = dataset[idx][0].to(device).unsqueeze(0)
    right_image_tensor = dataset[idx][1].to(device).unsqueeze(0)

    left_score = model.forward(left_image_tensor).item()
    right_score = model.forward(right_image_tensor).item()
    label = dataset[idx][2]

    utils.plot_tuple(vote, left_score, right_score, save_path=save_path)


# for x in range(50):
#     plot_results(x, pp2_train, f'data/Reuniones/2/comparaciones_train/comparacion_{x}.png')


def plot_hist(dataset, model, device, save_path=None):
    left_scores_arr = []
    right_scores_arr = []

    for vote in dataset:
        left_image_tensor = vote[0].to(device).unsqueeze(0)
        right_image_tensor = vote[1].to(device).unsqueeze(0)

        left_score = model.forward(left_image_tensor).item()
        right_score = model.forward(right_image_tensor).item()

        left_scores_arr.append(left_score)
        right_scores_arr.append(right_score)

    min_score = min(min(left_scores_arr), min(right_scores_arr))
    max_score = max(max(left_scores_arr), max(right_scores_arr))

    left_hist = go.Histogram(
        x=left_scores_arr,
        nbinsx=30,
        marker=dict(color='blue', line=dict(color='black', width=1)),
        name="Left Scores",
        opacity=0.7
    )

    right_hist = go.Histogram(
        x=right_scores_arr,
        nbinsx=30,
        marker=dict(color='red', line=dict(color='black', width=1)),
        name="Right Scores",
        opacity=0.7
    )

    fig = go.Figure(data=[left_hist, right_hist])

    fig.update_layout(
        title="Histograms of Left and Right Scores",
        xaxis_title="Score",
        yaxis_title="Frequency",
        barmode='overlay',
        xaxis=dict(range=[min_score, max_score]),
        template='plotly_dark',
        showlegend=True
    )

    fig.show()

    if save_path:
        fig.write_html(save_path)