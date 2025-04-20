import torch
import torchvision.models as models
from core_py.utils import utils
from core_py.nets.raw_feat_reg import RawFeatRegInference
from core_py.nets.raw_vit import RawViTInferenceAttn_temporal
from core_py.datasets.pp2 import PP2Dataset
from core_py.utils.image_utils import unique_images_votes_df, transform
import plotly.graph_objects as go


# Feature extractor
# resnet50 = models.resnet50(weights='DEFAULT')
# resnet18 = models.resnet18(weights='DEFAULT')

# Weights
weight_path = "/Users/brunocerdamardini/Desktop/repo/urban_perception/weights/RawViT_100k/model_epoch_1.pth"

# Model
# model = RawFeatInference(resnet50, weight_path)
# model = RawFeatRegInference(resnet18, weight_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RawViTInferenceAttn_temporal(weight_path=weight_path, device=device, hf_model_name='google/vit-base-patch16-224-in21k')
model.to(device)
model.eval()

SIMILARITY_THRESHOLD = 1
VOTES_SAMPLE_SIZE = 2000
IMAGE_TEST_SIZE = 0.25
TRAIN_SIZE = int(VOTES_SAMPLE_SIZE * 0.75)
VALIDATION_SIZE = VOTES_SAMPLE_SIZE - TRAIN_SIZE
LOCATIONS_PATH = '/Users/brunocerdamardini/Desktop/repo/urban_perception/data/cleaned_locations.tsv'
PLACES_PATH = '/Users/brunocerdamardini/Desktop/repo/urban_perception/data/places.tsv'
IMG_PATH = '/Users/brunocerdamardini/Desktop/repo/urban_perception/data/images'
VOTES_PATH = '/Users/brunocerdamardini/Desktop/repo/urban_perception/data/cleaned_votes.tsv'

# Datasets
train_df, val_df = unique_images_votes_df(VOTES_PATH, IMAGE_TEST_SIZE, study_id='50a68a51fdc9f05596000002')
pp2_train = PP2Dataset(train_df, LOCATIONS_PATH, PLACES_PATH, IMG_PATH, TRAIN_SIZE, transform=transform())
pp2_validation = PP2Dataset(val_df, LOCATIONS_PATH, PLACES_PATH, IMG_PATH, VALIDATION_SIZE, transform=transform())


def plot_results(idx, dataset, save_path=None):
    vote = dataset[idx]


    left_image_tensor = dataset[idx][0].to(device).unsqueeze(0)
    right_image_tensor = dataset[idx][1].to(device).unsqueeze(0)

    left_score, left_attn = model.forward(left_image_tensor)
    right_score, right_attn = model.forward(right_image_tensor)
    label = dataset[idx][2]

    # utils.plot_tuple(vote, left_score.item(), right_score.item(), save_path=save_path)
    utils.plot_tuple_with_attention(vote, left_score.item(), right_score.item(), left_attn, right_attn, save_path=save_path)


# for x in range(50):
#     plot_results(x, pp2_validation, f'/Users/brunocerdamardini/Desktop/repo/urban_perception/data/reuniones/3/val_with_attention_image/comparacion_{x}.png')


def plot_hist(dataset, model, device, save_path=None):
    left_scores_arr = []
    right_scores_arr = []

    for vote in dataset:
        left_image_tensor = vote[0].to(device).unsqueeze(0)
        right_image_tensor = vote[1].to(device).unsqueeze(0)

        left_score, left_attn = model.forward(left_image_tensor)
        right_score, right_attn = model.forward(right_image_tensor)

        left_scores_arr.append(left_score.item())
        right_scores_arr.append(right_score.item())

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

# plot_hist(pp2_validation, model, device, '/Users/brunocerdamardini/Desktop/repo/urban_perception/data/reuniones/3/hist.html')