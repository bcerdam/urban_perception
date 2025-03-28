import torch
import torch.nn as nn
import utils
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from pp2 import PP2Dataset


def plot_hist(scores_arr):
    plt.figure(figsize=(8, 6))
    plt.hist(scores_arr, bins=15, color='skyblue', edgecolor='black', alpha=0.7)

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Given Data")

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

class RawFeatInference(nn.Module):
    def __init__(self, model, weight_path):
        super(RawFeatInference, self).__init__()
        self.resnet50_4f = nn.Sequential(*list(model.children())[:-1])

        for param in self.resnet50_4f.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(2048, 4096)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(4096, 1)

        # Load trained weights
        self.load_weights(weight_path)

    def load_weights(self, weight_path):
        state_dict = torch.load(weight_path, map_location=torch.device('cuda'), weights_only=True)  # Change 'cpu' to 'cuda' if using GPU
        self.fc1.load_state_dict({'weight': state_dict['fc1.weight'], 'bias': state_dict['fc1.bias']})
        self.fc2.load_state_dict({'weight': state_dict['fc2.weight'], 'bias': state_dict['fc2.bias']})

    def forward(self, image):
        """ Forward pass for a single image. """
        image_features = self.resnet50_4f(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten

        image_score = self.fc1(image_features)
        image_score = self.relu(image_score)
        image_score = self.drop(image_score)
        image_score = self.fc2(image_score)

        return image_score


class RawFeatRegInference(nn.Module):
    def __init__(self, model, weight_path):
        super(RawFeatRegInference, self).__init__()
        self.resnet18_4f = nn.Sequential(*list(model.children())[:-1])

        for param in self.resnet18_4f.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        # self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

        # Load trained weights
        self.load_weights(weight_path)

    def load_weights(self, weight_path):
        state_dict = torch.load(weight_path, map_location=torch.device('cuda'), weights_only=True)  # Change 'cpu' to 'cuda' if using GPU
        self.fc1.load_state_dict({'weight': state_dict['fc1.weight'], 'bias': state_dict['fc1.bias']})
        self.fc2.load_state_dict({'weight': state_dict['fc2.weight'], 'bias': state_dict['fc2.bias']})

    def forward(self, image):

        image_features = self.resnet18_4f(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten

        image_score = self.fc1(image_features)
        # image_score = self.bn(image_score)
        image_score = self.relu(image_score)
        image_score = self.drop(image_score)
        image_score = self.fc2(image_score)

        return image_score


# Feature extractor
resnet50 = models.resnet50(weights='DEFAULT')
# resnet18 = models.resnet18(weights='DEFAULT')

# Weights
weight_path = "model_checkpoints/model_epoch_1.pth"

# Model
model = RawFeatInference(resnet50, weight_path)
# model = RawFeatRegInference(resnet18, weight_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

SIMILARITY_THRESHOLD = 0.15
VOTES_SAMPLE_SIZE = 1000
IMAGE_TEST_SIZE = 0.25
TRAIN_SIZE = int(VOTES_SAMPLE_SIZE * 0.75)
VALIDATION_SIZE = VOTES_SAMPLE_SIZE - TRAIN_SIZE
LOCATIONS_PATH = 'data/cleaned_locations.tsv'
PLACES_PATH = 'data/places.tsv'
IMG_PATH = 'data/images'
VOTES_PATH = 'data/cleaned_votes.tsv'

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
train_df, val_df = utils.unique_images_votes_df(VOTES_PATH, IMAGE_TEST_SIZE)
pp2_train = PP2Dataset(train_df, LOCATIONS_PATH, PLACES_PATH, IMG_PATH, TRAIN_SIZE, transform=utils.transform())
pp2_validation = PP2Dataset(val_df, LOCATIONS_PATH, PLACES_PATH, IMG_PATH, VALIDATION_SIZE, transform=utils.transform())

def test(size, split):

    l_correct = 0
    r_correct = 0
    t_correct = 0
    total = 0

    l_label = 0
    r_label = 0
    t_label = 0

    left_image_scores_arr = []
    right_image_scores_arr = []
    for i in range(size):
        left_image = split[i][0].to(device)
        right_image = split[i][1].to(device)
        label = split[i][2]

        left_image_score = model(left_image.unsqueeze(0))
        right_image_score = model(right_image.unsqueeze(0))

        left_image_scores_arr.append(left_image_score.item())
        right_image_scores_arr.append(right_image_score.item())

        if label == 1:
            l_label += 1
            if left_image_score > right_image_score:
                l_correct += 1
                total += 1
            else:
                total += 1
        elif label == -1:
            r_label += 1
            if left_image_score < right_image_score:
                r_correct += 1
                total += 1
            else:
                total += 1
        elif label == 0:
            t_label += 1
            if np.abs(left_image_score.item()-right_image_score.item()) < 0.5:
                t_correct += 1
                total += 1
            else:
                total += 1

        # print(f"Index: {i}, Label: {label}, Left Image Score: {left_image_score.item()}, Right Image Score: {right_image_score.item()}")

    total_l = l_label+ r_label + t_label
    print(f'Accuracy: {(l_correct+r_correct+t_correct)/total}, Left: {l_correct}/{l_label} -> {l_correct/total_l}, Right: {r_correct}/{r_label} -> {r_correct/total_l}, Tie: {t_correct}/{t_label} -> {t_correct/total_l}')
    return np.array(left_image_scores_arr), np.array(right_image_scores_arr)

# left_scores, right_scores = test(100, pp2_validation)
# left_scores, right_scores = test(100, pp2_train)
# plot_hist(left_scores)
# plot_hist(right_scores)

# t = pp2_validation[0]
#
# utils.plot_tuple(t)
#
# l_img = t[0].to(device)
# r_img = t[1].to(device)
#
# print('Left', model(l_img.unsqueeze(0)).item())
# print('Right', model(r_img.unsqueeze(0)).item())