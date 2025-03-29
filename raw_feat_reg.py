import torch.nn as nn
import torch

class RawFeatReg(nn.Module):
    def __init__(self, model):
        super(RawFeatReg, self).__init__()
        self.resnet18_4f = nn.Sequential(*list(model.children())[:-1])

        for param in self.resnet18_4f.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512, 8)
        self.fc2 = nn.Linear(8, 1)
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, left_images_batch, right_images_batch, left_batch_size, right_batch_size):

        left_image_features = self.resnet18_4f(left_images_batch)
        left_image_features = left_image_features.view(left_batch_size, 512)

        right_image_features = self.resnet18_4f(right_images_batch)
        right_image_features = right_image_features.view(right_batch_size, 512)

        left_image_score = self.fc1(left_image_features)
        left_image_score = self.bn(left_image_score)
        left_image_score = self.relu(left_image_score)
        left_image_score = self.drop(left_image_score)

        right_image_score = self.fc1(right_image_features)
        right_image_score = self.bn(right_image_score)
        right_image_score = self.relu(right_image_score)
        right_image_score = self.drop(right_image_score)

        left_image_score = self.fc2(left_image_score)
        right_image_score = self.fc2(right_image_score)

        return left_image_score, right_image_score


class RawFeatRegInference(nn.Module):
    def __init__(self, model, weight_path):
        super(RawFeatRegInference, self).__init__()
        self.resnet18_4f = nn.Sequential(*list(model.children())[:-1])

        for param in self.resnet18_4f.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512, 8)
        self.fc2 = nn.Linear(8, 1)
        self.bn = nn.BatchNorm1d(8)
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
        image_score = self.bn(image_score)
        image_score = self.relu(image_score)
        image_score = self.drop(image_score)
        image_score = self.fc2(image_score)

        return image_score




