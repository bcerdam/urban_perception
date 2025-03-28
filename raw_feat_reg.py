import torch.nn as nn


class RawFeatReg(nn.Module):
    def __init__(self, model):
        super(RawFeatReg, self).__init__()
        self.resnet18_4f = nn.Sequential(*list(model.children())[:-1])

        for param in self.resnet18_4f.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1)
        # self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, left_images_batch, right_images_batch, left_batch_size, right_batch_size):

        left_image_features = self.resnet18_4f(left_images_batch)
        left_image_features = left_image_features.view(left_batch_size, 512)

        right_image_features = self.resnet18_4f(right_images_batch)
        right_image_features = right_image_features.view(right_batch_size, 512)

        left_image_score = self.fc1(left_image_features)
        # left_image_score = self.bn(left_image_score)
        left_image_score = self.relu(left_image_score)
        left_image_score = self.drop(left_image_score)

        right_image_score = self.fc1(right_image_features)
        # right_image_score = self.bn(right_image_score)
        right_image_score = self.relu(right_image_score)
        right_image_score = self.drop(right_image_score)

        left_image_score = self.fc2(left_image_score)
        right_image_score = self.fc2(right_image_score)

        return left_image_score, right_image_score





