import torch.nn as nn


class RawFeat(nn.Module):

    def __init__(self, model):
        super(RawFeat, self).__init__()
        self.resnet50_4f = nn.Sequential(*list(model.children())[:-1])
        # self.resnet50_4f = nn.Sequential(*list(model.children())[:-4])

        for param in self.resnet50_4f.parameters():
            param.requires_grad = False

        # self.fc1 = nn.Linear(2048, 4096)
        self.fc1 = nn.Linear(512, 64)
        # self.fc2 = nn.Linear(4096, 1)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        # self.drop = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.75)

    def forward(self, image, batch_size):

        image_features = self.resnet50_4f(image)
        # image_features = image_features.view(batch_size, 2048)
        image_features = image_features.view(batch_size, 512)

        x = self.fc1(image_features)
        x = self.relu(x)
        x = self.drop(x)

        score = self.fc2(x)

        return score
