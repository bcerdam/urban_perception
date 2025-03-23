import torch.nn as nn


class RawFeat(nn.Module):

    def __init__(self, model):
        super(RawFeat, self).__init__()
        self.resnet50_4f = nn.Sequential(*list(model.children())[:-1])

        for param in self.resnet50_4f.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(2048, 4096)
        # self.fc1 = nn.Linear(512, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(4096, 1)
        # self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        # self.drop = nn.Dropout(0.5)

    # def forward(self, image, batch_size):
    def forward(self, left_images_batch, right_images_batch, left_batch_size, right_batch_size):

        '''
        Sequential Pass
        '''

        # image_features = self.resnet50_4f(image)
        # image_features = image_features.view(batch_size, 2048)
        # # image_features = image_features.view(batch_size, 512)
        #
        # x = self.fc1(image_features)
        # # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        #
        # score = self.fc2(x)
        #
        # return score

        '''
        Parallel Pass
        '''

        left_image_features = self.resnet50_4f(left_images_batch)
        left_image_features = left_image_features.view(left_batch_size, 2048)

        right_image_features = self.resnet50_4f(right_images_batch)
        right_image_features = right_image_features.view(right_batch_size, 2048)

        left_image_score = self.fc1(left_image_features)
        left_image_score = self.relu(left_image_score)
        left_image_score = self.drop(left_image_score)
        left_image_score = self.fc2(left_image_score)

        right_image_score = self.fc1(right_image_features)
        right_image_score = self.relu(right_image_score)
        right_image_score = self.drop(right_image_score)
        right_image_score = self.fc2(right_image_score)

        return left_image_score, right_image_score





