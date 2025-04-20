import torch.nn as nn
import torch
from transformers import ViTModel


class RawViT(nn.Module):
    def __init__(self, hf_model_name='google/vit-base-patch16-224-in21k'):
        super(RawViT, self).__init__()

        self.vit = ViTModel.from_pretrained(hf_model_name)

        for param in self.vit.parameters():
            param.requires_grad = False

        vit_output_dim = self.vit.config.hidden_size
        self.fc1 = nn.Linear(vit_output_dim, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, left_images_batch, right_images_batch, left_batch_size, right_batch_size):

        left_outputs = self.vit(pixel_values=left_images_batch)
        left_image_features = left_outputs.last_hidden_state[:, 0, :]

        right_outputs = self.vit(pixel_values=right_images_batch)
        right_image_features = right_outputs.last_hidden_state[:, 0, :]

        left_image_score = self.fc1(left_image_features)
        left_image_score = self.relu(left_image_score)
        left_image_score = self.drop(left_image_score)
        left_image_score = self.fc2(left_image_score)

        right_image_score = self.fc1(right_image_features)
        right_image_score = self.relu(right_image_score)
        right_image_score = self.drop(right_image_score)
        right_image_score = self.fc2(right_image_score)

        return left_image_score, right_image_score


class RawViTInference(nn.Module):
    def __init__(self, weight_path, device, hf_model_name='google/vit-base-patch16-224-in21k'):
        super(RawViTInference, self).__init__()

        self.vit = ViTModel.from_pretrained(hf_model_name)
        self.vit.to(device)

        for param in self.vit.parameters():
            param.requires_grad = False

        vit_output_dim = self.vit.config.hidden_size

        self.fc1 = nn.Linear(vit_output_dim, 4096)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(4096, 1)

        self.load_weights(weight_path, device)


    def load_weights(self, weight_path, device):

        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        self.fc1.load_state_dict({
            'weight': state_dict['fc1.weight'],
            'bias': state_dict['fc1.bias']
        })

        self.fc2.load_state_dict({
            'weight': state_dict['fc2.weight'],
            'bias': state_dict['fc2.bias']
        })


    def forward(self, image):

        outputs = self.vit(pixel_values=image)
        image_features = outputs.last_hidden_state[:, 0, :]

        image_score = self.fc1(image_features)
        image_score = self.relu(image_score)
        image_score = self.drop(image_score)
        image_score = self.fc2(image_score)

        return image_score


class RawViTInferenceAttn_temporal(nn.Module):
    def __init__(self, weight_path, device, hf_model_name='google/vit-base-patch16-224-in21k'):
        super(RawViTInferenceAttn_temporal, self).__init__()

        self.vit = ViTModel.from_pretrained(hf_model_name)
        self.vit.to(device)

        for param in self.vit.parameters():
            param.requires_grad = False

        vit_output_dim = self.vit.config.hidden_size
        self.fc1 = nn.Linear(vit_output_dim, 4096)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(4096, 1)

        self.load_weights(weight_path, device)


    def load_weights(self, weight_path, device):
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        self.fc1.load_state_dict({
            'weight': state_dict['fc1.weight'],
            'bias': state_dict['fc1.bias']
        })
        self.fc2.load_state_dict({
            'weight': state_dict['fc2.weight'],
            'bias': state_dict['fc2.bias']
        })
    def forward(self, image, output_attentions=True):
        outputs = self.vit(
            pixel_values=image,
            output_attentions=output_attentions
        )

        image_features = outputs.last_hidden_state[:, 0, :]

        image_score = self.fc1(image_features)
        image_score = self.relu(image_score)
        image_score = self.drop(image_score)
        image_score = self.fc2(image_score)

        return image_score, outputs.attentions
