import torch
import torch.nn as nn

class RawFeat_s(nn.Module):
    def __init__(self, model, weight_path):
        super(RawFeat_s, self).__init__()
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
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))  # Change 'cpu' to 'cuda' if using GPU
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


from torchvision import models

# Load pre-trained ResNet50 (or your trained model)
resnet = models.resnet50(weights='DEFAULT')

# Path to your trained weights file
weight_path = "model_checkpoints/avance/model_epoch_40.pth"

# Initialize the modified model with pre-trained weights
model = RawFeat_s(resnet, weight_path)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Process a single image



SAMPLE_SIZE = 1000
from sklearn.model_selection import train_test_split
from torchvision import transforms
from utils import crop_from_bottom
from custom_datasets import PP2Dataset
import pandas as pd


locations_path = 'data/cleaned_locations.tsv'
places_path = 'data/places.tsv'
img_dir = 'data/images'

transform = transforms.Compose([
    transforms.Lambda(lambda img: crop_from_bottom(img, 25)),  # Custom crop
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
votes_df = (
    pd.read_csv('data/cleaned_votes.tsv', sep='\t')
    .query("study_id == '50a68a51fdc9f05596000002'")  # Filter by study_id
)
all_images = set(votes_df["left"]).union(set(votes_df["right"]))
train_images, val_images = train_test_split(list(all_images), test_size=0.25, random_state=42)
train_df = votes_df[votes_df["left"].isin(train_images) & votes_df["right"].isin(train_images)]
val_df = votes_df[votes_df["left"].isin(val_images) & votes_df["right"].isin(val_images)]

train_size = int(SAMPLE_SIZE * 0.75)
validation_size = SAMPLE_SIZE - train_size

pp2_train = PP2Dataset(train_df, locations_path, places_path, img_dir, train_size, transform=transform)
pp2_validation = PP2Dataset(val_df, locations_path, places_path, img_dir, validation_size, transform=transform)


image = pp2_train[0][0]
score = model(image.unsqueeze(0))
print(score.item())  # Get the predicted score