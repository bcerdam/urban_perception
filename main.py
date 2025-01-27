import torch
from utils import crop_from_bottom, plot_tuple
from custom_datasets import PP2Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Transforms
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.Lambda(lambda img: crop_from_bottom(img, 25)),
])

# Dataset
pp2 = PP2Dataset('data/cleaned_votes.tsv', 'data/cleaned_locations.tsv', 'data/places.tsv', 'data/images', transform=transform)

# Visualization
test = pp2[0]
plot_tuple(test)

# Split
train, validation = torch.utils.data.random_split(pp2, [int(len(pp2)*0.75)+1, int(len(pp2)*0.25)])

# Dataloaders
train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation, batch_size=64, shuffle=True)

# Training loop
for batch_idx, batch in enumerate(train_dataloader):
    print(f'{batch_idx}/{len(train_dataloader)}')


