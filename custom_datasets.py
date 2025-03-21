import pandas as pd
import os
from utils import truncate_floats
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
pd.set_option('future.no_silent_downcasting', True)

class PP2Dataset(Dataset):
    def __init__(self, votes_path, locations_path, places_path, img_dir, transform=None):
        self.votes_df = pd.read_csv(votes_path, sep='\t').sample(n=5000, random_state=42)
        # self.votes_df = pd.read_csv(votes_path, sep='\t')
        self.locations_df = pd.read_csv(locations_path, sep='\t')
        self.places_df = pd.read_csv(places_path, sep='\t')

        valid_choices = ['left', 'equal', 'right']
        self.votes_df = self.votes_df[self.votes_df['choice'].isin(valid_choices)]

        self.votes_df['choice'] = self.votes_df['choice'].replace({'left': 1, 'right': -1, 'equal': 0})
        self.votes_df['study_id'] = self.votes_df['study_id'].replace\
            ({'50a68a51fdc9f05596000002': 'safer',
             '50f62c41a84ea7c5fdd2e454': 'livelier',
             '50f62c68a84ea7c5fdd2e456': 'more boring',
             '50f62cb7a84ea7c5fdd2e458': 'wealthier',
             '50f62ccfa84ea7c5fdd2e459': 'more depressing',
             '5217c351ad93a7d3e7b07a64': 'more beautiful'
            })

        self.votes_df = self.votes_df[self.votes_df['study_id'] == 'safer']

        self.img_labels = self.votes_df['choice']
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        left_image_id = self.votes_df.iloc[idx, 2]
        right_image_id = self.votes_df.iloc[idx, 3]

        left_place_id = self.locations_df[self.locations_df['_id'] == left_image_id]['place_id'].iloc[0]
        right_place_id = self.locations_df[self.locations_df['_id'] == right_image_id]['place_id'].iloc[0]

        left_place_name = self.places_df[self.places_df['_id'] == left_place_id]['place_name'].iloc[0].replace(" ", "")
        right_place_name = self.places_df[self.places_df['_id'] == right_place_id]['place_name'].iloc[0].replace(" ", "")

        left_loc0_coord = truncate_floats(round(self.locations_df[self.locations_df['_id'] == left_image_id]['loc.0'].iloc[0], 6))
        left_loc1_coord = truncate_floats(round(self.locations_df[self.locations_df['_id'] == left_image_id]['loc.1'].iloc[0], 6))

        right_loc0_coord = truncate_floats(round(self.locations_df[self.locations_df['_id'] == right_image_id]['loc.0'].iloc[0], 6))
        right_loc1_coord = truncate_floats(round(self.locations_df[self.locations_df['_id'] == right_image_id]['loc.1'].iloc[0], 6))

        left_image_name = f'{left_loc0_coord}_{left_loc1_coord}_{left_image_id}_{left_place_name}.JPG'
        right_image_name = f'{right_loc0_coord}_{right_loc1_coord}_{right_image_id}_{right_place_name}.JPG'

        left_img_path = os.path.join(self.img_dir, left_image_name)
        right_img_path = os.path.join(self.img_dir, right_image_name)

        left_image = read_image(left_img_path)
        right_image = read_image(right_img_path)

        label = self.img_labels.iloc[idx]
        study_question = self.votes_df.iloc[idx, 4]

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, label, study_question, left_place_name, right_place_name
        # return left_image, right_image, label
