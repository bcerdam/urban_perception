import pandas as pd
from torchvision.transforms import functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split


def crop_from_bottom(image, crop_pixels):
    image = F.to_pil_image(image)
    width, height = image.size
    image = image.crop((0, 0, width, height - crop_pixels))
    return image


def truncate_floats(value):
    value_str = str(value)
    if '.' in value_str:
        decimal_index = value_str.find('.')
        digits_after_dot = len(value_str) - decimal_index - 1
        if digits_after_dot < 6:
            value_str += '0' * (6 - digits_after_dot)
        else:
            value_str = value_str[:decimal_index + 7]
        return value_str


def transform():
    transform_img = transforms.Compose([
        transforms.Lambda(lambda img: crop_from_bottom(img, 25)),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])
    return transform_img


def unique_images_votes_df(votes_path, test_size, study_id, random_state=42):
    votes_df = (
        pd.read_csv(votes_path, sep='\t').query(f"study_id == '{study_id}'"))
    all_images = set(votes_df["left"]).union(set(votes_df["right"]))
    train_images, val_images = train_test_split(list(all_images), test_size=test_size, random_state=random_state)
    train_df = votes_df[votes_df["left"].isin(train_images) & votes_df["right"].isin(train_images)]
    val_df = votes_df[votes_df["left"].isin(val_images) & votes_df["right"].isin(val_images)]
    return train_df, val_df
