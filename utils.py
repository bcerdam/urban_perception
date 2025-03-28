import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split


def truncate_floats(value):
    value_str = str(value)
    if '.' in value_str:
        # Find the decimal point and check the number of digits after it
        decimal_index = value_str.find('.')
        digits_after_dot = len(value_str) - decimal_index - 1
        if digits_after_dot < 6:
            # Add zeros to make it exactly 6 digits after the dot
            value_str += '0' * (6 - digits_after_dot)
        else:
            # Truncate to 6 digits after the dot
            value_str = value_str[:decimal_index + 7]
        return value_str


def plot_tuple(data):
    """
    Plots a tuple containing two images, an integer, and a string.

    Parameters:
        data (tuple): A tuple of size 4.
                      - The first two elements are torch.Tensor images (C, H, W).
                      - The third element is an integer (1, 0, or -1).
                      - The fourth element is a string.
    """
    # Unpack the tuple
    image1, image2, label, title_text, left_place_name, right_place_name = data

    # Map integer label to preferred side
    label_map = {1: "Left", 0: "Equal", -1: "Right"}
    preference = label_map.get(label, "Unknown")

    # Convert torch.Tensor images to numpy for plotting
    def tensor_to_image(tensor):
        if tensor.ndim == 3:  # (C, H, W)
            return tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        return tensor.numpy()  # Assume already in (H, W)

    img1 = tensor_to_image(image1)
    img2 = tensor_to_image(image2)

    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    # Plot the first image
    ax[0].imshow(img1, cmap='gray' if img1.ndim == 2 else None)
    ax[0].axis('off')
    ax[0].set_title(left_place_name)

    # Plot the second image
    ax[1].imshow(img2, cmap='gray' if img2.ndim == 2 else None)
    ax[1].axis('off')
    ax[1].set_title(right_place_name)

    # Set the overall title
    plt.suptitle(f"Comparison: {title_text}, {preference} was selected", fontsize=16)
    plt.tight_layout()
    plt.show()


def crop_from_bottom(image, crop_pixels):
    image = F.to_pil_image(image)
    width, height = image.size  # Get the image dimensions
    image = image.crop((0, 0, width, height - crop_pixels))
    return image


def loss(left_images_batch_scores, right_images_batch_scores, labels_batch, m_w, m_t, device):
    labels_batch = labels_batch.float()  # Ensure it's float for gradients
    m_w = float(m_w)
    m_t = float(m_t)

    diff = left_images_batch_scores - right_images_batch_scores
    diff_adj_w = (-1 * labels_batch*diff + m_w)
    diff_win_w = diff_adj_w * torch.abs(labels_batch)

    diff_adj_t = torch.abs(diff) - m_t
    diff_tie_t = diff_adj_t * (1 - torch.abs(labels_batch))

    win_lose = torch.max(torch.tensor(0.0, device=device), diff_win_w)
    tie = torch.max(torch.tensor(0.0, device=device), diff_tie_t)

    return torch.mean(torch.add(win_lose, tie))


def custom_loss(left_images_batch_scores, right_images_batch_scores, labels_batch, m_w, m_t, device):
    # Convert labels to float (if needed)
    labels_batch = labels_batch.float()

    # MarginRankingLoss for win/lose scenario
    margin_loss_fn = nn.MarginRankingLoss(margin=m_w, reduction='none')  # Keep element-wise loss
    win_lose = margin_loss_fn(left_images_batch_scores, right_images_batch_scores, labels_batch)

    # Tie scenario (custom)
    # diff_adj_t = torch.abs(left_images_batch_scores - right_images_batch_scores) - m_t
    # diff_tie_t = diff_adj_t * (1 - torch.abs(labels_batch))
    # tie = torch.max(torch.tensor(0.0, device=device), diff_tie_t)
    huber_loss_fn = nn.SmoothL1Loss(beta=m_t, reduction='none')
    tie = huber_loss_fn(left_images_batch_scores, right_images_batch_scores) * (1 - torch.abs(labels_batch))

    # Combine and return mean loss
    return torch.mean(win_lose + tie)


def transform():
    transform_img = transforms.Compose([
        transforms.Lambda(lambda img: crop_from_bottom(img, 25)),  # Custom crop
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])
    return transform_img


def unique_images_votes_df(votes_path, test_size, random_state=42):
    votes_df = (pd.read_csv(votes_path, sep='\t').query("study_id == '50a68a51fdc9f05596000002'"))  # Filter by study_id)
    all_images = set(votes_df["left"]).union(set(votes_df["right"]))
    train_images, val_images = train_test_split(list(all_images), test_size=test_size, random_state=random_state)
    train_df = votes_df[votes_df["left"].isin(train_images) & votes_df["right"].isin(train_images)]
    val_df = votes_df[votes_df["left"].isin(val_images) & votes_df["right"].isin(val_images)]
    return train_df, val_df
