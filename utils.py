import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F


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
    """Crops the given image from the bottom by a specified number of pixels."""
    image = F.to_pil_image(image)
    width, height = image.size  # Get the image dimensions
    image = image.crop((0, 0, width, height - crop_pixels))
    return F.to_tensor(image)

def loss(left_images_batch_scores, right_images_batch_scores, labels_batch, m_w, m_t, device):
    # win_lose = torch.max(torch.tensor(0, device=device), -1 * labels_batch * (left_images_batch_scores - right_images_batch_scores) + m_w)
    # tie = torch.max(torch.tensor(0, device=device), torch.abs(left_images_batch_scores - right_images_batch_scores) - m_t)
    # return torch.mean(torch.add(win_lose, tie))

    win_lose = torch.max(torch.tensor(0, device=device), -1 * labels_batch * (left_images_batch_scores - right_images_batch_scores) + m_w*torch.abs(labels_batch))
    tie = torch.max(torch.tensor(0, device=device), (torch.abs(left_images_batch_scores - right_images_batch_scores) - m_t) * (1 - torch.abs(labels_batch)))

    return torch.mean(torch.add(win_lose, tie))


