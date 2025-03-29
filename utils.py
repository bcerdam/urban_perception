import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from raw_feat import RawFeatInference
from raw_feat_reg import RawFeatRegInference
from torchvision import models

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

    comb = torch.add(win_lose, tie)
    comb_mean = torch.mean(comb)
    return comb_mean


def metrics(dataset, inference_model, m_w, m_t, similarity_threshold, weight_path, epoch_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if inference_model == "RawFeat":
        resnet50 = models.resnet50(weights='DEFAULT')
        model = RawFeatInference(resnet50, weight_path)
        model.to(device)
        model.eval()
    elif inference_model == 'RawFeatReg':
        resnet18 = models.resnet18(weights='DEFAULT')
        model = RawFeatRegInference(resnet18, weight_path)
        model.to(device)
        model.eval()
    elif type(inference_model) != str:
        model = inference_model
        model.to(device)
        model.eval()

    running_loss = 0.0
    total = 0
    count_left = 0
    count_right = 0
    count_tie = 0

    count_left_label = 0
    count_right_label = 0
    count_tie_label = 0
    with torch.no_grad():
        for vote in dataset:
            left_image = vote[0].to(device)
            right_image = vote[1].to(device)
            label = torch.tensor(vote[2])

            if type(inference_model) != str:
                scores = model.forward(left_image.unsqueeze(0), right_image.unsqueeze(0), 1, 1)
                left_score = scores[0]
                right_score = scores[1]
                loss_value = loss(left_score, right_score, label, m_w, m_t, device)
                split = 'Train'
            else:
                left_score = model.forward(left_image.unsqueeze(0))
                right_score = model.forward(right_image.unsqueeze(0))
                loss_value = loss(left_score, right_score, label, m_w, m_t, device)
                split = 'Validation'

            running_loss += loss_value

            if label == 1:
                count_left_label += 1
                if left_score > right_score:
                    count_left += 1
                    total += 1
                else:
                    total += 1
            elif label == -1:
                count_right_label += 1
                if left_score < right_score:
                    count_right += 1
                    total += 1
                else:
                    total += 1
            elif label == 0:
                count_tie_label += 1
                if np.abs(left_score.item() - right_score.item()) < similarity_threshold:
                    count_tie += 1
                    total += 1
                else:
                    total += 1

    accuracy = ((count_left+count_right+count_tie)/len(dataset))

    if split == 'Train':
        output = (f'Epoch {epoch_index}\n'
                  f'{split} Loss: {running_loss/len(dataset)}, {split} Accuracy: {accuracy}\n'
                  f'Left: Predichos: {count_left}, Totales: {count_left_label}, Diff: {np.abs(count_left-count_left_label)}\n'
                  f'Right: Predichos: {count_right}, Totales: {count_right_label}, Diff: {np.abs(count_right-count_right_label)}\n'
                  f'Tie: Predichos: {count_tie}, Totales: {count_tie_label}, Diff: {np.abs(count_tie-count_tie_label)}\n'
                  f'Left: {count_left / total:.4f} -> {count_left_label / total:.4f} '
                  f'Right: {count_right / total:.4f} -> {count_right_label / total:.4f}, '
                  f'Tie: {count_tie / total:.4f} -> {count_tie_label / total:.4f}\n')
    else:
        output = (f'{split} Loss: {running_loss / len(dataset)}, {split} Accuracy: {accuracy}\n'
                  f'Left: Predichos: {count_left}, Totales: {count_left_label}, Diff: {np.abs(count_left - count_left_label)}\n'
                  f'Right: Predichos: {count_right}, Totales: {count_right_label}, Diff: {np.abs(count_right - count_right_label)}\n'
                  f'Tie: Predichos: {count_tie}, Totales: {count_tie_label}, Diff: {np.abs(count_tie - count_tie_label)}\n'
                  f'Left: {count_left / total:.4f} -> {count_left_label / total:.4f} '
                  f'Right: {count_right / total:.4f} -> {count_right_label / total:.4f}, '
                  f'Tie: {count_tie / total:.4f} -> {count_tie_label / total:.4f}\n'
                  '-----------------\n')

    print(output)

    # Ensure 'status' directory exists
    status_dir = "status"
    os.makedirs(status_dir, exist_ok=True)

    # Append results to 'status.txt'
    status_file = os.path.join(status_dir, "status.txt")
    with open(status_file, "a") as f:
        f.write(output)


def clear_directory(directory):
    """Deletes all files inside a given directory."""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")