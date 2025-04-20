import torch
import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from core_py.nets.raw_feat import RawFeatInference
from core_py.nets.raw_feat_reg import RawFeatRegInference
from torchvision import models
from core_py.nets.raw_vit import RawViTInference


def plot_tuple(data, value1, value2, save_path=None):
    """
    Plots a tuple containing two images, an integer, a string, and two float values.
    Optionally saves the plot as a PNG file.

    Parameters:
        data (tuple): A tuple of size 4.
                      - The first two elements are torch.Tensor images (C, H, W).
                      - The third element is an integer (1, 0, or -1).
                      - The fourth element is a string.
        value1 (float): The float value to be displayed below the first image.
        value2 (float): The float value to be displayed below the second image.
        save_path (str, optional): Path to save the plot as a PNG file. If not provided, it will only plot.
    """
    image1, image2, label, title_text, left_place_name, right_place_name = data

    label_map = {1: "Left", 0: "Equal", -1: "Right"}
    preference = label_map.get(label, "Unknown")

    def tensor_to_image(tensor):
        if tensor.ndim == 3:  # (C, H, W)
            return tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        return tensor.numpy()  # Assume already in (H, W)

    img1 = tensor_to_image(image1)
    img2 = tensor_to_image(image2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), dpi=300)

    ax[0].imshow(img1, cmap='gray' if img1.ndim == 2 else None)
    ax[0].axis('off')
    ax[0].set_title(left_place_name)

    ax[1].imshow(img2, cmap='gray' if img2.ndim == 2 else None)
    ax[1].axis('off')
    ax[1].set_title(right_place_name)

    fig.text(0.25, 0.02, f"Score: {value1:.3f}", ha='center', fontsize=12)
    fig.text(0.75, 0.02, f"Score: {value2:.3f}", ha='center', fontsize=12)

    plt.suptitle(f"Comparison: {title_text}, {preference} was selected", fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_tuple_with_attention(data, value1, value2, left_attention_map, right_attention_map, save_path=None):
    image1, image2, label, title_text, left_place_name, right_place_name = data

    label_map = {1: "Left", 0: "Equal", -1: "Right"}
    preference = label_map.get(label, "Unknown")

    def preprocess_for_plot(tensor_or_array):
        """Converts input to numpy array suitable for imshow."""
        if isinstance(tensor_or_array, torch.Tensor):
            # Handle tensors: permute C,H,W to H,W,C if needed, convert to numpy
            if tensor_or_array.ndim == 3:
                 # Remove channel dim if it's 1 (e.g., grayscale or single attention map channel)
                if tensor_or_array.shape[0] == 1:
                    return tensor_or_array.squeeze().cpu().numpy()
                else: # Assume C,H,W -> H,W,C
                    return tensor_or_array.permute(1, 2, 0).cpu().numpy()
            else: # Assume H,W
                 return tensor_or_array.cpu().numpy()
        elif isinstance(tensor_or_array, np.ndarray):
             # Handle numpy arrays: remove channel dim if it's 1
             if tensor_or_array.ndim == 3 and tensor_or_array.shape[-1] == 1: # H,W,1
                 return tensor_or_array.squeeze(axis=-1)
             elif tensor_or_array.ndim == 3 and tensor_or_array.shape[0] == 1: # 1,H,W
                 return tensor_or_array.squeeze(axis=0)
             else: # Assume H,W or H,W,C
                 return tensor_or_array
        else:
            raise TypeError(f"Input must be a torch.Tensor or np.ndarray, got {type(tensor_or_array)}")


    img1 = preprocess_for_plot(image1)
    img2 = preprocess_for_plot(image2)
    attn1 = preprocess_for_plot(left_attention_map)
    attn2 = preprocess_for_plot(right_attention_map)

    # Create a 2x2 subplot grid
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=300) # Increased figure size for 2 rows

    # --- Top Row: Images ---
    ax[0, 0].imshow(img1, cmap='gray' if img1.ndim == 2 else None)
    ax[0, 0].axis('off')
    ax[0, 0].set_title(left_place_name)

    ax[0, 1].imshow(img2, cmap='gray' if img2.ndim == 2 else None)
    ax[0, 1].axis('off')
    ax[0, 1].set_title(right_place_name)

    # --- Bottom Row: Attention Maps ---
    # Use a perceptually uniform colormap like 'viridis' or 'inferno'
    im_attn1 = ax[1, 0].imshow(attn1, cmap='viridis')
    ax[1, 0].axis('off')
    ax[1, 0].set_title(f"Left Attention (Score: {value1:.3f})") # Include score in title
    # Optional: Add a colorbar for the attention map
    # fig.colorbar(im_attn1, ax=ax[1, 0], fraction=0.046, pad=0.04)


    im_attn2 = ax[1, 1].imshow(attn2, cmap='viridis')
    ax[1, 1].axis('off')
    ax[1, 1].set_title(f"Right Attention (Score: {value2:.3f})") # Include score in title
    # Optional: Add a colorbar for the attention map
    # fig.colorbar(im_attn2, ax=ax[1, 1], fraction=0.046, pad=0.04)


    # --- Overall Title ---
    plt.suptitle(f"Comparison: {title_text}, {preference} was selected", fontsize=16)

    # Adjust layout - might need tweaking depending on title lengths etc.
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to prevent suptitle overlap

    # --- Save or Show ---
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig) # Close the figure after saving to free memory
    else:
        plt.show()


def loss(left_images_batch_scores, right_images_batch_scores, labels_batch, m_w, m_t, device):
    labels_batch = labels_batch.float()
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


def metrics(dataset, inference_model, m_w, m_t, similarity_threshold, weight_path, epoch_index, run_checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if inference_model == "RawFeat":
        resnet50 = models.resnet50(weights='DEFAULT')
        model = RawFeatInference(resnet50, weight_path, device)
        model.to(device)
        model.eval()
    elif inference_model == 'RawViT':
        model = RawViTInference(weight_path, device, hf_model_name='google/vit-base-patch16-224-in21k').to(device)
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

    os.makedirs(run_checkpoint_dir+'/status', exist_ok=True)
    status_file = os.path.join(run_checkpoint_dir+'/status', "status.txt")
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



def parse_and_plot_data(txt_path, plot_path):
    def parse_epoch_data(text):
        epoch_data = {}

        epoch_match = re.search(r'Epoch (\d+)', text)
        if epoch_match:
            epoch_data['Epoch'] = int(epoch_match.group(1))

        train_loss_match = re.search(r'Train Loss: (\S+)', text)
        train_accuracy_match = re.search(r'Train Accuracy: (\S+)', text)
        if train_loss_match and train_accuracy_match:
            epoch_data['Train Loss'] = float(train_loss_match.group(1).strip(','))
            epoch_data['Train Accuracy'] = float(train_accuracy_match.group(1))

        val_loss_match = re.search(r'Validation Loss: (\S+)', text)
        val_accuracy_match = re.search(r'Validation Accuracy: (\S+)', text)
        if val_loss_match and val_accuracy_match:
            epoch_data['Validation Loss'] = float(val_loss_match.group(1).strip(','))
            epoch_data['Validation Accuracy'] = float(val_accuracy_match.group(1))

        return epoch_data

    with open(txt_path, 'r') as f:
        data = f.read().split('-----------------')

    epochs = []
    for epoch_text in data:
        if 'Epoch' in epoch_text:
            epoch_data = parse_epoch_data(epoch_text)
            epochs.append(epoch_data)

    df = pd.DataFrame(epochs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), dpi=300)

    train_color = 'blue'
    val_color = 'red'

    ax1.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color=train_color, marker='o')
    ax1.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', color=val_color, marker='x')
    ax1.set_title('Loss', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy', color=train_color, marker='o')
    ax2.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy', color=val_color, marker='x')
    ax2.set_title('Accuracy', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    mpld3.save_html(fig, plot_path)

    plt.show()

# parse_and_plot_data('data/Reuniones/2/25k/status.txt', 'data/Reuniones/2/25k/metrics.html')
