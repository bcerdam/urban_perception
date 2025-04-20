import torch
import numpy as np
import os
import re
# import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpld3
import torch.nn.functional as F
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

# --- (Keep your existing preprocess_for_plot for the *images*) ---
def preprocess_for_plot(tensor_or_array):
    # ... (your existing function is fine for images) ...
    if isinstance(tensor_or_array, torch.Tensor):
        # Handle tensors: permute C,H,W to H,W,C if needed, convert to numpy
        if tensor_or_array.ndim == 3:
                # Remove channel dim if it's 1 (e.g., grayscale or single attention map channel)
            if tensor_or_array.shape[0] == 1:
                return tensor_or_array.squeeze().cpu().numpy()
            else: # Assume C,H,W -> H,W,C
                return tensor_or_array.permute(1, 2, 0).cpu().numpy()
        elif tensor_or_array.ndim == 2: # Assume H,W
                return tensor_or_array.cpu().numpy()
        else: # Handle the 4D attention map case *outside* this function for now
            raise TypeError(f"preprocess_for_plot expects 2D/3D input, got {tensor_or_array.ndim}D")

    elif isinstance(tensor_or_array, np.ndarray):
            # Handle numpy arrays: remove channel dim if it's 1
            if tensor_or_array.ndim == 3 and tensor_or_array.shape[-1] == 1: # H,W,1
                return tensor_or_array.squeeze(axis=-1)
            elif tensor_or_array.ndim == 3 and tensor_or_array.shape[0] == 1: # 1,H,W
                return tensor_or_array.squeeze(axis=0)
            elif tensor_or_array.ndim == 2: # H, W
                return tensor_or_array
            elif tensor_or_array.ndim == 3: # Assume H, W, C
                 return tensor_or_array
            else:
                 raise TypeError(f"preprocess_for_plot expects 2D/3D input, got {tensor_or_array.ndim}D")

    else:
        raise TypeError(f"Input must be a torch.Tensor or np.ndarray, got {type(tensor_or_array)}")


def process_vit_attention(attn_raw, target_height, target_width, num_patches_side=14):
    """Processes raw ViT attention for visualization."""
    if attn_raw.shape[1] != 12 or attn_raw.shape[2] != 197 or attn_raw.shape[3] != 197:
         print(f"Warning: Unexpected attention map shape: {attn_raw.shape}. Assuming ViT [1, 12, 197, 197] structure.")
         # Add fallback or error handling if needed

    # 1. Average across heads
    attn_mean_heads = attn_raw.mean(dim=1) # Shape: [1, 197, 197]

    # 2. Get attention from CLS token to patch tokens
    cls_attention = attn_mean_heads[0, 0, 1:] # Shape: [196]

    # 3. Reshape to grid
    # Ensure number of patches matches
    if cls_attention.shape[0] != num_patches_side * num_patches_side:
        raise ValueError(f"Number of attention scores ({cls_attention.shape[0]}) doesn't match grid ({num_patches_side}x{num_patches_side})")
    attention_grid = cls_attention.reshape(num_patches_side, num_patches_side) # Shape: [14, 14]

    # 4. Resize to image size
    attention_grid_tensor = attention_grid.unsqueeze(0).unsqueeze(0).float() # Shape: [1, 1, 14, 14]
    attention_heatmap_resized = F.interpolate(
        attention_grid_tensor,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    ) # Shape: [1, 1, H, W]

    # Remove batch and channel for plotting
    final_attention_map = attention_heatmap_resized.squeeze().cpu().numpy() # Shape: [H, W]
    return final_attention_map

def overlay_attention_on_image(image, attention_map, colormap_name='viridis', alpha=0.5):
    """
    Overlays a normalized attention map onto an image using a colormap.

    Args:
        image (np.ndarray): Original image (H, W) or (H, W, C). Assumed to be float[0,1] or uint8[0,255].
        attention_map (np.ndarray): 2D attention map (H, W). Should have same H, W as image.
        colormap_name (str): Name of the matplotlib colormap to use.
        alpha (float): Transparency level for the attention heatmap (0.0 to 1.0).

    Returns:
        np.ndarray: Image with attention map overlaid (H, W, C), dtype=uint8 [0, 255].
    """
    # 1. Normalize attention map to 0-1 range
    norm_attn = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-6) # Add epsilon for flat maps

    # 2. Get the colormap and apply it
    cmap = cm.get_cmap(colormap_name)
    heatmap_rgba = cmap(norm_attn) # Get RGBA heatmap [0, 1]
    heatmap_rgb = heatmap_rgba[:, :, :3] # Take only RGB channels

    # 3. Ensure image is in RGB format and float [0, 1]
    if image.ndim == 2: # Grayscale
        image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) # Convert to uint8 RGB
    elif image.shape[2] == 1: # Grayscale with channel dim
         image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) # Convert to uint8 RGB
    elif image.dtype == np.uint8:
        image_rgb = image # Already uint8 RGB
    else: # Assume float color image
        image_rgb = (image * 255).astype(np.uint8) # Convert float[0,1] to uint8[0,255] RGB

    # Convert image to float [0, 1] for blending
    image_rgb_float = image_rgb.astype(float) / 255.0

    # 4. Convert heatmap to match image dimensions (should already be done by process_vit_attention)
    # Ensure heatmap_rgb is also float [0,1] (it should be from cmap)

    # 5. Alpha Blending
    blended_img_float = alpha * heatmap_rgb + (1.0 - alpha) * image_rgb_float

    # 6. Clip values and convert back to uint8
    blended_img_uint8 = np.clip(blended_img_float * 255, 0, 255).astype(np.uint8)

    return blended_img_uint8


# --- Modified Plotting Function ---
def plot_tuple_with_attention(data, value1, value2, left_attention_input, right_attention_input,
                              save_path=None, alpha=0.5, colormap_name='viridis'): # Added colormap_name arg
    """
    Plots two images (top row) and the images with attention overlaid (bottom row).
    Includes colorbars corresponding to the attention values in the overlay.
    Handles attention input potentially being a tuple of tensors (selects the last).
    """
    # --- 1. Unpack Data and Map Label ---
    image1, image2, label, title_text, left_place_name, right_place_name = data
    label_map = {1: "Left", 0: "Equal", -1: "Right"}
    preference = label_map.get(label, "Unknown")

    # --- 2. Preprocess Images ---
    try:
        img1_processed = preprocess_for_plot(image1)
        img2_processed = preprocess_for_plot(image2)
    except Exception as e:
        print(f"Error preprocessing images: {e}")
        return
    target_h, target_w = img1_processed.shape[0], img1_processed.shape[1]

    # --- 3. Select and Process Attention Tensor ---
    try:
        # (Selection logic for tuples - same as before)
        if isinstance(left_attention_input, tuple):
            if not left_attention_input: raise ValueError("Empty tuple for left_attention_input")
            left_attn_tensor = left_attention_input[-1]
        else: left_attn_tensor = left_attention_input
        if isinstance(right_attention_input, tuple):
            if not right_attention_input: raise ValueError("Empty tuple for right_attention_input")
            right_attn_tensor = right_attention_input[-1]
        else: right_attn_tensor = right_attention_input

        if not isinstance(left_attn_tensor, torch.Tensor): raise TypeError(f"Left attn not Tensor: {type(left_attn_tensor)}")
        if not isinstance(right_attn_tensor, torch.Tensor): raise TypeError(f"Right attn not Tensor: {type(right_attn_tensor)}")

        # Process attention (resize etc.)
        num_patches_sqrt = int(np.sqrt(left_attn_tensor.shape[-1] - 1))
        attn1_processed = process_vit_attention(left_attn_tensor, target_h, target_w, num_patches_sqrt)
        attn2_processed = process_vit_attention(right_attn_tensor, target_h, target_w, num_patches_sqrt)

    except (TypeError, ValueError, IndexError) as e: # Added IndexError for [-1]
        print(f"Error selecting/processing attention tensor: {e}")
        return
    except Exception as e:
        print(f"Error during attention processing: {e}")
        return

    # --- 4. Create Overlay Images ---
    try:
        # Pass the colormap name to the overlay function
        overlay1 = overlay_attention_on_image(img1_processed, attn1_processed, colormap_name=colormap_name, alpha=alpha)
        overlay2 = overlay_attention_on_image(img2_processed, attn2_processed, colormap_name=colormap_name, alpha=alpha)
    except Exception as e:
        print(f"Error creating overlay images: {e}")
        return

    # --- 5. Plotting ---
    fig, ax = plt.subplots(2, 2, figsize=(13, 12), dpi=300) # Slightly wider figsize for colorbars

    # --- Top Row: Original Images ---
    ax[0, 0].imshow(img1_processed)
    ax[0, 0].axis('off')
    ax[0, 0].set_title(f"{left_place_name} (Original)")

    ax[0, 1].imshow(img2_processed)
    ax[0, 1].axis('off')
    ax[0, 1].set_title(f"{right_place_name} (Original)")

    # --- Bottom Row: Overlays with Colorbars ---

    # Left Overlay + Colorbar
    # Plot raw heatmap invisibly first to get the mappable object
    norm1 = plt.Normalize(vmin=np.min(attn1_processed), vmax=np.max(attn1_processed)) # Normalize based on this map's range
    im_map1 = ax[1, 0].imshow(attn1_processed, cmap=colormap_name, norm=norm1) # Store mappable
    # Plot the actual overlay on top
    ax[1, 0].imshow(overlay1)
    ax[1, 0].axis('off')
    ax[1, 0].set_title(f"Left Attention Overlay (Score: {value1:.3f})")
    # Add colorbar linked to the raw heatmap's mappable
    fig.colorbar(im_map1, ax=ax[1, 0], fraction=0.046, pad=0.04)

    # Right Overlay + Colorbar
    norm2 = plt.Normalize(vmin=np.min(attn2_processed), vmax=np.max(attn2_processed)) # Normalize based on this map's range
    im_map2 = ax[1, 1].imshow(attn2_processed, cmap=colormap_name, norm=norm2) # Store mappable
    ax[1, 1].imshow(overlay2) # Plot overlay on top
    ax[1, 1].axis('off')
    ax[1, 1].set_title(f"Right Attention Overlay (Score: {value2:.3f})")
    fig.colorbar(im_map2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    # --- Overall Title & Layout ---
    plt.suptitle(f"Comparison: {title_text}, {preference} was selected", fontsize=16)
    # Use constrained_layout or adjust tight_layout further if needed
    plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Adjust right boundary slightly for colorbars if needed

    # --- 6. Save or Show ---
    try:
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
    except Exception as e:
        print(f"Error saving or showing plot: {e}")
        plt.close(fig)


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
