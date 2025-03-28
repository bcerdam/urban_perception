import utils
import torch
from test import RawFeatInference
from torchvision import models


def validate_model(weight_path, validation_dataloader, m_w, m_t, similarity_threshold):
    resnet50 = models.resnet50(weights='DEFAULT')
    model = RawFeatInference(resnet50, weight_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    running_loss = 0.0
    last_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    count_left = 0
    count_right = 0
    count_tie = 0
    count_left_label = 0
    count_right_label = 0
    count_tie_label = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dataloader):
            left_images_batch = batch[0].to(device)
            right_images_batch = batch[1].to(device)
            labels_batch = batch[2].unsqueeze(dim=1).to(device)

            left_scores_batch = model.forward(left_images_batch)
            right_scores_batch = model.forward(right_images_batch)
            loss_batch = utils.loss(left_scores_batch, right_scores_batch, labels_batch, m_w, m_t, device)

            running_loss += loss_batch.item()
            last_loss = running_loss / (batch_idx + 1)

            left_scores = left_scores_batch.squeeze()
            right_scores = right_scores_batch.squeeze()
            predictions = torch.zeros_like(labels_batch.squeeze())
            predictions[left_scores > right_scores] = 1
            predictions[left_scores < right_scores] = -1
            predictions[torch.abs(left_scores - right_scores) < similarity_threshold] = 0

            correct_predictions += (predictions == labels_batch.squeeze()).sum().item()
            total_samples += labels_batch.squeeze().shape[0]

            count_left += (predictions == 1).sum().item()
            count_right += (predictions == -1).sum().item()
            count_tie += (predictions == 0).sum().item()

            count_left_label += (labels_batch.squeeze() == 1).sum().item()
            count_right_label += (labels_batch.squeeze() == -1).sum().item()
            count_tie_label += (labels_batch.squeeze() == 0).sum().item()

    accuracy = (correct_predictions / total_samples) * 100
    # print(f'Epoch: {epoch_index}/{num_epochs}, Validation Loss: {last_loss}, Validation Accuracy: {accuracy:.2f}%')
    print(f'Validation Loss: {last_loss}, Validation Accuracy: {accuracy:.2f}%')
    print(f'Left: Predichos: {count_left}, Totales: {count_left_label}')
    print(f'Right: Predichos: {count_right}, Totales: {count_right_label}')
    print(f'Tie: Predichos: {count_tie}, Totales: {count_tie_label}')
    total = count_left_label + count_right_label + count_tie_label
    print(f'Left: {count_left/total} -> {count_left_label/total}, Right: {count_right/total} -> {count_right_label/total}, Tie: {count_tie/total} -> {count_tie_label/total}')
    print('-----------------')





# def validate_model(epoch_index, num_epochs, validation_dataloader, device, model, m_w, m_t, similarity_threshold):
#     model.eval()
#
#     running_loss = 0.0
#     last_loss = 0.0
#     correct_predictions = 0
#     total_samples = 0
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(validation_dataloader):
#             left_images_batch = batch[0].to(device)
#             right_images_batch = batch[1].to(device)
#             labels_batch = batch[2].unsqueeze(dim=1).to(device)
#
#             scores_batch = model.forward(left_images_batch, right_images_batch, left_images_batch.shape[0], right_images_batch.shape[0])
#             loss_batch = utils.loss(scores_batch[0], scores_batch[1], labels_batch, m_w, m_t, device)
#
#             running_loss += loss_batch.item()
#             last_loss = running_loss / (batch_idx + 1)
#
#             left_scores = scores_batch[0].squeeze()
#             right_scores = scores_batch[1].squeeze()
#             predictions = torch.zeros_like(labels_batch.squeeze())
#             predictions[left_scores > right_scores] = 1
#             predictions[left_scores < right_scores] = -1
#             predictions[torch.abs(left_scores - right_scores) < similarity_threshold] = 0
#
#             correct_predictions += (predictions == labels_batch.squeeze()).sum().item()
#             total_samples += labels_batch.squeeze().shape[0]
#
#     accuracy = (correct_predictions / total_samples) * 100
#     print(f'Epoch: {epoch_index}/{num_epochs}, Validation Loss: {last_loss}, Validation Accuracy: {accuracy:.2f}%')
#
#     return last_loss