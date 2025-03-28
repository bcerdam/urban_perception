import utils
import torch


def train_one_epoch(epoch_index, num_epochs, train_dataloader, device, optimizer, model, m_w, m_t, similarity_threshold):
    model.train()

    running_loss = 0.0
    last_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for batch_idx, batch in enumerate(train_dataloader):
        left_images_batch = batch[0].to(device)
        right_images_batch = batch[1].to(device)
        labels_batch = batch[2].unsqueeze(dim=1).to(device)

        optimizer.zero_grad()

        scores_batch = model.forward(left_images_batch, right_images_batch, left_images_batch.shape[0], right_images_batch.shape[0])
        loss_batch = utils.loss(scores_batch[0], scores_batch[1], labels_batch, m_w, m_t, device)

        loss_batch.backward()
        optimizer.step()

        running_loss += loss_batch.item()
        last_loss = running_loss / (batch_idx + 1)

        left_scores = scores_batch[0].squeeze()
        right_scores = scores_batch[1].squeeze()
        predictions = torch.zeros_like(labels_batch.squeeze())
        predictions[left_scores > right_scores] = 1
        predictions[left_scores < right_scores] = -1
        predictions[torch.abs(left_scores - right_scores) < similarity_threshold] = 0

        correct_predictions += (predictions == labels_batch.squeeze()).sum().item()
        total_samples += labels_batch.squeeze().shape[0]

    accuracy = (correct_predictions / total_samples) * 100
    print(f'Epoch: {epoch_index}/{num_epochs}, Train Loss: {last_loss}, Train Accuracy: {accuracy:.2f}%')
    return last_loss