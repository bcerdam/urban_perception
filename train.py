import utils
import torch
import os
from torch.utils.data import Dataset
# from torch.utils.data import Subset


class SampledDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Store sampled data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Return sample

def train_one_epoch(epoch_index, train_dataloader, device, optimizer, model, m_w, m_t, similarity_threshold, pp2_train):
    model.train()

    for batch_idx, batch in enumerate(train_dataloader):
        left_images_batch = batch[0].to(device)
        right_images_batch = batch[1].to(device)
        labels_batch = batch[2].unsqueeze(dim=1).to(device)

        optimizer.zero_grad()

        scores_batch = model.forward(left_images_batch, right_images_batch, left_images_batch.shape[0], right_images_batch.shape[0])
        loss_batch = utils.loss(scores_batch[0], scores_batch[1], labels_batch, m_w, m_t, device)

        loss_batch.backward()
        optimizer.step()

    os.makedirs('model_checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('model_checkpoints', f"model_epoch_{epoch_index}.pth")
    torch.save(model.state_dict(), checkpoint_path)

    # dataset_size = len(pp2_train)
    # indices = torch.randperm(dataset_size)[:100]  # Randomly select 100 samples
    # sampled_data = [pp2_train[int(i)] for i in indices]
    # sampled_pp2_train = SampledDataset(sampled_data)
    #
    # utils.metrics(sampled_pp2_train, model, m_w, m_t, similarity_threshold, checkpoint_path, epoch_index)
    utils.metrics(pp2_train, model, m_w, m_t, similarity_threshold, checkpoint_path, epoch_index)
    return checkpoint_path