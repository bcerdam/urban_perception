from core_py.utils import utils
import torch
import os


def train_one_epoch(epoch_index, train_dataloader, device, optimizer, model, m_w, m_t, similarity_threshold, pp2_train, run_checkpoint_dir):
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

    os.makedirs(f'{run_checkpoint_dir}/epoch_pth', exist_ok=True)
    checkpoint_path = os.path.join(run_checkpoint_dir+'/epoch_pth', f"model_epoch_{epoch_index}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    utils.metrics(pp2_train, model, m_w, m_t, similarity_threshold, checkpoint_path, epoch_index, run_checkpoint_dir)

    return checkpoint_path