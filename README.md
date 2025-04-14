# About

Pytorch implementation for: "Reconciling explainability and performance in neural networks by means of semantic segmentation-guided feature attention - An application to urban space perception" paper.


# Installation

## Step 1: Clone repo.

From a desired directory, do this:

```console
git clone https://github.com/bcerdam/urban_perception.git
cd urban_perception
python3 -m venv env
source ./env/bin/activate
```

## Step 2: Install requirements.

```console
pip install -r requirements.txt
```

## Training.

For training, the PlacePulse dataset is a necessity. A preprocessed version can be found [here](https://youtu.be/xvFZjo5PgG0).

```console
python3 train.py --model RawFeat --epochs 5 --votes_sample_size 5000 --votes_train_size_percentage 0.75 --image_test_size_percentage 0.25 --batch_size 64 --learning_rate 0.001 --m_w 1 --m_t 1 --similarity_threshold 1 --locations_path data/cleaned_locations.tsv --places_path data/places.tsv --images_path data/images --votes_path data/cleaned_votes.tsv
```

- **model**: The only model available right now is RawFeat.
- **votes_sample_size**: Samples a certain amount of votes from the votes dataset.
- **votes_train_size_percentage**: Train split % for the PP2 votes dataset.
- **image_test_size_percentage**: Test split % for the PP2 images dataset.
- **m_w**: Hyperparameter that models the win margin
- **m_t**: Hyperparameter that models the tie margin.
- **similarity_threshold**: Threshold for detecting a tie between two scores.

Weight checkpoints and training/validation info can be found on model_checkpoints/.

## Inference

- RawFeat: Weights can be found [here](https://drive.google.com/drive/folders/1Y_H_ZLpzu4EFxfknKRVrssycGnG-3Sct?usp=sharing).

```console
python3 inference.py --input_path data/test_inference_images --weights_path weights/RawFeat/model_epoch_1.pth
```

You can find the scores on inference_results/.
