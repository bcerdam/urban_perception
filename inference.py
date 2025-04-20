import argparse
import os
import torch
import sys
import torchvision.transforms as T
import datetime
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", module="torch._utils")
from PIL import Image
from torchvision.models import resnet50
from core_py.nets.raw_feat import RawFeatInference
from core_py.nets.raw_vit import RawViTInference


def process_image(image_path, transform, model):
    image_basename = os.path.basename(image_path)

    with Image.open(image_path) as img:
        if img.mode == 'RGB':
            img_rgb = img
        elif img.mode in ['RGBA', 'L', 'P']:
            img_rgb = img.convert('RGB')

        tensor = transform(img_rgb)
        score_tensor = model.forward(tensor.unsqueeze(0))
        score = score_tensor.item()

        return score, image_basename


def main():
    parser = argparse.ArgumentParser(description="Process single image or folder for inference.")
    parser.add_argument("--model", type=str, help="Model, RawFeat or RawViT.")
    parser.add_argument("--input_path", type=str, help="Path to a single image file or a folder containing images.")
    parser.add_argument('--weights_path', type=str, help="Path to model weights")
    args = parser.parse_args()

    execution_start_time = datetime.datetime.now()
    time_based_subdir_name = execution_start_time.strftime("%d_%m_%H-%M-%S")
    base_results_dir = "inference_results"
    run_results_dir = os.path.join(base_results_dir, time_based_subdir_name)
    log_filename = "results.txt"
    log_filepath = os.path.join(run_results_dir, log_filename)

    # Create the nested directories if they don't exist
    os.makedirs(run_results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = args.input_path

    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToTensor()
    ])

    if args.model == 'RawFeat':
        model = resnet50(weights='DEFAULT')
        model = RawFeatInference(model, args.weights_path, device)
        model.load_weights(args.weights_path, device)
        model.eval()
    elif args.model == 'RawViT':
        model = RawViTInference(weight_path=args.weights_path, device=device, hf_model_name='google/vit-base-patch16-224-in21k').to(device)
        model.eval()

    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(log_filepath, 'a') as log_file:
        if os.path.isfile(input_path):
            score, img_name = process_image(input_path, transform, model)
            log_file.write(f"{img_name}: {score}\n")

        elif os.path.isdir(input_path):
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            for filename in sorted(os.listdir(input_path)):
                file_path = os.path.join(input_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
                    score, img_name = process_image(file_path, transform, model)
                    log_file.write(f"{img_name}: {score}\n")
                elif os.path.isfile(file_path):
                     print(f"Skipping non-image file: {filename}")
        else:
            print(f"Error: Input path is neither a file nor a directory: {input_path}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()