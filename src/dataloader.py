import torch
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker

from configs.cfg_importer import *
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
import mediapipe as mp

cfg = import_cfg()

test_path = "data/asl_processed/test"
train_path = "data/asl_processed/train"
asl_test = os.path.join(cfg['working_path'], test_path)
asl_train = os.path.join(cfg['working_path'], train_path)



def init_resnet18():
    transforms = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.RandomRotation(15),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(asl_train, transforms)

    # Includes real-world augmentations
    test_messy_ds = ImageFolder(asl_test, transforms)

    # Ideal set
    test_clean_ds = ImageFolder(asl_test, v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    tr_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    ts_clean = DataLoader(test_clean_ds, batch_size=cfg['batch_size'], shuffle=True)
    ts_messy = DataLoader(test_messy_ds, batch_size=cfg['batch_size'], shuffle=True)

    images, labels = next(iter(tr_loader))
    class_names = train_ds.classes

    # Unnormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img = images[i] * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    return tr_loader, ts_clean, ts_messy

def init_lmc():
    # mediapipe stuff
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    mp_modelpath = os.path.join(cfg['working_path'], 'models/hand_landmarker.task')

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_modelpath),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        return get_landmarks(landmarker)


def get_landmarks(landmarker: HandLandmarker):
    """Returns a list of classes with their respective landmarks per image"""

    hand_landmarks: dict[str, list[mp.tasks.vision.HandLandmarkerResult]] = {}
    for class_folder in sorted(os.listdir(asl_train)):
        class_images = os.path.join(asl_train, class_folder)
        if not os.path.isdir(class_images):
            continue

        for path in os.listdir(class_images):
            img_path = os.path.join(class_images, path)
            mp_image = mp.Image.create_from_file(file_name=img_path)

            landmarks = landmarker.detect(image=mp_image)
            landmarks = landmarks.hand_landmarks

            cls_lmarks = []

            if landmarks:
                for hand in landmarks:
                    cls_lmarks.append(hand)
                hand_landmarks.update({f"{class_folder}": cls_lmarks})

        print(f"loaded landmarks for class: {class_folder}")

    # Save landmarks for future use
    torch.save(hand_landmarks, f"../data/hand_landmarks.pth")





# Print sample images
if __name__ == "__main__":

    # Setup training data for resnet18
    train_loader, test_loader_aug, test_loader_clean = init_resnet18()

    # Setup training data for landmark classifier
    lm_path = torch.load('../data/hand_landmarks.pth')

    # Only run if no save exists
    if not lm_path:
        init_lmc()



