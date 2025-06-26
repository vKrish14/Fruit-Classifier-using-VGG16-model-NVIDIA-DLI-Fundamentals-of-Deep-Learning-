# Fruit-Classifier-using-VGG16-model-NVIDIA-DLI-Fundamentals-of-Deep-Learning
This project trains a deep learning model to classify images of fresh and rotten fruits using transfer learning with a VGG16 model pre-trained on ImageNet.

## Dataset

The dataset is organized into training and validation folders:

data/fruits/
├── train/
│ ├── freshapples/
│ ├── freshbanana/
│ ├── freshoranges/
│ ├── rottenapples/
│ ├── rottenbanana/
│ └── rottenoranges/
└── valid/
├── freshapples/
├── freshbanana/
├── freshoranges/
├── rottenapples/
├── rottenbanana/
└── rottenoranges/


Each folder contains `.png` images for one class.

## Objective

Build a multi-class image classifier that can identify six fruit categories and reach at least **92% accuracy** on the validation set.

## Method

- Used VGG16 pretrained on ImageNet
- Added a custom classifier with a final output layer of 6 neurons
- Applied transfer learning (freezing base layers) and later fine-tuning
- Used data augmentation to improve generalization
- Trained using `CrossEntropyLoss` and `Adam` optimizer
- Runs on GPU if available (via CUDA)

## How to Run

1. Place the dataset in the `data/fruits/` directory as shown above.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

    Run the training script:

python main.py

Evaluate the trained model (if provided):

    from run_assessment import run_assessment
    run_assessment(my_model)

Notes

    Batch size: 32

    Input size: 224x224 RGB

    Total classes: 6

    Target accuracy: ≥92%

License

This project is provided under the MIT License.
