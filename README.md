# Lab Equipment Segmentation Project

This project focuses on segmenting images of equipment from chemistry and medical laboratories using a Mask R-CNN model. It utilizes the LabPics dataset for training and testing.

## Dataset

The project uses the LabPics dataset, which contains annotated images for both materials and vessels in chemistry and medical labs. The dataset can be downloaded from:

[LabPics Dataset on Zenodo](https://zenodo.org/record/4736111#.YsXzLIRByUk)

## Project Structure

The project consists of two main Python scripts:

1. `Weight_Train.py`: Used for training the Mask R-CNN model.
2. `Segmentation_Test.py`: Used for testing the trained model on new images.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy

## Training the Model

To train the model, run the `Weight_Train.py` script. This script:

- Loads images and masks from the LabPics Chemistry dataset
- Initializes a Mask R-CNN model with a ResNet-50 backbone
- Trains the model for 10,000 iterations
- Saves the model weights every 500 iterations

Make sure to update the `trainDir` variable with the correct path to your training data.

## Testing the Model

To test the trained model, use the `Segmentation_Test.py` script. This script:

- Loads a trained model
- Processes a single input image
- Displays the original image alongside the segmentation results

Update the `imgPath` variable with the path to your test image.

## Model Architecture

The project uses a Mask R-CNN model with a ResNet-50 backbone, pre-trained on COCO dataset and fine-tuned on the LabPics dataset.

## Notes

- The training script uses a batch size of 2 and resizes images to 600x600 pixels.
- The test script applies a confidence threshold of 0.8 for displaying segmentation results.
- Make sure to adjust file paths according to your local setup.

## Acknowledgements

This project utilizes the LabPics dataset. Please cite the dataset appropriately in any derived works:

```bibtex
@dataset{sagi_eppel_2021_4736111,
  author       = {Sagi Eppel and
                  Haoping Xu and
                  Alan Aspuru-Guzik and
                  Mor Bismuth},
  title        = {{LabPics dataset for visual understanding of 
                   Medical and Chemistry Labs}},
  month        = may,
  year         = 2021,
  publisher    = {Zenodo},
  version      = 2,
  doi          = {10.5281/zenodo.4736111},
  url          = {https://doi.org/10.5281/zenodo.4736111}
}
