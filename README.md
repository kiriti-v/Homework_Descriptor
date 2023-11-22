		
# Student Homework Description Generator

Author: Venkata (Kiriti) Vundavilli
Email: kiriti.v@icloud.com

I went with a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN) to generate the descriptions as this is a tried and tested approach for any preliminary result - which is what I was aiming for. The CNN acts as an encoder, extracting features from the images, while the RNN serves as a decoder, generating textual descriptions based on these features. There are of course different model choices that can be made, which are discussed in the **Potential Improvements** section below.

## Features

- Feature extraction from images using the VGG16 model.
- Preprocessing and cleaning of caption text data.
- Training a deep learning model to generate captions.
- Evaluating the model using BLEU scores.
- Generating captions for new images.

## Dependencies

To run the code, ensure that the following Python libraries are installed:

- os
- pickle
- numpy
- tqdm
- tensorflow.keras
- pandas
- nltk

## Dataset

The dataset used is private and but you can use this with potentially any image+caption dataset. Just make sure your directory is structured like so:

- Base directory for images: `/project/input/dataset/images`
- Captions CSV file: `/project/input/dataset/descriptions.csv`

## Usage

1. **Feature Extraction**: The code extracts features from each image in the dataset using the VGG16 model and stores them in a pickle file.

2. **Load and Preprocess Text Data**: Captions are read from a `.txt` file and a `.csv` file, cleaned, and preprocessed to add start and end sequence tokens.

3. **Train-Test Split**: The dataset is split into training and testing sets with a default 90-10 split.

4. **Model Training**: A model is created by merging features from the CNN and the RNN and is then trained on the image-caption pairs.

5. **Caption Generation**: The trained model can generate captions for a given image.

6. **Evaluation**: The model's performance is evaluated using BLEU scores on the test dataset.

7. **Visualization**: The actual and predicted descriptions for student work can be displayed alongside the image itself.

8. **Testing with a Real Image**: The trained model can be used to generate a caption a test image from the student work dataset.

## Model Architecture

The model uses a combination of neural network architectures:

- VGG16 for feature extraction from images.
- A dense layer on top of the VGG16 features.
- An LSTM network for processing the sequences.

## Training the Model

The model is trained for 20 epochs, with checkpointing to save the best model.

## Generating Captions

Run the function `generate_caption(image_name)` to generate and visualize the caption for a specific image in the dataset.

## Shortcomings and Potential Improvements

1. The predictor seems to identify most geometric shapes and student work examples accurately but seems to struggle to put 2 and 2 together and describe the problem, or identify handwriting.

2. Using an image-feature extractor like Inception + Transformer decoder for text, or an end-to-end transformer architecture such as VisionTransformer (ViT) for image features and GPT2 for decoding text - and training on the 4516 student work images dataset can potentially improve performance and get us closer to the goal of aiding teachers in giving feedback automatically.
