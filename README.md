# Facial-Emotional-Recognizer

## [Link to Colab](https://colab.research.google.com/drive/1dT5wB7T59Tv99pqu2TkkkRkBzGYuvJD-?usp=sharing)

## [Link to the Kaggle Dataset](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## [Link to Blogpost](https://medium.com/@derrell.downey/applying-transfer-learning-for-facial-emotion-recognition-e1f7e96cdbcc)

# Facial Emotion Recognition

Facial emotion recognition technology has significantly transformed over the past decade, driven by advancements in artificial intelligence and machine learning. This project explores the effectiveness of various convolutional neural network (CNN) architectures in discerning human emotions from facial expressions. The potential applications of this technology range from interactive gaming to psychological research and enhancing user experiences in AI-driven interfaces.

## Project Overview

Facial emotion recognition is an integral component of affective computing. This study focuses on evaluating different deep learning models to classify facial images into one of seven primary emotional states: anger, fear, happiness, sadness, surprise, neutral, and contempt. The emotion 'disgust' was excluded from the dataset to improve model accuracy due to its minimal representation.

## Data Utilization

We employed the ICML Face Data, which consists of grayscale images at 48x48 pixels, each labeled with one of the seven emotions. To prepare the data for training, we implemented various preprocessing steps, including normalization and data augmentation, to enhance the model's robustness and ability to generalize across unseen data.

## Models and Architectures

### 1. Simple CNN
- Consists of three convolutional layers with ReLU activation and max-pooling layers.
- Includes a dense layer for non-linear transformation and a softmax output layer for classification.
- Achieved a training accuracy of 94.85% and a testing accuracy of 52.56%.

### 2. VGG16
- Renowned for its deep architecture, consisting of 13 convolutional layers interspersed with five max-pooling layers.
- Implemented dropout techniques to prevent overfitting.
- The model struggled with test accuracy, predominantly classifying images as 'Happy'.

### 3. ResNet101
- Features residual connections to handle the vanishing gradient problem.
- Comprises 101 layers, facilitating the training of deep models.
- Achieved a testing accuracy of 48%, with a bias towards classifying images as 'Happy' or 'Sad'.

### 4. MobileNetV2
- Utilizes depthwise separable convolutions to reduce computational cost while maintaining performance.
- Slightly more accurate than ResNet101, with a testing accuracy of 51%.

### 5. DenseNet169
- Employs dense connections between layers to overcome the vanishing gradient problem.
- Achieved the highest testing accuracy of 61% among all models evaluated.

## Data Augmentation

To address overfitting and the imbalanced dataset, we implemented data augmentation techniques, including mirroring images to equalize the training data across different emotion classes. This approach helped improve the testing accuracy of the Simple CNN and DenseNet models.

## Results and Analysis

Confusion matrices for each model were analyzed to evaluate performance. DenseNet169 emerged as the most effective model, achieving the highest accuracy and demonstrating resilience against the vanishing gradient problem. However, despite attempts to enhance model performance through data augmentation, significant improvements were not observed, highlighting the complexity of optimizing deep learning models for facial emotion recognition.

## Concluding Remarks

This project underscores the potential and challenges of using CNN architectures for facial emotion recognition. While DenseNet169 showed the most promise, ongoing research and optimization are necessary to further enhance model accuracy and generalization.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/facial-emotion-recognition.git
    cd facial-emotion-recognition
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the dataset:
    - Download the ICML Face Data and place it in the `data` directory.

2. Train the model:
    ```bash
    python train.py --model simple_cnn
    ```

3. Evaluate the model:
    ```bash
    python evaluate.py --model simple_cnn
    ```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on GitHub or contact me at [your.email@example.com](mailto:derrell.downey@gmail.com).

