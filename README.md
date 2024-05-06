# Facial Emotion Detection using Deep Learning
This project is focused on the development of a facial expression detection system. The system utilizes a convolutional neural network (CNN) to recognize and classify different human emotions based on facial expressions. The model is trained on a dataset of images labeled with seven emotion categories: angry, disgust, fear, happy, neutral, sad, and surprise.

Facial expression recognition dataset from kaggle : https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

## Dataset
The dataset is organized into two main directories:

- ``` images/train: ``` 
     Contains the training images.
- ``` images/test: ``` 
     Contains the testing images.
  
Each directory has subdirectories named after the emotion labels and each subdirectory contains the respective images.
## Model Architecture
The model architecture is a Sequential Convolutional Neural Network (CNN) consisting of four convolutional layers with 3x3 kernel sizes and ReLU activation functions. These layers progressively increase the number of filters from 128 to 512. MaxPooling layers are used to reduce spatial dimensions, while Dropout layers with rates between 0.3 and 0.4 prevent overfitting. 

A Flatten layer prepares the data for two dense layers with 512 and 256 neurons and ReLU activation. The final layer, with 7 neurons and softmax activation, outputs probabilities for 7 emotion categories, facilitating emotion recognition from input images.

## Training and Performance

During the training phase, the model is compiled using the "Adam optimizer" and categorical crossentropy loss function. It undergoes training for 100 epochs with a batch size of 128 and the training process also includes a validation step using the test dataset.

The model’s performance is evaluated based on its accuracy metric, which is the proportion of correctly predicted images over the total number of images in the test set. 
At the final epoch, the model achieved an accuracy of **75.6%** on the test set.

## Usage of the Trained Model
The trained model is saved as ```emotiondetector.h5``` and can be loaded for real-time emotion detection. A separate script, ```realtimedetection.py```, utilizes the model to predict emotions from live webcam feed.

## Example Prediction

```python
image = 'images/train/happy/25.jpg'
print("Original image of an Happy face")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("Model prediction is:", pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')
