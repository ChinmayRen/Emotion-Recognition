# Emotion-Recognition
This is a CNN based model trained to predict facial emotions in real-time.


This project implements a Convolutional Neural Network (CNN) for facial emotion detection using Keras and TensorFlow. The model is trained on grayscale images of faces to classify them into seven emotion categories: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral.

Table of Contents
Installation
Dataset
Model Architecture
Training
Evaluation
Results
Usage
Installation
To run this project, you need to have Python installed along with the necessary libraries. You can install the required packages using:

bash
Copy code
pip install tensorflow matplotlib seaborn scikit-learn
Dataset
The dataset used for training and testing should be structured in a directory format as follows:

bash
Copy code
train/
    class_1/
        img1.jpg
        img2.jpg
        ...
    class_2/
        img1.jpg
        img2.jpg
        ...
    ...
test/
    class_1/
        img1.jpg
        img2.jpg
        ...
    class_2/
        img1.jpg
        img2.jpg
        ...
train/: Directory containing the training images, organized into subdirectories for each class.
test/: Directory containing the testing images, organized into subdirectories for each class.
Replace class_1, class_2, etc., with the actual class names, such as Anger, Happiness, etc.

Model Architecture
The model architecture consists of several convolutional layers, batch normalization, max pooling, and dropout layers. The model is designed to classify images into one of the seven emotion categories using the following layers:

Convolutional Layers with ReLU activation
Batch Normalization
Max Pooling
Dropout
Dense Layers
Softmax Activation for the output layer
Training
The model is trained using the following setup:

Image Augmentation: Applied width and height shifts, horizontal flips, and rescaling.
Optimizer: Adam optimizer with a learning rate of 0.0001.
Loss Function: Categorical Crossentropy.
Batch Size: 64
Epochs: 50
Checkpointing: Save the best model based on validation accuracy.
Training Command
python
Copy code
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint_callback]
)
Evaluation
The model's performance is evaluated using a confusion matrix and plots for training and validation accuracy and loss.

Evaluation Command
python
Copy code
validation_pred_probs = model.predict(validation_generator)
confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
Results
Training and validation accuracy and loss are plotted for each epoch. The confusion matrix is generated to visualize the model's performance across different emotion classes.

Usage
Prepare the Dataset: Ensure the dataset is structured as mentioned in the Dataset section.
Run the Training Script: Execute the script to start training the model.
Evaluate the Model: Use the evaluation script to generate plots and the confusion matrix.
Saving Results
The following files are saved during training and evaluation:

model_weights.h5: Best model weights based on validation accuracy.
EmoDetector.h5: Final trained model.
trainingloss.png: Plot of training and validation loss.
trainingaccuracy.png: Plot of training and validation accuracy.
confusionmatrix.png: Confusion matrix of the model's predictions.
Contributing
If you have suggestions or improvements, please feel free to submit a pull request or open an issue.
