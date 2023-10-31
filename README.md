# Image-Classification-Convolutional-Neural-Network
Project for Victoria University of Wellington course Machine Learning Tools &amp; Techniques (COMP309)

The objective of this project was to classify 300x300 images of cherries, strawberries, and tomatoes using a range of neural network techniques. The project followed these key steps:

1. Initial Exploratory Data Analysis (EDA) was performed to ensure that all images conformed to the 300x300 format.

2. The project began by investigating a basic or "vanilla" neural network for image classification.

3. Subsequently, convolutional layers were introduced, and various optimizers, loss functions, and hyperparameters were explored and fine-tuned using cross-validation techniques to enhance model performance.

4. In the final stage of the project, more advanced methods such as data augmentation and transfer learning were considered and investigated to further improve the classification accuracy.

This project's approach demonstrates a comprehensive strategy for image classification, starting from basic techniques and gradually advancing to more sophisticated methods to achieve accurate results.

The project consists of two main components:

1. "EDA & Tuning": This section of the project is a Jupyter notebook that was executed in Google Colab to address computing constraints. It includes code that likely requires organization and cleaning up. The primary goal of this notebook is to perform Exploratory Data Analysis (EDA) and fine-tuning of the neural network model.

2. "Final Model": This section involves two Python files:

   a. "train.py": This script can be executed to train a neural network model and produce a file named "model.pth." The model is trained on the provided dataset.

   b. "test.py": This script is designed to use the trained "model.pth" to classify images. Images for classification should be placed in the "testdata" directory.

For access to the data, you can visit the provided Google Drive link: [Data Available](https://drive.google.com/drive/folders/1tHmxtfVdePLT4s5SOQwduDoxJlZ7MyVh?usp=drive_link). This link should contain the necessary datasets for your project.

