# Kannada-MNIST
Image classification competition at Kaggle (https://www.kaggle.com/c/Kannada-MNIST/overview/description)

# Describtion
The goal of this competition is to provide a simple extension to the classic MNIST competition we're all familiar with. Instead of using Arabic numerals, it uses a recently-released dataset of Kannada digits.

Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The language has roughly 45 million native speakers and is written using the Kannada script. 
(https://en.wikipedia.org/wiki/Kannada)

# Soluiton
In architecture.ipynb file you can see reason why I have chosen given CNN structure. It is motivated by manny sources. 
Training is executed on Image Generators provided by keras to achieve best volatility in the data. With this setup one epoch last approximately 100s - 120s on my machine.

My best resut in training was:

loss: 0.0356 - accuracy: 0.9893; on training data

loss: 0.0393 - accuracy: 0.9939; on validation data

This model has 0.9786 accuracy on Kaggle test dataset.
