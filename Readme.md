## The Problem

"Write a program using Tensorflow, Keras, Pytorch which can recognize a traffic light, a moving object, a moving object while traffic light is red, and a moving object while traffic light is green."

This problem is not entirely sensible since Pytorch is a competitor/replacement for TensorFlow and it only seems to makes sense to use one or the other. It seems best to just use TensorFlow, and Keras, which is a high-level API that is included with it.

What we are doing is training a neural network on a multi-label classification problem; multi-label means more than one valid label is possible per input. The ideal way to do this seems to be with transfer learning, meaning that the first layers in the model are copied from another that was trained on a gigantic set of images, such as AlexNet (older, suggested by the paper) or ImageNet (newer, many TensorFlow-compatible files available.) One example of this kind of problem that seems very good is [here](https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb).

We need data to train our model on. The [udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) self-driving car dataset seems best; there are others listed in sources.txt. create_categories.py will read its labels and output them to a csv in a multi-hot vector form that is suitable for TensorFlow; create_datasets.py will load its images with the vectors in that csv and create TensorFlow Dataset objects for training and validation. This gets us to the "Model Building" section of the example linked above.

## Notes on libraries

TensorFlow is a low-level machine learning library; Keras is a higher-level API built on top of it. PyTorch is a competitor to TensorFlow that is more often used in academic contexts.

In machine learning, a preliminary output layer can be called a logits layer; this layer produces the outputs that are fed into softmax to get the classification probabilities (and then maybe argmax to get a single prediction result.) The output values themselves are sometimes called logits. The term "logit" comes from statistics but its meaning has shifted within the deep learning scene.

Keras models consist of a set of neural network layers (generally a sequence), instantiated with the high-level classes available in the Keras API, which are compiled into the final model given an optimization algorithm, a loss function, and judgement metrics to be used during training. You can then fit the model to some training data and evaluate it with some testing data. With the addition of a softmax layer to convert logits into probabilities, you can ask for a "prediction" given an input data.

## Existing research the professor sent to us (PDFs are in papers folder)

Paper #10 starts by summarizing the path some of the research into CNNs for image recognition has taken so far. It says it's easy for CNNs to overfit, given a generally limited dataset, and lists numerous approaches that have been taken to avoid this, but it thinks it has come up with a better one. It proposes a strategy based on AlexNet, with a well-trained feature extractor and carefully optimized convolutional layers.

AlexNet uses a feature extractor that includes convolutional layers, activation function units, and pooling layers. The convolutional layers use kernels and padding to avoid shrinking the data. Pooling layers are what is used to sample and simplify the output of the upper convolutional layers. The activation function is the ReLU function.

The classifier in AlexNet is a fully connected feed forward network. Dropout layers are introduced to avoid overfitting: some neurons are randomly turned off to effectively divide the network into subnetworks which must all be able to work independently.

Transfer learning is used, meaning that AlexNet's trained feature extractor network is stolen and hooked up to a new classifier for this paper's purpose. The first fully connected layer of AlexNet is changed into a convolutional layer; besides that, not many details are given.
