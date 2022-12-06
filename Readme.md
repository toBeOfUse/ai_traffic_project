## The Problem

"Write a program using Tensorflow, Keras, Pytorch which can recognize a traffic light, a moving object, a moving object while traffic light is red, and a moving object while traffic light is green."

This problem is not entirely sensible since Pytorch is a competitor/replacement for TensorFlow and it only seems to makes sense to use one or the other. It seems best to just use TensorFlow, and Keras, which is a high-level API that is included with it. We will also use Tensorflow Hub which contains pre-trained models in a format that can be loaded in TensorFlow.

What we are doing is training a neural network on a multi-class, multi-label classification problem; multi-class means there is more than on valid label and multi-label means more than one valid label is possible per input, which is true for us because one image may need to receive a label for a traffic light and another label for another object. The ideal way to do image recognition seems to be with transfer learning, meaning that the first layers in the model are copied from another that was trained on a gigantic set of images, such as AlexNet (older, suggested by the paper #10) or ImageNet (newer, many TensorFlow-compatible files available.) This project currently uses MobileNet v3, which is a smaller version of the ImageNet network. Specifically, the form of MobileNet v3 used is the layers used for feature vector creation. Then, multiple dense hidden layers of neurons are used to get classification output based on the vectors.

For our training data, we are using the [udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) self-driving car dataset. This consists of 15,000 images taken through the windshield of a car with accompaying labels that identify the features found in the images. We transform these labels into a multi-hot vector representation for each image that TensorFlow neural networks typically use. These vectors have either a 0 or 1 for each label in our list of labels, depending on if the feature isn't or is present in the image.

We can load the images using the Keras function `image_dataset_from_directory`. This function also takes care of resizing the images to the input size expected by the initial MobileNet layers. The pixel values from the images are used as the input variables to the neural network; to further preprocess our image data, we must map our image pixel values to the range [0, 1]. The image dataset is then randomly split into data used for training and data used for validation.

We can then create our model using the layer types built into Keras. We use Dense layers, which are fully-connected hidden layers of neurons, and BatchNormalization layers, which normalize data that passes through them on the the way to the next layer. The Dense layers use the popular ReLU activation function to fire if their input is greater than 0. The output layer, which is a dense layer with one neuron for each of the potential output labels, uses the sigmoid activation function to get a probability in the range [0, 1] for each label. The BatchNormalization layer is needed to avoid saturating the sigmoid output layer activation function with large values that all get mapped to 1.

The model is trained with the binary cross-entropy loss function for evaluating the correctness of a model where the outputs should be between one and zero. It is trained over epochs; during each epoch, the entire input training data is run through the model with steps taken to minimize the error, and multiple epochs means that the entire training data is used multiple times. After each epoch, we have output that looks like this:

```
loss: 0.1082 - val_loss: 0.1905
```

The value for loss indicates the observed error during the training session for that epoch. After the training, the data that was set aside for validation is used to get another error measurement (this one is labeled val_loss, for validation loss.) During training, loss might have been minimized on the cost of overfitting to the training data, which is why the loss on separate validation data must be measured also to get an accurate picture of the model's performance.

After training, which takes a long time, TensorFlow has the ability to save the state of the trained model into files that may be reloaded later. In this way, the trained model can be stored and brought back to life in order to make classifications based on input later even long after the training session.

One example of this kind of problem that seems very good is [here](https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb)

## Notes on libraries

TensorFlow is a low-level machine learning library; Keras is a higher-level API built on top of it. PyTorch is a competitor to TensorFlow that is more often used in academic contexts.

In machine learning, a preliminary output layer can be called a logits layer; this layer produces the outputs that are fed into softmax to get the classification probabilities (and then maybe argmax to get a single prediction result.) The output values themselves are sometimes called logits. The term "logit" comes from statistics but its meaning has shifted within the deep learning scene.

Keras models consist of a set of neural network layers (generally a sequence), instantiated with the high-level classes available in the Keras API, which are compiled into the final model given an optimization algorithm, a loss function, and judgement metrics to be used during training. You can then fit the model to some training data and evaluate it with some testing data. With the addition of a softmax layer to convert logits into probabilities, you can ask for a "prediction" given an input data.

## Existing research the professor sent to us (PDFs are in papers folder)

Paper #10 starts by summarizing the path some of the research into CNNs for image recognition has taken so far. It says it's easy for CNNs to overfit, given a generally limited dataset, and lists numerous approaches that have been taken to avoid this, but it thinks it has come up with a better one. It proposes a strategy based on AlexNet, with a well-trained feature extractor and carefully optimized convolutional layers.

AlexNet uses a feature extractor that includes convolutional layers, activation function units, and pooling layers. The convolutional layers use kernels and padding to avoid shrinking the data. Pooling layers are what is used to sample and simplify the output of the upper convolutional layers. The activation function is the ReLU function.

The classifier in AlexNet is a fully connected feed forward network. Dropout layers are introduced to avoid overfitting: some neurons are randomly turned off to effectively divide the network into subnetworks which must all be able to work independently.

Transfer learning is used, meaning that AlexNet's trained feature extractor network is stolen and hooked up to a new classifier for this paper's purpose. The first fully connected layer of AlexNet is changed into a convolutional layer; besides that, not many details are given.
