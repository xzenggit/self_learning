

## Notes for [Deep Learning with Python](https://livebook.manning.com/book/deep-learning-with-python)

### Ch1. What is deep learning?

#### What makes deep learning different?

The primary reason deep learning took off so quickly is that it offered better performance on many problems. But that’s not the only reason. Deep learning also makes problem-solving much easier, because it completely automates what used to be the most crucial step in a machine-learning workflow: feature engineering. With deep learning, you learn all features in one pass rather than having to engineer them yourself. This has greatly simplified machine-learning workflows, often replacing sophisticated multistage pipelines with a single, simple, end-to-end deep-learning model.

 What is transformative about deep learning is that it allows a model to learn all layers of representation jointly, at the same time, rather than in succession (greedily, as it’s called). With joint feature learning, whenever the model adjusts one of its internal features, all other features that depend on it automatically adapt to the change, without requiring human intervention.

These are the two essential characteristics of how deep learning learns from data: the incremental, layer-by-layer way in which increasingly complex representations are developed, and the fact that these intermediate incremental representations are learned jointly, each layer being updated to follow both the representational needs of the layer above and the needs of the layer below. Together, these two properties have made deep learning vastly more successful than previous approaches to machine learning.

#### Why deep learning? Why now?

The key ideas of deep learning were already well understood in 1989. Three main reasons it becomes popular now: hardware, dataset and benchmarks, and algorithmic advances.
In terms of algorithmic advances, now we have
* Better activation functions for neural layers
* Better weight-initialization schemes, starting with layer-wise pretraining, which was quickly abandoned
* Better optimization schemes, such as RMSProp and Adam
Finally, in 2014, 2015, and 2016, even more advanced ways to help gradient propagation were discovered, such as batch normalization, residual connections, and depthwise separable convolutions. Today we can train from scratch models that are thousands of layers deep.

Following a scientific revolution, progress generally follows a sigmoid curve: it starts with a period of fast progress, which gradually stabilizes as researchers hit hard limitations, and then further improvements become incremental. Deep learning in 2017 seems to be in the first half of that sigmoid, with much more progress to come in the next few years.

### Ch2. The mathematical building blocks of neural networks

Real word examples of data tensors:
* Vector data— 2D tensors lx pseah (samples, features)
* Timeseries data or sequence data— 3D tensors lk shpea (samples, timesteps, features)
* Images— 4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
* Video— 5D tensors of shape (samples, frames, height, width, channels) or (samples, frames, channels, height, width)

Neural networks consist entirely of chains of tensor operations and that all of these tensor operations are just geometric transformations of the input data. Uncrumpling paper balls is what machine learning is about: finding neat representations for complex, highly folded data manifolds. Deep learning takes the approach of incrementally decomposing a complicated geometric transformation into a long chain of elementary ones, which is pretty much the strategy a human would follow to uncrumple a paper ball. Each layer in a deep network applies a transformation that disentangles the data a little—and a deep stack of layers makes tractable an extremely complicated disentanglement process.

Training loop:
* Draw a batch of training samples x and corresponding targets y.
* Run the network on x (a step called the forward pass) to obtain predictions y_pred.
* Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y.
* Update all weights of the network in a way that slightly reduces the loss on this batch.

### Ch3. Getting started with neural networks

The typical Keras workflow looks just like:
* Define your training data: input tensors and target tensors.
* Define a network of layers (or model) that maps your inputs to your targets.
* Configure the learning process by choosing a loss function, an optimizer, and some metrics to monitor.
* Iterate on your training data by calling the fit() method of your model.

### Ch4. Fundamentals of machine learning

Four types of machine learning:
* Supervised Learning: regression
* Unsuperivised Learning: dimensionality reduction and clustering
* Self-supervised learning: supervised learning without human-annotated labels. e.g. autoencoder
* Reinforcement learning

The universal workflow of machine learning:
* Define the problem and assemble a dataset
* Choose a measure of success
* Decide on an evaluation protocol
* Prepare data (e.g. normalization)
* Develop a model that does better than a baseline
* Scale up: develop a model that overfits
* Regularize your model and tune hyperparameters

To figure out how big a model you’ll need, you must develop a model that overfits. This is fairly easy:
1. Add layers.
2. Make the layers bigger.
3. Train for more epochs.
Always monitor the training loss and validation loss, as well as the training and validation values for any metrics you care about. When you see that the model’s performance
on the validation data begins to degrade, you’ve achieved overfitting. The next stage is to start regularizing and tuning the model, to get as close as possible
to the ideal model that neither underfits nor overfits.

Things you can try for regularization:
* Add dropout
* Try different architectures: add or remove layers.
* Add L1 and/or L2 regularization.
* Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.
* Optionally, iterate on feature engineering: add new features, or remove features that don’t seem to be informative.

Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the model. Repeated just a few times, this is innocuous; but done systematically over many iterations, it will eventually cause your model to overfit to the validation process (even though no model is directly trained on any of the validation data). This makes the evaluation process less reliable.
Once you’ve developed a satisfactory model configuration, you can train your final production model on all the available data (training and validation) and evaluate it one last time on the test set. If it turns out that performance on the test set is significantly worse than the performance measured on the validation data, this may mean either that your validation procedure wasn’t reliable after all, or that you began overfitting to the validation data while tuning the parameters of the model. In this case, you may want to switch to a more reliable evaluation protocol (such as iterated K-fold validation).

Problem type | Last-layer activation | Loss function
:-----------:|:---------------------:|:-------------:
binary classification | sigmoid | binary_crossentropy
multiclass, single-label classification | softmax | categorical_crossentropy
multiclass, multilabel classification | sigmoid | binary_crossentropy
regression to arbitrary values | None | mse
regression to values between 0 and 1 | sigmoid | mse or binary_crossentropy

### Ch5. Deep learning for computer vision

Convnets interesting properties:
* The patterns they learn are translation invariant.
* They can learn spatial hierarchies of patterns

In short, the reason to use downsampling (e.g. MaxPooling) is to reduce the number of feature-map coefficients to process, as well as to induce spatial-filter hierarchies by making successive convolution layers look at increasingly large windows.

You’ll sometimes hear that deep learning only works when lots of data is available. This is valid in part: one fundamental characteristic of deep learning is that it can find interesting features in the training data on its own, without any need for manual feature engineering, and this can only be achieved when lots of training examples are available. This is especially true for problems where the input samples are very highdimensional, like images.

What’s more, deep-learning models are by nature highly repurposable: you can take, say, an image-classification or speech-to-text model trained on a large-scale dataset and reuse it on a significantly different problem with only minor changes.

Convnets used for image classification comprise two parts: they start with a series of pooling and convolution layers, and they end with a densely connected classifier. The first part is called the convolutional base of the model. In the case of convnets, feature extraction consists of taking the convolutional base of a previously trained network, running the new data through it, and training a new classifier on top of the output.

Why only reuse the convolutional base? Could you reuse the densely connected classifier as well? In general, doing so should be avoided. The reason is that the representations learned by the convolutional base are likely to be more generic and therefore more reusable: the feature maps of a convnet are presence maps of generic concepts over a picture, which is likely to be useful regardless of the computer-vision problem at hand. But the representations learned by the classifier will necessarily be specific to the set of classes on which the model was trained—they will only contain information about the presence probability of this or that class in the entire picture. Additionally, representations found in densely connected layers no longer contain any information about where objects are located in the input image: these layers get rid of the notion of space, whereas the object location is still described by convolutional feature maps. For problems where object location matters, densely connected features are largely useless.

Note that the level of generality (and therefore reusability) of the representations extracted by specific convolution layers depends on the depth of the layer in the model. Layers that come earlier in the model extract local, highly generic feature maps (such as visual edges, colors, and textures), whereas layers that are higher up extract more-abstract concepts (such as “cat ear” or “dog eye”). So if your new dataset differs a lot from the dataset on which the original model was trained, you may be better off using only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

A list of image classification models:
* Xception
* Inception V3
* ResNet50
* VGG16
* VGG19
* MobileNet

Visualize what convets learn
* Visualizing intermediate convnet outputs (intermediate activations)—Useful for
understanding how successive convnet layers transform their input, and for getting
a first idea of the meaning of individual convnet filters.
* Visualizing convnets filters—Useful for understanding precisely what visual pattern
or concept each filter in a convnet is receptive to.
* Visualizing heatmaps of class activation in an image—Useful for understanding
which parts of an image were identified as belonging to a given class, thus allowing
you to localize objects in images.

### Ch6. Deep learning for text and sequences

There are two ways to obtain word embeddings:
* Learn word embeddings jointly with the main task you care about (such as document
classification or sentiment prediction). In this setup, you start with random
word vectors and then learn word vectors in the same way you learn the
weights of a neural network.
* Load into your model word embeddings that were precomputed using a different
machine-learning task than the one you’re trying to solve. These are called
pretrained word embeddings.

An RNN is a for loop that reuses quantities computed during the previous iteration of the loop, nothing more. SimpleRNN has a major issue: although it should theoretically be able to retain at time t information about inputs seen many timesteps before, in practice, such long-term dependencies are impossible to learn. This is due to the vanishing gradient problem, an effect that is similar to what is observed with non-recurrent networks (feedforward networks) that are many layers deep: as you keep adding layers to a network, the network eventually becomes untrainable. The theoretical reasons for
this effect were studied by Hochreiter, Schmidhuber, and Bengio in the early 1990s.The LSTM and GRU layers are designed to solve this problem.

Just keep in mind what the LSTM cell is meant to do: allow past information to be reinjected at a later time, thus fighting the vanishing-gradient problem.
Three RNN techniques:
* Recurrent dropout—This is a specific, built-in way to use dropout to fight overfitting
in recurrent layers.
* Stacking recurrent layers—This increases the representational power of the network
(at the cost of higher computational loads).
* Bidirectional recurrent layers—These present the same information to a recurrent
network in different ways, increasing accuracy and mitigating forgetting issues.

RNNs are notably order dependent, or time dependent: they process the timesteps
of their input sequences in order, and shuffling or reversing the timesteps can completely
change the representations the RNN extracts from the sequence. This is precisely
the reason they perform well on problems where order is meaningful, such as
the temperature-forecasting problem. A bidirectional RNN exploits the order sensitivity
of RNNs: it consists of using two regular RNNs, such as the GRU and LSTM layers
you’re already familiar with, each of which processes the input sequence in one direction
(chronologically and antichronologically), and then merging their representations.
By processing a sequence both ways, a bidirectional RNN can catch patterns that
may be overlooked by a unidirectional RNN.

Such 1D convnets can be competitive with RNNs on certain sequence-processing
problems, usually at a considerably cheaper computational cost. Recently, 1D convnets,
typically used with dilated kernels, have been used with great success for audio
generation and machine translation. In addition to these specific successes, it has long
been known that small 1D convnets can offer a fast alternative to RNNs for simple tasks
such as text classification and timeseries forecasting.

One strategy to combine the speed and lightness of convnets with the order-sensitivity of RNNs is to use a 1D convnet as a preprocessing step before an RNN. This is especially beneficial when you’re dealing
with sequences that are so long they can’t realistically be processed with RNNs, such as sequences with thousands of steps. The convnet will turn the long input sequence into
much shorter (downsampled) sequences of higher-level features. This sequence of extracted features then becomes the input to the RNN part of the network.

### Ch7. Advanced deep learning best practices

Keras Functional API: treat layers as functions
* Go beyond Sequential model
* Layer weight sharing
* Models as layers

Here are some examples of ways you can use callbacks:
* Model checkpointing—Saving the current weights of the model at different points
during training.
* Early stopping—Interrupting training when the validation loss is no longer
improving (and of course, saving the best model obtained during training).
* Dynamically adjusting the value of certain parameters during training—Such as the
learning rate of the optimizer.
* Logging training and validation metrics during training, or visualizing the representations
learned by the model as they’re updated—The Keras progress bar that you’re
familiar with is a callback!

TensorBoard gives you access to several neat features, all in your browser:
* Visually monitoring metrics during training
* Visualizing your model architecture
* Visualizing histograms of activations and gradients
* Exploring embeddings in 3D

Batch normalization is a type of layer (BatchNormalization in Keras) introduced in 2015 by Ioffe and Szegedy; it can adaptively normalize data even as the mean and
variance change over time during training.

What if I told you that there’s a layer you can use as a drop-in replacement for Conv2D
that will make your model lighter (fewer trainable weight parameters) and faster
(fewer floating-point operations) and cause it to perform a few percentage points better
on its task? That is precisely what the depthwise separable convolution layer does
(SeparableConv2D). This layer performs a spatial convolution on each channel of its
input, independently, before mixing output channels via a pointwise convolution (a
1 × 1 convolution). This is equivalent to separating the learning
of spatial features and the learning of channel-wise features, which makes a lot of
sense if you assume that spatial locations in the input are highly correlated, but different
channels are fairly independent. It requires significantly fewer parameters and
involves fewer computations, thus resulting in smaller, speedier models. And because
it’s a more representationally efficient way to perform convolution, it tends to learn
better representations using less data, resulting in better-performing models.

#### Hyperparameter tuning
You need to explore the space of possible decisions automatically, systematically,
in a principled way. You need to search the architecture space and find the bestperforming
ones empirically. That’s what the field of automatic hyperparameter optimization
is about: it’s an entire field of research, and an important one.
The process of optimizing hyperparameters typically looks like this:
1 Choose a set of hyperparameters (automatically).
2 Build the corresponding model.
3 Fit it to your training data, and measure the final performance on the validation
data.
4 Choose the next set of hyperparameters to try (automatically).
5 Repeat.
6 Eventually, measure performance on your test data.

Many different techniques are possible: Bayesian optimization, genetic algorithms, simple random search, and so on. Often, it turns out that random
search (choosing hyperparameters to evaluate at random, repeatedly) is the best solution,
despite being the most naive one. But one tool I have found reliably better than
random search is [Hyperopt](https://github.com/hyperopt/hyperopt), a Python
library for hyperparameter optimization that internally uses trees of Parzen estimators
to predict sets of hyperparameters that are likely to work well. Another library called
[Hyperas](https://github.com/maxpumperla/hyperas) integrates Hyperopt for use
with Keras models.

#### Model ensembling
Ensembling consists of pooling together the predictions of a set of different models, to produce better predictions.
A smarter way to ensemble classifiers is to do a weighted average, where the
weights are learned on the validation data—typically, the better classifiers are given a
higher weight, and the worse classifiers are given a lower weight. The key to making ensembling work is the diversity of the set of classifiers. Diversity
is strength. For this reason, you should ensemble models that are as good as possible while being
as different as possible. This typically means using very different architectures or even
different brands of machine-learning approaches.

One thing I have found to work well in practice—but that doesn’t generalize to every problem domain—is the use of an ensemble of tree-based methods (such as random
forests or gradient-boosted trees) and deep neural networks.

### Ch8. Generative deep learning

#### Neural style transfer
As you already know, activations from earlier layers in a network contain local information
about the image, whereas activations from higher layers contain increasingly global,
abstract information. Therefore, you’d expect the content of an image, which is more global and
abstract, to be captured by the representations of the upper layers in a convnet. A good candidate for content loss is thus the L2 norm between the activations of
an upper layer in a pretrained convnet, computed over the target image, and the activations
of the same layer computed over the generated image. This guarantees that, as
seen from the upper layer, the generated image will look similar to the original target
image. Assuming that what the upper layers of a convnet see is really the content of
their input images, then this works as a way to preserve image content.

The content loss only uses a single upper layer, but the style loss as defined by Gatys
et al. uses multiple layers of a convnet: you try to capture the appearance of the stylereference
image at all spatial scales extracted by the convnet, not just a single scale.
For the style loss, Gatys et al. use the Gram matrix of a layer’s activations: the inner
product of the feature maps of a given layer. This inner product can be understood as
representing a map of the correlations between the layer’s features. These feature correlations
capture the statistics of the patterns of a particular spatial scale, which empirically
correspond to the appearance of the textures found at this scale. Hence, the style loss aims to preserve similar internal correlations within the activations
of different layers, across the style-reference image and the generated image. In
turn, this guarantees that the textures found at different spatial scales look similar
across the style-reference image and the generated image.

#### Generative Adversarial Networks (GANs)
A GAN is made of two parts:
* Generator network—Takes as input a random vector (a random point in the
latent space), and decodes it into a synthetic image
* Discriminator network (or adversary)—Takes as input an image (real or synthetic),
and predicts whether the image came from the training set or was created by
the generator network.

Remarkably, a GAN is a system where the optimization minimum isn’t fixed, unlike in
any other training setup you’ve encountered in this book. Normally, gradient descent
consists of rolling down hills in a static loss landscape. But with a GAN, every step
taken down the hill changes the entire landscape a little. It’s a dynamic system where
the optimization process is seeking not a minimum, but an equilibrium between two
forces. For this reason, GANs are notoriously difficult to train—getting a GAN to work
requires lots of careful tuning of the model architecture and training parameters.

To recapitulate, this is what the training loop looks like
schematically. For each epoch, you do the following:
1 Draw random points in the latent space (random noise).
2 Generate images with generator using this random noise.
3 Mix the generated images with real ones.
4 Train discriminator using these mixed images, with corresponding targets:
either “real” (for the real images) or “fake” (for the generated images).
5 Draw new random points in the latent space.
6 Train gan using these random vectors, with targets that all say “these are real
images.” This updates the weights of the generator (only, because the discriminator
is frozen inside gan) to move them toward getting the discriminator to
predict “these are real images” for generated images: this trains the generator
to fool the discriminator.

### Ch9. Conclusions

In deep learning, everything is a vector: everything is a point in a geometric space.
Model inputs (text, images, and so on) and targets are first vectorized: turned into an
initial input vector space and target vector space. Each layer in a deep-learning model
operates one simple geometric transformation on the data that goes through it.
Together, the chain of layers in the model forms one complex geometric transformation,
broken down into a series of simple ones. This complex transformation attempts
to map the input space to the target space, one point at a time. This transformation is
parameterized by the weights of the layers, which are iteratively updated based on how
well the model is currently performing. A key characteristic of this geometric transformation
is that it must be differentiable, which is required in order for us to be able to
learn its parameters via gradient descent. Intuitively, this means the geometric morphing
from inputs to outputs must be smooth and continuous—a significant constraint.

The entire process of applying this complex geometric transformation to the input
data can be visualized in 3D by imagining a person trying to uncrumple a paper ball:
the crumpled paper ball is the manifold of the input data that the model starts with.
Each movement operated by the person on the paper ball is similar to a simple geometric
transformation operated by one layer. The full uncrumpling gesture sequence
is the complex transformation of the entire model. Deep-learning models are mathematical
machines for uncrumpling complicated manifolds of high-dimensional data.

That’s the magic of deep learning: turning meaning into vectors, into geometric
spaces, and then incrementally learning complex geometric transformations that map
one space to another. All you need are spaces of sufficiently high dimensionality in
order to capture the full scope of the relationships found in the original data.

The whole thing hinges on a single core idea: that meaning is derived from the pairwise
relationship between things (between words in a language, between pixels in an image,
and so on) and that these relationships can be captured by a distance function.

Here’s a quick overview of the mapping
between input modalities and appropriate network architectures:
* Vector data—Densely connected network (Dense layers).
* Image data—2D convnets.
* Sound data (for example, waveform)—Either 1D convnets (preferred) or RNNs.
* Text data—Either 1D convnets (preferred) or RNNs.
* Timeseries data—Either RNNs (preferred) or 1D convnets.
* Other types of sequence data—Either RNNs or 1D convnets. Prefer RNNs if data
ordering is strongly meaningful (for example, for timeseries, but not for text).
* Video data—Either 3D convnets (if you need to capture motion effects) or a
combination of a frame-level 2D convnet for feature extraction followed by
either an RNN or a 1D convnet to process the resulting sequences.
* Volumetric data—3D convnets.

Densely Connected Networks:
* the units of a Dense layer are connected to every other unit
* To perform binary classification, end your stack of layers with a Dense layer with a single unit and a sigmoid activation, and use binary_crossentropy as the loss.
* To perform single-label categorical classification (where each sample has exactly one class, no more), end your stack of layers with a Dense layer with a number of units equal to the number of classes, and a softmax activation. If your targets are one-hot encoded, use categorical_crossentropy as the loss; if they’re integers, use sparse_categorical_crossentropy.
* To perform multilabel categorical classification (where each sample can have several classes), end your stack of layers with a Dense layer with a number of units equal to the number of classes and a sigmoid activation, and use binary_crossentropy as the loss. Your targets should be k-hot encoded
* To perform regression toward a vector of continuous values, end your stack of layers with a Dense layer with a number of units equal to the number of values you’re trying  to predict (often a single one, such as the price of a house), and no activation. Several losses can be used for regression, most commonly mean_squared_error (MSE) and mean_absolute_error (MAE).

Convnets:
* Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor. This results in representations that are translation invariant, making convolution layers highly data efficient and modular.
* The pooling layers let you spatially downsample the data, which is required to keep feature maps to a reasonable size as the number of features grows, and to allow subsequent convolution layers to “see” a greater spatial extent of the inputs.
* Note that it’s highly likely that regular convolutions will soon be mostly (or completely) replaced by an equivalent but faster and representationally efficient alternative: the depthwise separable convolution (SeparableConv2D layer). This is true for 3D, 2D, and 1D inputs. When you’re building a new network from scratch, using depthwise separable convolutions is definitely the way to go. The SeparableConv2D layer can be used as a drop-in replacement for Conv2D, resulting in a smaller, faster network that also performs better on its task.

RNNs:
* In order to stack multiple RNN layers on top of each other, each layer prior to the last layer in the stack should return the full sequence of its outputs (each input timestep
will correspond to an output timestep); if you aren’t stacking any further RNN layers, then it’s common to return only the last output, which contains information
about the entire sequence.

In short, deep-learning models don’t have any understanding of their input—at least,
not in a human sense. Our own understanding of images, sounds, and language is
grounded in our sensorimotor experience as humans. Machine-learning models have
no access to such experiences and thus can’t understand their inputs in a humanrelatable
way. By annotating large numbers of training examples to feed into our models,
we get them to learn a geometric transform that maps data to human concepts on
a specific set of examples, but this mapping is a simplistic sketch of the original model
in our minds—the one developed from our experience as embodied agents.

#### The future of deep learning

At a high level, these are the main directions in which I see promise:
* Models closer to general-purpose computer programs, built on top of far richer primitives
than the current differentiable layers. This is how we’ll get to reasoning and
abstraction, the lack of which is the fundamental weakness of current models.
* New forms of learning that make the previous point possible, allowing models to move
away from differentiable transforms.
* Models that require less involvement from human engineers. It shouldn’t be your job to
tune knobs endlessly.
* Greater, systematic reuse of previously learned features and architectures, such as metalearning
systems using reusable and modular program subroutines.















```python

import numpy as np
import xarray as xr
import dswx.grid as gridu

# s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V0_UHH/1km_daily_precipitation_Germany_20180101_20181231.zarr'
# new_s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V0_UHH/hyperlocal_precipitation_Germany_20180101_20181231_v0.1.1.UHH.zarr'

# s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V0_UHH/1km_daily_precipitation_IL_20180101_20181231.zarr'
# new_s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V0_UHH/hyperlocal_precipitation_IL_20180101_20181231_v0.1.1.UHH.zarr'

# s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V1_MODIS/Germany_GPM_ERA5cbhtcwv_MODISwvcth_Residualstyle_prediction_output_2018.zarr'
# new_s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V1_MODIS/hyperlocal_precipitation_Germany_20180101_20181231_v0.1.1.MODIS.zarr'

s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V1_MODIS/IL_GPM_ERA5cbhtcwv_MODISwvcth_Residualstyle_prediction_output_2018.zarr'
new_s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V1_MODIS/hyperlocal_precipitation_IL_20180101_20181231_v0.1.1.MODIS.zarr'


ds = gridu.read_zarr_from_S3(s3_path)
ds.attrs['institution'] = 'BASF Digital Farming'
ds.attrs['title'] = 'BASF daily downscaled precipitation at 1km resolution'
#ds.attrs['source'] = 'BASF daily precipitation downscaling product at 1km resolution based on UHH elevation correction.'
ds.attrs['source'] = 'BASF daily precipitation downscaling product at 1km resolution based on MODIS cloud property correction.'

ds.precip_adjusted.attrs['units'] = 'mm'
ds.precip_adjusted.attrs['long_name'] = 'adjusted daily accumulated precipitation'
ds.precip_adjusted.attrs['comment'] = 'Dates are UTC calendar days. For visualization purpose, please use precip_raw. For other uses, please use precip_adjusted.'

ds.precip_raw.attrs['units'] = 'mm'
ds.precip_raw.attrs['long_name'] = 'raw daily accumulated precipitation'
ds.precip_raw.attrs['comment'] = 'Dates are UTC calendar days. For visualization purpose, please use precip_raw. For other uses, please use precip_adjusted.'

del ds.precip_adjusted.attrs['unit']
del ds.precip_adjusted.attrs['name']

gridu.save_zarr_to_S3(ds, new_s3_path).


import numpy as np
import xarray as xr
import dswx.grid as gridu

# s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V1_MODIS/hyperlocal_precipitation_IL_20180101_20181231_v0.1.1.MODIS.zarr'
# new_s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/vmodis/hyperlocal_precipitation_IL_20180101_20181231_v0.1.1-modis.zarr'

s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/V0_UHH/hyperlocal_precipitation_IL_20180101_20181231_v0.1.1.UHH.zarr'
new_s3_path = 'basf-weather-dev-projects/downscaling/product_releases/precipitation/vuhh/hyperlocal_precipitation_IL_20180101_20181231_v0.1.1-uhh.zarr'

ds = gridu.read_zarr_from_S3(s3_path)
ds.attrs['version'] = 'v0.1.1-uhh'
#gridu.save_zarr_to_S3(ds, new_s3_path) #, mode='a'
gridu.save_zarr_to_S3(ds, s3_path, mode='a')

```
