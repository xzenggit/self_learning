
## Tricks to train a deep CNN
* Use ReLU activation
* Use He et al. initialization
* Try to add Batch Normalization or dropout
* Try to augment your training data

Batch normalization controls mean and variance of outputs before activations.

## Learning new tasks with pre-trained CNNs

Small dataset:
* ImageNet domain: Train last MLP layers
* Not similar to ImageNet: Collect more data

Big dataset:
* ImageNet domain: Fine-tuning of deeper layers
* Not similar to ImageNet: Train from scratch

## Unsupervised Learning

* Find most relevant features
* Compress information
* Retrieve similar objects
* Generate new dataset samples
* Explore high-dimension data

Autoencoders: Take data in some original space, and project data into a new space from which it can then be accurately restored.
* Encoder = data to hidden
* Decoder = hidden to data

## Generative Adversarial Networks

Art transfer style - formulate and optimize texture loss:
* L = |Texture(x_ref) - Texture(X_candidate)| + |Content(X_ref) - Content(X_candidate)|

## Recurrent Neural Networks

* Exploding gradients are easy to detect but it is not clear how to detect vanishing gradients
* Exploding gradients treatment: gradient clipping and truncated BPTT
* Vanishing gradients treatment: ReLU nonlinearity, orthogonal initialization of the recurrent weights, skip connections.

* LSTM: more flexible
* GRU: less parameters
* First train LSTM, then train GRU, at last, compare each other and choose the better one.
