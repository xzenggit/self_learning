
# Notes for NLP is [here](https://github.com/xzenggit/Deeplearning.ai-Natural-Language-Processing-Specialization)

## Course1 Natural Language Processing with Classification and Vector Spaces

### Week1 Sentiment Analysis with Logistic Regression
Feature extraction: Xm = [1, SUM_freqs(w, 1), SUM_freqs(w, 0)], where 1 indicates positive and 0 represents negative. freqs(w, 1) indicates the number of positive words in the context.

After we extract features from corups, we can use logistic regression to train a classification model.

### Week2  Sentiment Analysis with Naïve Bayes

* Laplacian Smoothing: p(wi|class) = [freq(wi, class) + 1]/(N_class + V), where N_class is frequency of all words in class and V is number of unique words in vocabulary.
* Naive Bayes' inference: P(pos)/P(neg) * Multi_i_m P(wi/pos)/P(wi/neg)

### Week3 Vector Space Models

Represent words and documents using vecotrs and Representation that captures relative meaning

* Word by Word design: number of times they occur together within a ceitain distance k.
* Word by Document design: number of times a word occurs within a certain category.

PCA algorithm:
* Eigenvector: Uncorrelated features for your data
* Eigenvalue: the amount of information retained by each feature

### Week4 Machine Translation and Document Search

XR=Y

## Course2 Natural Language Processing with Probabilistic Models

### Week1 Autocorrect and Minimum Edit Distance
How it works:
* Identify a misspelled word
* Find strings n edit distance away
* Filter candidates
* Calculate word probabilities

### Week2 Part of Speech Tagging and Hidden Markov Models

Markov Model: transition matrix
Viterbi algorithm:
* Initialization step
* Forward pass
* Backward pass

### Week3 Autocomplete and Language Models

Create language model from text corpus to
* Estimate probability of word sequences
* Estimate probability of a word following a sequence

N-grams: A sequence of N words
P(A, B, C, D) = P(A) P(B|A) P(C|A, B) P(D|A, B, C)
Markov assumption: only last N words matter

### Week4 Word Embedding

CBOW (Continuous bag of words): e.g. w(t) = f(w(t-2), w(t-1), w(t+1), w(t+2))

Cleaning and tokenization matters:
* letter case
* Punctuation
* Numbers
* Special characters and words

Intrinsic evaluation - test relationship between words:
* Analogies
* Clustering
* Visualization

Extrinsic evaluation - test word embedding on external task:
* Evaluates actual usefulness of embeddings
* Time-consuming
* More difficult to trobuleshoot

## Course3 Natural Language Processing with Sequence Models

### Week1 Neural Networks for Sentiment Analysis

Trax - clear and fast code for deep learning

### Week2 Recurrent Neural Networks for Language Modeling

One to one
One to many
Many to one
Many to Many

Bi-directionaly RNNS: infromation flows from the past and from the future independently.

Deep RNN: RNNs stack together

### Week3 LSTMs and Named Entity Recognition

RNNs - Advantages:
* Captures dependencies within a short range
* Takes up less RAM than other n-gram models

RNNs - Disadvantages:
* Struggles with longer sequences
* Prone to vanishing or exploding gradients

Solving for vanishing or exploding gradients:
* Identity RNN with ReLU activation
* Gradient clipping
* Skip connections

LSTMs: a memorable solution
* Learns when to remember and when to forget
* Basic anatomy:
  * A cell state
  * A hidden state with three gates
  * Loops back again at the end of each time step
* Gates allow gradients to flow unchanged.

Named Entity Recognition (NER):
* Locates and extracts predefined entities from text
* Geographical, organizations, time indicators, artifacts.
* Search engine efficiency
* Recommendation engines
* Customer service
* Automatic trading

Processing data for NERs:
* Assign each class a number
* Assign each word a number

Token padding: For LSTMs, all sequences need to be the same size.

### Week4 Siamese Networks

Siamese Networks: identify similarity between things
* Question duplicates
* Handwritten checks
* Queries

## Course4 Natural Language Processing with Attention Models

### Week1 Neural Machine Translation

Neural Machine Translation

Seq2Seq model:
* Maps variable-length sequences to a fixed memory
* Becomes a bottleneck for longer sequences
* Solution: focus attention in the right place
  * Prevent sequence overload by giving the model a way to focus the likeliest words at each step
  * Do this by providing the information specific to each input word

Information retrieval:
* Attention matches the key and query by assigning a value to the place the key is most likely to be.
* Keys and values are pairs, both coming from the encoder hidden state, whiles queries come from the decoder hidden states.
* Attention = softmax(Q K^T) V, where Q is query, K is key and V is value score.
* Attention is an added layer that lets a model focus on what's important
* Queries, Values and Keys are used for information retrieval inside the Attention layer

Teacher forcing provides faster training and higher accuracy by allowing the model to use the decoder’s actual output to compare its predictions against

BLEU Score: Bilingual Evaluation Understudy
ROUGE: recall-oriented understudy for gisting evaluation
