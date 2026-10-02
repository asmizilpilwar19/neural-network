# Handwritten Digit Classifier — Neural Network from Scratch

A simple neural network that recognizes handwritten digits (0–9) from the
MNIST dataset, built using **only NumPy** for the math — no TensorFlow,
PyTorch, Keras, or scikit-learn. Every piece of the network (forward pass,
loss, backpropagation, gradient descent) is written out explicitly so it
can be read, understood, and explained line by line.

## Project structure

```
neural-network/
├── README.md
├── requirements.txt
├── train.py                 # main script: run this to train & evaluate
├── data/                     # MNIST data, downloaded and cached here
├── outputs/                  # generated plots land here
└── src/
    ├── data_loader.py        # downloads MNIST, converts it to NumPy arrays
    ├── activations.py        # ReLU and softmax, plus ReLU's derivative
    ├── losses.py             # cross-entropy loss and accuracy
    ├── neural_network.py     # the NeuralNetwork class (forward/backward/train)
    └── visualize.py          # Matplotlib plots of training curves & predictions
```

## Setup

```bash
pip install -r requirements.txt
python train.py
```

The first run downloads MNIST (~11 MB, from a public mirror) and caches it
in `data/mnist.npz`, so later runs start instantly. Training 10,000 images
for 20 epochs takes well under a minute on a normal laptop CPU — no GPU
needed.

## The dataset

MNIST contains 70,000 grayscale images of handwritten digits, each 28×28
pixels. `src/data_loader.py`:

1. Downloads the four original MNIST files (images + labels, for both the
   training and test sets) and parses their binary ("IDX") format using
   only `struct` and `numpy`.
2. **Flattens** each 28×28 image into a single row of 784 numbers (one per
   pixel) — the network treats an image as a long list of brightness
   values, not a 2D grid.
3. **Normalizes** pixel values from the 0–255 range down to 0–1, which
   helps training converge faster and more stably.
4. **One-hot encodes** the labels: digit `3` becomes
   `[0,0,0,1,0,0,0,0,0,0]`, i.e. a 1 in the 4th position (index 3) and 0s
   elsewhere. This format matches the network's output layer, which
   produces 10 probabilities (one per digit).

By default `train.py` trains on a 10,000-image subset for speed; change
`TRAIN_SUBSET` in `train.py` to `None` to train on the full 60,000.

## Network architecture

```
Input (784)  →  Hidden (64, ReLU)  →  Output (10, softmax)
```

- **Input layer — 784 units**: one per pixel of the flattened image.
- **Hidden layer — 64 units, ReLU activation**: a "fully connected" layer
  where every input pixel connects to every hidden unit. This is where the
  network learns intermediate patterns (edges, curves, strokes).
- **Output layer — 10 units, softmax activation**: one unit per digit. The
  softmax function turns the network's raw scores into probabilities that
  sum to 1, e.g. `[0.01, 0.02, ..., 0.91, ...]` meaning "91% sure this is
  an 8".

The network's learnable parameters are two weight matrices and two bias
vectors:

| Parameter | Shape     | Meaning                                         |
|-----------|-----------|--------------------------------------------------|
| `W1`      | (784, 64) | connects each input pixel to each hidden neuron  |
| `b1`      | (64,)     | one bias per hidden neuron                        |
| `W2`      | (64, 10)  | connects each hidden neuron to each output neuron |
| `b2`      | (10,)     | one bias per output neuron                        |

Weights start as small random numbers (He initialization, scaled by
`sqrt(2 / fan_in)`) so activations don't explode or vanish early in
training.

## How forward propagation works

Forward propagation is simply pushing an image through the network to get
a prediction, one layer at a time:

```
Z1 = X · W1 + b1        # weighted sum of inputs into the hidden layer
A1 = ReLU(Z1)           # hidden layer activation: max(0, Z1)
Z2 = A1 · W2 + b2        # weighted sum of hidden activations into the output layer
A2 = softmax(Z2)        # convert output scores into probabilities
```

- **ReLU** (`max(0, x)`) simply zeroes out negative values and passes
  positive ones through unchanged. It is cheap to compute and lets the
  network model non-linear relationships (without a non-linear activation,
  stacking layers would collapse into one big linear function).
- **Softmax** takes the output layer's raw scores and turns them into a
  probability distribution: it exponentiates each score, then divides by
  the sum of all exponentiated scores, so the results are all positive and
  add up to 1. The class with the highest probability is the network's
  prediction.

## Loss and accuracy

- **Loss (categorical cross-entropy)** measures how wrong the predicted
  probabilities are, given the true label:
  `loss = -log(predicted probability of the correct class)`.
  This value is large when the network is confident but wrong, and small
  when it is confident and right — exactly the signal needed to improve
  the weights.
- **Accuracy** is simpler: the percentage of images where the network's
  highest-probability digit matches the true digit.

Both are computed in `src/losses.py` and tracked every epoch for both the
training set and the held-out test set, so we can see whether the network
is actually learning to generalize (and not just memorizing).

## How training works (backpropagation + gradient descent)

Training repeats this loop for several **epochs** (full passes over the
training data):

1. **Shuffle** the training data and split it into small **mini-batches**
   (64 images at a time, by default). Training on small batches instead of
   the whole dataset at once makes each weight update faster and adds
   helpful noise that avoids getting stuck.
2. For each batch, run a **forward pass** to get predictions and compute
   the loss.
3. Run **backpropagation**: starting from the output layer's error, use
   the chain rule to work backward through the network and calculate
   exactly how much each weight and bias contributed to that error (its
   *gradient*).
   - `dZ2 = A2 - y_true` — the gradient of the loss with respect to the
     output layer's scores turns out to be this simple expression when
     softmax and cross-entropy are combined.
   - That error is propagated back through `W2` into the hidden layer, and
     masked by ReLU's derivative (error only flows through neurons that
     were actually "on" during the forward pass).
   - Gradients for `W1`, `b1`, `W2`, `b2` are derived from these values.
4. **Gradient descent update**: nudge every weight and bias a small step
   (controlled by the *learning rate*, 0.1 by default) in the direction
   that reduces the loss:
   `weight = weight - learning_rate * gradient`.
5. After each epoch, evaluate loss and accuracy on both the training and
   test sets and print/record them.

Repeating this thousands of times lets the network gradually adjust its
784×64 + 64×10 weights to separate the 10 digit classes.

## Example results

Training 10,000 images for 20 epochs (default settings) typically reaches:

```
Epoch  1/20 - train_loss: 0.4090 - train_acc: 0.8871 - val_loss: 0.4340 - val_acc: 0.8745
...
Epoch 20/20 - train_loss: 0.0709 - train_acc: 0.9833 - val_loss: 0.1954 - val_acc: 0.9410

Final test accuracy: 0.9410
```

Two plots are saved to `outputs/`:

- **`loss_accuracy.png`** — training and validation loss/accuracy curves
  over all 20 epochs, showing the network steadily improving.
- **`sample_predictions.png`** — a grid of 10 test digits with their true
  label and the network's prediction (green = correct, red = mistake),
  useful for showing concrete examples during a review.

Exact numbers vary slightly run to run due to the random weight
initialization and data shuffling, but ~93–95% test accuracy is typical
for this architecture and training budget.

## Things to try next

- Train on the full 60,000 images (`TRAIN_SUBSET = None` in `train.py`)
  for a small accuracy boost.
- Add a second hidden layer, or change `HIDDEN_SIZE`, to see the
  accuracy/speed trade-off.
- Try different learning rates — too high causes the loss to jump around
  (as seen briefly around epoch 7 above); too low makes training slow.
