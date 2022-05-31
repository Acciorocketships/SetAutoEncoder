# Set Autoencoder

An autoencoder for sets of elements. 
While set encoders have existed for some time, (such as [Deep Sets](https://arxiv.org/abs/1703.06114)), 
there is currently no good method for decoding sets in prior literature. The difficulty of this problem
stems from the fact that sets have no ordering and a variable number of elements, while standard neural networks
produce a fixed number of outputs with a definite ordering. There are some networks which can produce a variable number
of outputs (like [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network)), but they impose an ordering on the output.
The issue with networks which produce ordered outputs is that they recognise each ordering as a distinct output. Consequently,
the task of learning becomes combinatorial–since the network can't generalise, it must be trained on each possible permutation
of the output.

The usefulness of a set decoder (and, by extension, a set autoencoder) is quite clear. To list just a few possible applications:
- classification of a variable number of objects in an image
- multiple task assignment given a variable number of tasks
- producing continuous, multi-modal action distributions with a sum of a variable number of gaussians (usually to produce a multi-modal distribution, it must be discrete)
- A building block for a graph autoencoder, which can be used for neural cellular automata

## Architecture

### Encoder
![Encoder](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/encoder.png)

### Decoder
![Decoder](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/decoder.png)
