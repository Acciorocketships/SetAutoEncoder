# Set Autoencoder

An autoencoder for sets of elements. 
While set encoders have existed for some time, (such as [Deep Sets](https://arxiv.org/abs/1703.06114)), 
there is currently no good method for decoding sets in prior literature. The difficulty of this problem
stems from the fact that sets have no ordering and a variable number of elements, while standard neural networks
produce a fixed number of outputs with a definite ordering. There are some networks which can produce a variable number
of outputs (like [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network)), but they impose an ordering on the output.
The issue with these networks is that they recognise each ordering as a distinct output. Consequently,
the task of learning becomes combinatorial–since the network can't generalise, it must be trained on each possible permutation
of the output.

Set Decoder Tasks:
- Prediction of a gaussian mixture model with a variable number of modes
- Prediction of multiple objects in a scene (e.g. objects = {blob at [-1,1], blob at [1,0]})

Autoencoder Tasks:
- Training communication in a GNN without a downstream objective
- Pre-training an encoder for a team of agents to produce a global state

![Encoder](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/encoder.png)
![Decoder](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/decoder.png)
![Correlation](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/correlation.png)
![Cardinality](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/cardinality.png)
![Ablation](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/ablation.png)
![Interpolation](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/interpolation.png)
![MNIST](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/mnist.png)
![Fusion Problem](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/fusion1.png)
![Fusion Results](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/fusion2.png)
