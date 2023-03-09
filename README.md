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

## Installation

## Training
```python
from sae import AutoEncoder

max_n = 16
hidden_dim = 96
feat_dim = 6

model = AutoEncoder(dim=feat_dim, hidden_dim=hidden_dim, max_n=max_n)

data_list = []
size_list = []
for i in range(batch_size):
  n = torch.randint(low=1, high=max_n, size=(1,))
  x = torch.randn(n[0], feat_dim)
  data_list.append(x)
  size_list.append(n)
x = torch.cat(data_list, dim=0)
sizes = torch.cat(size_list, dim=0)
batch = torch.arange(sizes.numel()).repeat_interleave(sizes)

x = x.to(device)
batch = batch.to(device)

xr, batchr = model(x, batch)

loss_data = model.loss()
loss = loss_data["loss"]

loss.backward()
optim.step()

optim.zero_grad()
```

## Architecture

![Encoder](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/encoderschema.png)
![Decoder](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/decoderschema.png)

## Results

![Correlation](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/correlation.png)
![Cardinality](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/cardinality.png)
![Ablation](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/ablation.png)
![Interpolation](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/interpolation.png)
![MNIST](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/mnist.png)
![Fusion Problem](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/fusion1.png)
![Fusion Results](https://github.com/Acciorocketships/SetAutoEncoder/blob/main/schema/fusion2.png)
