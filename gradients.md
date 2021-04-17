# Gradients computation

Let's consider a 2 hidden layer fully-connected neural network.
$$
x^{(0)} = x\in\R^{n \times d_0} \\
x^{(1)} = x^{(0)}(W^{(1)})^T+b^{(1)}\in\R^{n\times d_1},\;W^{(1)}\in\R^{d_1 \times d_0} \\
a^{(1)} = \sigma(x^{(1)}),\;\text{pointwise} \\
x^{(2)} = a^{(1)}(W^{(2)})^T+b^{(2)}\in\R^{n\times d_2},\;W^{(2)}\in\R^{d_2 \times d_1} \\
y = x^{(2)} \\
\mathcal{L}(x)=l(y)
$$
The function $\mathcal{L}(x)$ can be expressed as a composition of functions:
$$
\mathcal{L}(x) = l(f_2(\sigma(f_1(x))))
$$
where,
$$
f_i(x) = x(W^{(i)})^T+b^{(i)},\;\text{for i=1, 2}
$$
and $\sigma(x)$ is applied point wise.

In terms of gradients dimensions, we have for $n=1$:
$$
\nabla f_i(x)\in \R^{d_{i+1}\times d_{i}}\\
\nabla\sigma(x^{(i)}) \R^{n\times d_i}
$$
The back propagation algorithm requires the gradient of layer $i-1$ to be fed into the computation of the gradient of layer $i$:
$$
\frac{\part{\mathcal{L}}}{\part{y}} = \frac{\part{\mathcal{L}}}{\part{x^{(2)}}} \in \R^{n\times d_2}\\
\frac{\part{\mathcal{L}}}{\part{a^{(1)}}} = \frac{\part{\mathcal{L}}}{\part{x^{(2)}}}\cdot \frac{\part{x^{(2)}}}{\part{a^{(1)}}}=\frac{\part{\mathcal{L}}}{\part{x^{(2)}}}\cdot W^{(2)} \in \R^{n \times d_1} \\
\frac{\part{\mathcal{L}}}{\part{x^{(1)}}} = \frac{\part{\mathcal{L}}}{\part{a^{(1)}}}\cdot \frac{\part{a^{(1)}}}{\part{x^{(1)}}}=\frac{\part{\mathcal{L}}}{\part{a^{(1)}}}\cdot \nabla\sigma(x^{(1)})  \in \R^{n \times d_1} \\
...
$$
This leads to the following scheme for our deep learning modules:

```python
class SomeModule(object):
    def __init__(self, *args):
        [...]
    def forward(self, *input):
        self.last_input = input # store input tensor for backpropagation computation
        [...]
    def backward(self, *gradwrtoutput):
        gradwrtinput = somefunction(self.last_input, gradwrtoutput)
        [...]
        self.propagate_gradient(gradwrtinput)
    def propagate_gradient(self, gradwrtinput):
        if self.previous_module is None:
            return gradwrtinput
       	else:
            return self.previous_module.backward(gradwrtinput)
```

