import torch
import torchvision

from mnist import MnistNet

model_pt = torch.load("./model/mnist.pt")
dummy_input = torch.randn(size=(1, 1, 28, 28))

input_names = ['actual_input'] + ['learned_%d' % i for i in range(4)]

torch.onnx.export(model=model_pt, args=dummy_input, f="./model/mnist.onnx",
                  verbose=True, input_names=input_names)

