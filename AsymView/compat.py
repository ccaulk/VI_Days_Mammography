"""Load the official legacy checkpoint without importing unused training packages.

Forward methods below reproduce the MIT-licensed upstream Downsampler and
BasicBlock. Their parameters and architecture are restored from the checkpoint.
"""
import pickle
import torch


class Downsampler(torch.nn.Module):
    """Compatibility implementation of the upstream input downsampling block."""
    # Reproduce the upstream layer forward pass with restored parameters.
    def forward(self, x):
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))


class BasicBlock(torch.nn.Module):
    """Compatibility implementation of an upstream residual block."""
    # Reproduce the upstream layer forward pass with restored parameters.
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Unpickler(pickle.Unpickler):
    """Resolve legacy class names for the verified official checkpoint."""
    # Map the two legacy backbone classes while loading the official checkpoint.
    def find_class(self, module, name):
        if (module, name) == ('onconet.models.resnet_base', 'Downsampler'):
            return Downsampler
        if (module, name) == ('onconet.models.blocks.basic_block', 'BasicBlock'):
            return BasicBlock
        return super().find_class(module, name)


# torch.load expects a pickle-like module with these members.
load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps
