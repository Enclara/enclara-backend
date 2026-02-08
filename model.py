"""
Brevitas-quantized VGG11 for 32x32 patches, 7-class skin cancer classification.

Two objects:
  - QuantVGG11Patch (nn.Module): the per-patch model compiled to FHE.
    Input: [B, 3, 32, 32] -> Output: [B, 7]. Pure feedforward, no control flow.
  - PatchAggregator: plain Python wrapper for training/inference.
    Splits 224x224 image into 49 patches, runs each through QuantVGG11Patch,
    takes max logit per class. Never compiled, never traced.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import brevitas.nn as qnn


NUM_CLASSES = 7
PATCH_SIZE = 32
GRID_SIZE = 7  # 224 / 32
NUM_PATCHES = GRID_SIZE * GRID_SIZE  # 49
BIT_WIDTH = 5


class QuantVGG11Patch(nn.Module):
    """
    Brevitas VGG11 variant for 32x32 patches.

    Changes from stock VGG11:
      - All Conv2d -> QuantConv2d (5-bit weights)
      - All ReLU -> QuantReLU (5-bit activations)
      - All MaxPool2d -> AvgPool2d
      - AdaptiveAvgPool2d output_size 7 -> 1
      - Classifier: single QuantLinear(512, 7) instead of 3-layer MLP

    forward() is pure sequential -- no loops, no max, no branching.
    This is what gets compiled to FHE.
    """

    def __init__(self):
        super().__init__()

        # input quantizer -- required by Concrete-ML as the very first layer
        self.quant_inp = qnn.QuantIdentity(bit_width=BIT_WIDTH, return_quant_tensor=True)

        # --- features (mirrors VGG11 layout) ---
        # block 1: conv 3->64, relu, avgpool, re-quantize
        self.conv1 = qnn.QuantConv2d(3, 64, kernel_size=3, padding=1, weight_bit_width=BIT_WIDTH, bias_quant=None)
        self.relu1 = qnn.QuantReLU(bit_width=BIT_WIDTH, return_quant_tensor=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.quant_pool1 = qnn.QuantIdentity(bit_width=BIT_WIDTH, return_quant_tensor=True)

        # block 2: conv 64->128, relu, avgpool, re-quantize
        self.conv2 = qnn.QuantConv2d(64, 128, kernel_size=3, padding=1, weight_bit_width=BIT_WIDTH, bias_quant=None)
        self.relu2 = qnn.QuantReLU(bit_width=BIT_WIDTH, return_quant_tensor=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.quant_pool2 = qnn.QuantIdentity(bit_width=BIT_WIDTH, return_quant_tensor=True)

        # block 3: conv 128->256, relu, conv 256->256, relu, avgpool, re-quantize
        self.conv3 = qnn.QuantConv2d(128, 256, kernel_size=3, padding=1, weight_bit_width=BIT_WIDTH, bias_quant=None)
        self.relu3 = qnn.QuantReLU(bit_width=BIT_WIDTH, return_quant_tensor=True)
        self.conv4 = qnn.QuantConv2d(256, 256, kernel_size=3, padding=1, weight_bit_width=BIT_WIDTH, bias_quant=None)
        self.relu4 = qnn.QuantReLU(bit_width=BIT_WIDTH, return_quant_tensor=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.quant_pool3 = qnn.QuantIdentity(bit_width=BIT_WIDTH, return_quant_tensor=True)

        # block 4: conv 256->512, relu, conv 512->512, relu, avgpool, re-quantize
        self.conv5 = qnn.QuantConv2d(256, 512, kernel_size=3, padding=1, weight_bit_width=BIT_WIDTH, bias_quant=None)
        self.relu5 = qnn.QuantReLU(bit_width=BIT_WIDTH, return_quant_tensor=True)
        self.conv6 = qnn.QuantConv2d(512, 512, kernel_size=3, padding=1, weight_bit_width=BIT_WIDTH, bias_quant=None)
        self.relu6 = qnn.QuantReLU(bit_width=BIT_WIDTH, return_quant_tensor=True)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.quant_pool4 = qnn.QuantIdentity(bit_width=BIT_WIDTH, return_quant_tensor=True)

        # block 5: conv 512->512, relu, conv 512->512, relu, avgpool, re-quantize
        self.conv7 = qnn.QuantConv2d(512, 512, kernel_size=3, padding=1, weight_bit_width=BIT_WIDTH, bias_quant=None)
        self.relu7 = qnn.QuantReLU(bit_width=BIT_WIDTH, return_quant_tensor=True)
        self.conv8 = qnn.QuantConv2d(512, 512, kernel_size=3, padding=1, weight_bit_width=BIT_WIDTH, bias_quant=None)
        self.relu8 = qnn.QuantReLU(bit_width=BIT_WIDTH, return_quant_tensor=True)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.quant_pool5 = qnn.QuantIdentity(bit_width=BIT_WIDTH, return_quant_tensor=True)

        # fixed-size pool: input is already 1x1 after pool5 with 32x32 input,
        # use AvgPool2d instead of AdaptiveAvgPool2d to avoid unsupported
        # GlobalAveragePool ONNX op in Concrete-ML
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

        # re-quantize before classifier
        self.quant_pre_classifier = qnn.QuantIdentity(bit_width=BIT_WIDTH, return_quant_tensor=True)

        # classifier: single linear layer (replaced VGG's 3-layer MLP)
        self.classifier = qnn.QuantLinear(512, NUM_CLASSES, weight_bit_width=BIT_WIDTH, bias_quant=None)

    def forward(self, x):
        # pure sequential -- no control flow, safe for ONNX export / FHE compilation
        x = self.quant_inp(x)

        x = self.quant_pool1(self.pool1(self.relu1(self.conv1(x))))
        x = self.quant_pool2(self.pool2(self.relu2(self.conv2(x))))

        x = self.relu3(self.conv3(x))
        x = self.quant_pool3(self.pool3(self.relu4(self.conv4(x))))

        x = self.relu5(self.conv5(x))
        x = self.quant_pool4(self.pool4(self.relu6(self.conv6(x))))

        x = self.relu7(self.conv7(x))
        x = self.quant_pool5(self.pool5(self.relu8(self.conv8(x))))

        # avgpool removed: spatial size is already 1x1 after pool5 with 32x32 input
        x = torch.flatten(x, 1)
        x = self.quant_pre_classifier(x)
        x = self.classifier(x)
        return x


def load_pretrained_weights(model):
    """
    Copy conv weights from torchvision's pretrained VGG11 into our QuantVGG11Patch.
    Only conv layers are transferred -- classifier is randomly initialized (new task).
    """
    vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)

    # mapping: VGG11 features index -> our model attribute name
    # VGG11 features: 0=Conv, 1=ReLU, 2=MaxPool, 3=Conv, 4=ReLU, 5=MaxPool,
    #   6=Conv, 7=ReLU, 8=Conv, 9=ReLU, 10=MaxPool,
    #   11=Conv, 12=ReLU, 13=Conv, 14=ReLU, 15=MaxPool,
    #   16=Conv, 17=ReLU, 18=Conv, 19=ReLU, 20=MaxPool
    conv_indices = [0, 3, 6, 8, 11, 13, 16, 18]
    our_convs = [model.conv1, model.conv2, model.conv3, model.conv4,
                 model.conv5, model.conv6, model.conv7, model.conv8]

    for vgg_idx, our_conv in zip(conv_indices, our_convs):
        src = vgg11.features[vgg_idx]
        our_conv.weight.data.copy_(src.weight.data)
        if src.bias is not None and our_conv.bias is not None:
            our_conv.bias.data.copy_(src.bias.data)

    print(f"copied weights from {len(conv_indices)} pretrained VGG11 conv layers")
    return model


class PatchAggregator:
    """
    Plain Python wrapper -- NOT an nn.Module.
    Splits a 224x224 image into 49 patches of 32x32, runs each through
    the patch model, takes max logit per class.

    Used during training and CPU/simulated inference.
    Never compiled to FHE, never traced, never exported.
    """

    def __init__(self, patch_model):
        self.patch_model = patch_model

    def split_into_patches(self, image):
        """
        image: tensor [C, 224, 224] or [B, C, 224, 224]
        returns: tensor [B*49, C, 32, 32] and the batch size
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape

        # unfold height then width into 32x32 patches
        # [B, C, 7, 32, 7, 32] -> [B, 49, C, 32, 32]
        patches = image.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.contiguous().view(B, C, GRID_SIZE, GRID_SIZE, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * NUM_PATCHES, C, PATCH_SIZE, PATCH_SIZE)
        return patches, B

    def forward(self, image):
        """
        image: [B, 3, 224, 224]
        returns: [B, 7] max-aggregated logits
        """
        patches, B = self.split_into_patches(image)

        # run all patches through the model
        all_logits = self.patch_model(patches)  # [B*49, 7]

        # reshape to [B, 49, 7] then max over patch dimension
        all_logits = all_logits.view(B, NUM_PATCHES, NUM_CLASSES)
        max_logits = all_logits.max(dim=1).values  # [B, 7]
        return max_logits

    def __call__(self, image):
        return self.forward(image)


if __name__ == "__main__":
    # quick sanity check
    model = QuantVGG11Patch()
    model = load_pretrained_weights(model)

    # test single patch
    x = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE)
    out = model(x)
    print(f"patch model: input {x.shape} -> output {out.shape}")

    # test aggregator
    agg = PatchAggregator(model)
    img = torch.randn(2, 3, 224, 224)
    out = agg(img)
    print(f"aggregator: input {img.shape} -> output {out.shape}")
