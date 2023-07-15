import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch_tensorrt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from tqdm import tqdm
import time
import numpy as np
import torch.backends.cudnn as cudnn
# import sys
# import warnings
from pytorch_quantization import tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

# sys.stderr = open(os.devnull, 'w')
# warnings.filterwarnings("ignore")

training_dataset = datasets.CIFAR10(root='./CIFAR10', transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]))
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=512, shuffle=True, num_workers=2)
testing_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]))
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=32, shuffle=False, num_workers=2)

device = torch.device("cuda")


def train(net, epochs=1):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=False,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('epochs:', end=' ')
    for epoch in range(epochs):
        print(f'{epoch + 1}', end=' ')
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print('')


def test(net, name=''):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            if labels.shape[0] == predicted.shape[0]:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    if total > 0:
        print('Accuracy of the network on the 10000 test images: %9.6f %%' % (100 * correct / total))
    else:
        print('No test data')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, padding=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, 7, padding=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def set_module(base_model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = base_model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def qat_init_model_manu(base_model):
    conv2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    conv2d_input_default_desc = QuantDescriptor(calib_method='max')

    convtrans2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    convtrans2d_input_default_desc = QuantDescriptor(calib_method='max')

    for k, m in base_model.named_modules():
        if isinstance(m, nn.Conv2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = quant_nn.QuantConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              quant_desc_input=conv2d_input_default_desc,
                                              quant_desc_weight=conv2d_weight_default_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                # noinspection PyUnresolvedReferences
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(base_model, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                            out_channels,
                                                            kernel_size,
                                                            stride,
                                                            padding,
                                                            quant_desc_input=convtrans2d_input_default_desc,
                                                            quant_desc_weight=convtrans2d_weight_default_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                # noinspection PyUnresolvedReferences
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(base_model, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      ceil_mode,
                                                      quant_desc_input=conv2d_input_default_desc)
            set_module(base_model, k, quant_maxpool2d)
        else:
            continue


def quant_setup(base_model):
    qat_init_model_manu(base_model)
    base_model.to(device)


def model_params(base_model):
    ctr = 0
    for k, m in base_model.named_modules():
        ctr += 1
        print(ctr)
        print(k, '          :          ', m)
        if hasattr(m, 'weight'):
            w = m.weight
            print('weight type', type(w))
            if hasattr(w, 'dtype'):
                print('type:', w.dtype)
        if hasattr(m, 'bias'):
            b = m.bias
            print('weight type', type(b))
            if hasattr(b, 'dtype'):
                print('type:', b.dtype)


# model = Net()
# model = resnet18(pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10)
# model = ResNet18(num_classes=10)
model = torchvision.models.mobilenet_v3_large(pretrained=True)
set_module(model, 'classifier.3', nn.Linear(1280, 10))
model.train()
model = model.cuda()

print('Training base model on CIFAR10 for 10 epochs')

train(model, 1)
torch.cuda.empty_cache()
test(model, name='Base model')
torch.cuda.empty_cache()

torch.save({'model_state_dict': model.state_dict()}, "net_base_ckpt")

print('Quantizing model')
quant_modules.initialize()

# qat_model = Net()
# qat_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# num_ftrs = qat_model.fc.in_features
# qat_model.fc = nn.Linear(num_ftrs, 10)
# qat_model = ResNet18(num_classes=10)
qat_model = torchvision.models.mobilenet_v3_large(pretrained=True)
set_module(qat_model, 'classifier.3', nn.Linear(1280, 10))
qat_model.train()
qat_model = qat_model.cuda()
# quant_setup(qat_model)

ckpt = torch.load("./net_base_ckpt")
modified_state_dict = {}
for key, val in ckpt["model_state_dict"].items():
    if key.startswith('module'):
        modified_state_dict[key[7:]] = val
    else:
        modified_state_dict[key] = val

qat_model.load_state_dict(modified_state_dict)


def compute_amax(quantized_model, **kwargs):
    for name, module in quantized_model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    quantized_model.cuda()


def collect_stats(quantized_model, data_loader, num_batches):
    for name, module in quantized_model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        quantized_model(image.cuda())
        if i >= num_batches:
            break
    for name, module in quantized_model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def calibrate_model(quantized_model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir):
    if num_calib_batch > 0:
        with torch.no_grad():
            collect_stats(quantized_model, data_loader, num_calib_batch)

        if not calibrator == "histogram":
            compute_amax(quantized_model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch * data_loader.batch_size}.pth")
            torch.save(quantized_model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(quantized_model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(quantized_model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(quantized_model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(quantized_model.state_dict(), calib_output)


print('Calibrating model')
with torch.no_grad():
    calibrate_model(
        quantized_model=qat_model,
        model_name="net",
        data_loader=training_dataloader,
        num_calib_batch=32,
        calibrator="max",
        hist_percentile=[99.9, 99.99, 99.999, 99.9999],
        out_dir="./")

test(qat_model, name='Fake Quantized model before QAT(fine tuning)')
print('Quantization aware training (fine tuning) the quantized model')
train(qat_model, 1)
torch.cuda.empty_cache()
test(qat_model, name='Fake Quantized model after QAT(fine tuning)')
torch.cuda.empty_cache()

torch.save({'model_state_dict': qat_model.state_dict()}, "net_qat_ckpt")

quant_nn.TensorQuantizer.use_fb_fake_quant = True
with torch.no_grad():
    data2 = iter(testing_dataloader)
    images, _ = next(data2)
    jit_model = torch.jit.trace(qat_model, images.to("cuda"))
    test(jit_model, 'jit model')
    torch.cuda.empty_cache()
    torch.jit.save(jit_model, "trained_net_qat.jit.pt")

qat_model = torch.jit.load("trained_net_qat.jit.pt").eval()

test(qat_model, 'loaded jit model')
torch.cuda.empty_cache()

print('Converting to TensorRT')

compile_spec = {
    "inputs": [torch_tensorrt.Input([32, 3, 32, 32])],
    "enabled_precisions": {torch.int8},
}

trt_mod = torch_tensorrt.compile(qat_model, **compile_spec)
trt_mod = torch_tensorrt.compile(qat_model,
                                 inputs=[torch_tensorrt.Input([32, 3, 32, 32])],
                                 enabled_precisions={torch.int8})
test(trt_mod, name='TensorRT model')
torch.cuda.empty_cache()
cudnn.benchmark = True


def benchmark(test_model, input_shape=(32, 3, 32, 32), dtype='fp32', nwarmup=50, nruns=1000, name=''):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == 'fp16':
        input_data = input_data.half()

    with torch.no_grad():
        for _ in range(nwarmup):
            test_model(input_data)
    torch.cuda.synchronize()
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            test_model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)

    print(f'Average batch time of {name}:\n\t\t\t\t\t\t%.2f ms' % (np.mean(timings) * 1000))


print('Comparing the speed of 3 models')
benchmark(model, name='Base model')
torch.cuda.empty_cache()
benchmark(qat_model, name='Fake Quantized model')
torch.cuda.empty_cache()
benchmark(trt_mod, name='TensorRT model')
torch.cuda.empty_cache()

print('Saved the results in paths:\n\tnet_base_ckpt\n\tnet_qat_ckpt\n\ttrained_net_qat.jit.pt')
