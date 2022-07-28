import torchvision

print('Downloading Resnet50 pretrained weights')
# torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)  # currently not working
torchvision.models.resnet.__dict__['resnet50'](pretrained=True)

print('Downloading AlexNet pretrained weights')
torchvision.models.alexnet(pretrained=True)

print('Downloading VGG16 pretrained weights')
torchvision.models.vgg16(pretrained=True)

print('Downloading VGG19 pretrained weights')
torchvision.models.vgg19(pretrained=True)
