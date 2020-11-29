import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from PIL import Image
from torchvision import transforms

# 定义所需网络
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 迁移训练
def transfer_train(model, data, device, epoch, optimizer, tensorboard):
    model.train()
    for idx, (data, label) in enumerate(data):   
        data, label = data.to(device), label.to(device)
        output = model(data)
        # output = torch.nn.functional.softmax(output[0], dim=0)
        # 定义损失函数
        loss = F.cross_entropy(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if idx % 100 == 0:
            tensorboard.add_scalar('Loss', loss, global_step = epoch * 500 + idx)
            print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, idx, loss))

# 验证函数
def test_model(model, data, device, epoch, tensorboard):
    model.eval()
    correct = 0.
    acc = 0.
    with torch.no_grad():
        for idx, (x, labels) in enumerate(data):
            x, labels = x.to(device), labels.to(device)
            output = model(x)
            pred = output.argmax(dim = 1)
            correct += pred.eq(labels.view_as(pred)).sum()
    acc = correct / len(data.dataset)
    
    tensorboard.add_scalar('Accuracy', acc, global_step = epoch)
    print('Accuracy: {}'.format(acc))

# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
        
# 单张图像预测
def predict(filename, model):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    output = torch.nn.functional.softmax(output[0], dim=0)
    print(output.argmax(dim = 0))

# 查看网络参数
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

