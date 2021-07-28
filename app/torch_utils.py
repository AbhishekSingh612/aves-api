import numpy as np
import io
from PIL import Image
from torch import nn, optim, from_numpy, FloatTensor
from torch.autograd import Variable
from torchvision import models
import torch
from app.names import getName

# Here we are configuring our model structure
# (no. of hidden layers and no. of neurons in it)
# Loading resnet model and changing number of neurons in
# last layer to 200

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)

# criterion here is the loss function that will be used to
# adjust the parameters of our model
# here we used CrossEntropyLoss which is commonly used cost function
# for the classification model

criterion = nn.CrossEntropyLoss()

# here we used stochastic gradient decent
# first parameter is tensor parameters of the model
# second is the learning rate
# third is momentum factor which momentum is method
# which helps accelerate gradients vectors in the right directions,
# thus leading to faster converging.

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

# checkpoint contains the parameters of the model that gave the best accuracy during the training
# it contains the parameters (weight and bias)
# and also the optimizer parameters

checkpoint = torch.load('app/model.pth', map_location=torch.device('cpu'))
print("model loaded")
# here we load model parameters and optimizer parameters

model_ft.load_state_dict(checkpoint['model'])
optimizer_ft.load_state_dict(checkpoint['optim'])

# Model Eval is like a switch that tells that the model should not learn now
# it is required because many layers behave differentely in different phase

model_ft.eval()


def transform_image(image_bytes):
    '''This function is used to transform the image
    into a format that could be used by our model'''

    img = Image.open(io.BytesIO(image_bytes))
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    image_tensor = from_numpy(img).type(FloatTensor)

    # Add batch of size 1 to image
    return image_tensor.unsqueeze(0)


def get_prediction(image_tensor):
    output = model_ft(Variable(image_tensor))
    # print(output)
    index = output.data.cpu().numpy().argmax() + 1
    return index


def get_top5(image_tensor):
    logits = model_ft(Variable(image_tensor))
    soft_max = torch.nn.Softmax(dim=1)
    probs = soft_max(logits)
    top_probs, top_labs = probs.topk(5)
    top_probs, top_labs = top_probs.data, top_labs.data
    top_probs = top_probs.cpu().numpy().tolist()[0]
    top_labs = top_labs.cpu().numpy().tolist()[0]
    top5_dic = {
        getName(x): y for x, y in zip(top_labs, top_probs)
    }
    return top5_dic
