import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nClasses = 20

KNOWN_MODELS = {
    "resnet": {
        "constructor": models.resnet50,
        "head": "fc"
    },
    "efficientnet": {
        "constructor": models.efficientnet_b4,
        "head": "clf_sequential"
    },
    "vgg": {
        "constructor": models.vgg19_bn,
        "head": "clf_sequential"
    },
    "googlenet": {
        "constructor": models.googlenet,
        "head": "fc"
    },
}


def replace_head(model, model_name):
    if KNOWN_MODELS[model_name]["head"] == "fc":
        nFeatures = model.fc.in_features
        model.fc = nn.Linear(nFeatures, nClasses)
    elif KNOWN_MODELS[model_name]["head"] == "clf_sequential":
        nFeatures = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(nFeatures, nClasses)
    else:
        raise ValueError("Unkown classifier head.")


def make_model(model_name, finetuning=True, pretrained=True):
    if not model_name in KNOWN_MODELS:
        raise ValueError(f'Unknown model `{model_name}`.')
    # construct the model:
    model = KNOWN_MODELS[model_name]["constructor"](pretrained=pretrained)
    # freeze weights if we want to do transfer learning instead of finetuning:
    if not finetuning:
        for param in model.parameters():
            param.requires_grad = False
    replace_head(model, model_name)
    return model
