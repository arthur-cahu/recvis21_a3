import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch

from model import KNOWN_MODELS, make_model
from data import val_transforms

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--model-name', type=str, default='efficientnet', metavar='MN',
                    help=f'model name; one of {", ".join(KNOWN_MODELS.keys())} (default: efficientnet).')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model)
model = make_model(args.model_name, pretrained=False)
model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

test_dir = os.path.join(args.data, 'test_images/mistery_category')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = val_transforms(pil_loader(os.path.join(test_dir, f)))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile +
      ', you can upload this file to the kaggle competition website')
