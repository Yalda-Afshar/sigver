""" This example shows how to extract features for a new signature,
    using the CNN trained on the GPDS dataset [1]. It also compares the
    results with the ones obtained by the authors, to ensure consistency.

    [1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features
    for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks"

"""
import torch

from skimage.io import imread
from skimage import img_as_ubyte

from sigver.preprocessing.normalize import preprocess_signature
from sigver.featurelearning.models import SigNet

import sys
import sigver.datasets.util as util

#canvas_size = (952, 1360)  # Maximum signature size

# If GPU is available, use it:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))

## Load and pre-process the signature
#original = img_as_ubyte(imread('data/some_signature.png', as_gray=True))
#processed = preprocess_signature(original, canvas_size)
#
## Note: the image needs to be a pytorch tensor with pixels in the range [0, 1]
#input = torch.from_numpy(processed).view(1, 1, 150, 220)
#input = input.float().div(255).to(device)

x, y, yforg, usermapping, filenames = util.load_dataset("persian_1_115_150-220.npz")
i = int(sys.argv[1])
j = int(sys.argv[2])
input1 = torch.from_numpy(x[i][0]).view(1, 1, 150, 220)
input1 = input1.float().div(255).to(device)
input2 = torch.from_numpy(x[j][0]).view(1, 1, 150, 220)
input2 = input2.float().div(255).to(device)

# Load the model
#state_dict, _, _ = torch.load('pre_trained/signet_f_lambda_0.95.pth')
state_dict, _, _ = torch.load('signet_with_forgery_150-220/model_best.pth')
base_model = SigNet().to(device).eval()
base_model.load_state_dict(state_dict)

# Extract features
with torch.no_grad(): # We don't need gradients. Inform torch so it doesn't compute them
    features1 = base_model(input1)
    features2 = base_model(input2)

# Check against the results obtained by the author:
#expected_features = torch.load('data/some_signature_features.pth')

print((features1.cpu() - features2.cpu()).abs().max())
print(y[i], y[j])
print(yforg[i], yforg[j])

from PIL import Image
img1 = Image.fromarray(x[i][0], 'L')
img1.show(title=filenames[i])
print(filenames[i])

img2 = Image.fromarray(x[j][0], 'L')
img2.show(title=filenames[j])
print(filenames[j])