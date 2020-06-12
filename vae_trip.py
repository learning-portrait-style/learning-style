# -*- coding: utf-8 -*-

######################## IMPORTING / GOOGLE DRIVE INTEGRATION ##############################
import torch
import argparse
import copy
import cv2
import json
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import random
import scipy.ndimage
import shutil
import sklearn.preprocessing
import sys
#import tensorflow as tf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image 
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import pickle
import json
import pandas as pd
#import tensorflow as tf
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import pdb
import sklearn
import yaml

from shutil import copy2
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from yaml import load, dump

from functools import partial
import signal
import time
import sys
import numpy as np

"""# SETTING UP PROGRAM ARGUMENTS"""


################################ SET UP PATHS ##############################################
# See README.md for more details

mount_filepath = "" # Put the path to the root directory of your project here
model_name_prefix = f"vae_trip_tuning_sweep"


################################ SETTING UP PROGRAM ARGUMENTS ##############################

IS_RUNNING_NOTEBOOK = False  # ** NOTE: PLEASE DO NOT FORGET TO CHANGE THIS FLAG WHEN YOU DEPLOY **

exit_code = None
# The safe_exit function works by only registering a flag that is
# monitored in the actual training loop
#
# In addition, this is the central authority on the exit code for this program 
# and will be used for the actual exit code at the end
def safe_exit(signal_num, ec=0, total_iters=0, exp_name = ""):
    print("Caught a signal", signal_num)
    sys.stdout.flush()
    global exit_code
    exit_code = signal_num

start_time = time.time()

# Catch the signal and report to safe_exit
signal.signal(signal.SIGTERM, safe_exit)

device = torch.device("cuda")
parser = argparse.ArgumentParser()
parser.add_argument("yaml_config", help = "No config file specified. Please specify a yaml file.")
args = parser.parse_args()

config_file = open(args.yaml_config, 'r')
config_args = yaml.load(config_file)

LATENT_SIZE = config_args['LATENT_SIZE']
LATENT_ARTIST_SIZE = config_args['LATENT_ARTIST_SIZE']

PERCEP_RATIO = config_args['PERCEP_RATIO']
KLD_RATIO = config_args['KLD_RATIO']
TRIP_RATIO = config_args['TRIP_RATIO']
RECON_RATIO = config_args['RECON_RATIO']
TRAIN_TEST_RATIO = config_args['TRAIN_TEST_SPLIT']

OUTLIERS_REMOVED = config_args['OUTLIERS']

TIMEOUT = 3600
 
MASK_TYPE = "mask"

################################ HYPERPARAMETERS ##############################

# Base VAE Hyperparameters
TRIPLET_ALPHA = 1.0
#LATENT_ARTIST_WEIGHT = 1.0 * NON_VAE_WEIGHT
RECONSTRUCTION_WEIGHT = RECON_RATIO
PERCEP_LOSS_WEIGHT = PERCEP_RATIO
VAE_DIVERGENCE_WEIGHT = KLD_RATIO
TRIPLET_LOSS_WEIGHT = TRIP_RATIO

# Loss Thresholds
MAX_VAE_LOSS_THRESHOLD = 200
MAX_LATENT_ARTIST_LOSS_THRESHOLD = 10

# Random sampling hyperparameters
SAME_ARTIST_PROBABILITY_THRESHOLD = 0.7

# Train Test Split
TRAIN_TEST_SPLIT = TRAIN_TEST_RATIO

interval = 1 # Interval for which to print the loss
num_batches_per_epoch = 478
epochs = num_batches_per_epoch * 500
epoch = 0 # Start epoch to run from (gets automatically set in the checkpointing process)

"""# SETTING UP DIRECTORY STRUCTURE"""

################################ SETTING UP DIRECTORY STRUCTURE ##############################
## TODO: Update this with a better name? Let's stick with 1024 parameters and 512 for the artist latent vector

# hyperparameter_name = f"kld_{VAE_DIVERGENCE_WEIGHT}_percep_{PERCEP_LOSS_WEIGHT}_recon_{RECONSTRUCTION_WEIGHT}_trip_{TRIPLET_LOSS_WEIGHT}"
#hyperparameter_name = f"nonvaeweight_{NON_VAE_WEIGHT}_artist_{LATENT_ARTIST_SIZE}_latent_{LATENT_SIZE}"

hyperparameter_name = f"all_trips_TTSPLIT_{TRAIN_TEST_SPLIT}_kld_{VAE_DIVERGENCE_WEIGHT}_trip_{TRIPLET_LOSS_WEIGHT}" 

model_name = f"{model_name_prefix}_{hyperparameter_name}" # Name of the model generated from the program arguments

writer = SummaryWriter(f'./tensorboard/{model_name}') # Tensorboard writer
kld_writer = SummaryWriter(f'./tensorboard/{model_name}/kld_loss') # Tensorboard writer
recon_writer = SummaryWriter(f'./tensorboard/{model_name}/recon_loss') # Tensorboard writer
artist_writer = SummaryWriter(f'./tensorboard/{model_name}/artist_loss') # Tensorboard writer
triplet_writer = SummaryWriter(f'./tensorboard/{model_name}/triplet_loss') # Tensorboard writer
perceptual_writer = SummaryWriter(f'./tensorboard/{model_name}/perceptual_loss') # Tensorboard writer

prefix = "padded_" # Prefix for all the files being saved
continue_training = True # Specifies whether you want to continuing training from the previous epoch. Honestly don't know why you'd set it to false but you can if you want.
models_folder = f"{mount_filepath}/saved_models/{model_name}/models"

model_save_root = mount_filepath + '/saved_models'
results_folder = f"{model_save_root}/{model_name}/images"
plot_folder = f"{model_save_root}/{model_name}/plots"
log_folder = f"{model_save_root}/{model_name}/logs"

load_model_folder = f"{model_save_root}/{model_name}/models"

os.makedirs(f"{model_save_root}/{model_name}", exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

LATENT_PREFIX = "padded_"

#DISCRIMINATOR_PREFIX = "discriminator" # DONT CHANGE THIS

#LATENT_ARTIST_MODEL_NAME = f"{DISCRIMINATOR_PREFIX}_{model_name_prefix}_{hyperparameter_name}"
#os.makedirs(f"{model_save_root}/{LATENT_ARTIST_MODEL_NAME}", exist_ok=True)
#os.makedirs(f"{model_save_root}/{LATENT_ARTIST_MODEL_NAME}/images", exist_ok=True)
#os.makedirs(f"{model_save_root}/{LATENT_ARTIST_MODEL_NAME}/plots", exist_ok=True)
#os.makedirs(f"{model_save_root}/{LATENT_ARTIST_MODEL_NAME}/logs", exist_ok=True)
#os.makedirs(f"{model_save_root}/{LATENT_ARTIST_MODEL_NAME}/models", exist_ok=True)

# If you're training the base VAE, you must still specify a load path for the latent artist network
#LATENT_ARTIST_MODEL_LOAD_PATH = f"{model_save_root}/{LATENT_ARTIST_MODEL_NAME}/models"

#model_name = LATENT_ARTIST_MODEL_NAME
#prefix = LATENT_PREFIX
#load_model_folder = LATENT_ARTIST_MODEL_LOAD_PATH

"""# SETTING UP PATHS / DIRECTORY STRUCTURE"""

################################ SETTING UP PATHS / DIRECTORY STRUCTURE ##############################

metadata_filepath = mount_filepath + 'FINAL_DATASET_SIZES.csv'
image_zip_folder = mount_filepath + 'Images.zip'
image_folder = mount_filepath + "Images"
# triplets_path = mount_filepath + "Triplets/4700_labels_copy.json" 
triplets_path = mount_filepath + "Triplets/6384_labels.json" 

writer = SummaryWriter(f'./tensorboard/{model_name}') # Tensorboard writer
prefix = "padded_" # Prefix for all the files being savedc
models_folder = f"{mount_filepath}/saved_models/{model_name}/models"

model_save_root = mount_filepath + '/saved_models'
results_folder = f"{model_save_root}/{model_name}/images"
log_folder = f"{model_save_root}/{model_name}/logs"

os.makedirs(f"{model_save_root}/{model_name}", exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)
os.makedirs(f"{model_save_root}/{model_name}/models", exist_ok=True)

"""# ART DATASET CLASS"""

################################ ART DATASET CLASS ##############################

class ArtDataset(Dataset):
    def __init__(self, image_folder, triplet_df, metadata_df, largest_height, largest_width, transform=None):
        """
        Args:
            image_folder (string): Directory with all the images.
            metadata_df (Pandas Dataframe): Dataframe containing the metadata
            transform (Torchvision.Transform) Transform to be applied to the image data
        """
        self.transform = transform
        self.metadata = metadata_df.reset_index()
        self.triplets = triplet_df
        self.image_folder = image_folder
        self.largest_height = largest_height
        self.largest_width = largest_width

    def __len__(self):
        return len(self.triplets.index)

    def return_transformed_image(self, image_filename):
        image = Image.open(os.path.join(self.image_folder, image_filename))
        (width, height) = image.size
        tensor_image = self.transform(image)
        pad_layer = nn.ZeroPad2d((0, self.largest_width - width, 0, self.largest_height - height))
        return pad_layer(tensor_image)

    def get_random_artist_image(self, artist):
        artist_indices = self.metadata[self.metadata['cleaned_artist'] == artist].index
        random_artist_idx = random.choice(artist_indices)
        return self.metadata.iloc[random_artist_idx]

    def __getitem__(self, idx):
      row = self.triplets.iloc[idx]
      
      positive_idx = int(row['Positive']) - 1
      negative_idx = (1 - positive_idx)

      triplet_names = ['1', '2', 'anchor']

      # Re-Arranges it so now it's of the form positive, negative, anchor.
      triplet_files = [row[triplet_names[positive_idx]], row[triplet_names[negative_idx]], row[triplet_names[2]]]
      labels = ['positive', 'negative', 'anchor']

      triplet_objects = {}

      for filename, label in zip(triplet_files, labels):
        triplet_object = {}

        triplet_image = {}
        triplet_corresponding = {}
        
        image_metadata = self.metadata.loc[self.metadata['filename'] == filename]
        image_name = image_metadata['filename'].values[0]
        
        triplet_image['artist'] = image_metadata['cleaned_artist'].values[0]
        triplet_image['image'] = self.return_transformed_image(image_name)

        triplet_image['normalized_midpoint'] = image_metadata['normalized_midpoint'].values[0]
        triplet_image['normalized_start'] = image_metadata['normalized_start'].values[0]
        triplet_image['normalized_end'] = image_metadata['normalized_end'].values[0]

        triplet_image['width'] = image_metadata['width'].values[0]
        triplet_image['height'] = image_metadata['height'].values[0]

        #random_artist = self.get_random_artist_image(triplet_image['artist'])

        #triplet_corresponding['image'] = self.return_transformed_image(random_artist['filename'])
        #triplet_corresponding['filename'] = random_artist['filename']

        #triplet_corresponding['normalized_midpoint'] = random_artist['normalized_midpoint']
        #triplet_corresponding['normalized_start'] = random_artist['normalized_start']
        #triplet_corresponding['normalized_end'] = random_artist['normalized_end']

        #triplet_corresponding['width'] = random_artist['width']
        #triplet_corresponding['height'] = random_artist['height']

        #triplet_corresponding['should_mask'] = (random_artist['filename'] == image_metadata['filename'].values[0])

        triplet_object['triplet'] = triplet_image
        #triplet_object['corresponding'] = triplet_corresponding

        triplet_objects[label] = triplet_object

      return triplet_objects

"""# READING IN DATA FROM CSV / PREPROCESSING"""

################################ READING IN DATA FROM CSV / PREPROCESSING ##############################

import json
import pandas as pd
#pdb.set_trace()

metadata_filepath = mount_filepath + 'FINAL_DATASET_SIZES.csv'
image_zip_folder = mount_filepath + 'Images.zip'

image_folder = mount_filepath + "Images"

metadata_df = pd.read_csv(metadata_filepath)
metadata_df = metadata_df.fillna("")
artists_column = metadata_df['cleaned_artist'].str.lower()
known_artists = metadata_df[(artists_column != 'unsure') & (artists_column != 'anonymous')]

anonymous_artists = metadata_df[(artists_column == 'unsure') ^ (artists_column == 'anonymous')]

# To use all the data just set min_thresh = 0 and max_thresh = 1
min_thresh = 0.00
max_thresh = 1.00

# Only keep the artists that are known
# CURRENTLY COMMENTED SINCE WE DON'T CARE IF THE ARTISTS ARE UNKOWN!
# metadata_df = known_artists

sorted_widths = sorted(metadata_df['width'].tolist())
sorted_heights = sorted(metadata_df['height'].tolist())

if OUTLIERS_REMOVED == "outliers":
  bottom_range_height = sorted_heights[max(0, int(len(sorted_heights) * min_thresh) - 1)]
  top_range_height = sorted_heights[max(0, int(len(sorted_heights) * max_thresh) - 1)] + 1
  bottom_range_width = sorted_widths[max(0, int(len(sorted_widths) * min_thresh) - 1)]
  top_range_width = sorted_widths[max(0, int(len(sorted_widths) * max_thresh) - 1)]
else:
  bottom_range_width = 100
  top_range_width = 200
  bottom_range_height = 150
  top_range_height = 200

print(bottom_range_height)
print(top_range_height)
print(bottom_range_width)
print(top_range_width)

#middle_df = metadata_df.loc[(metadata_df['width'] >= bottom_range_width) & (metadata_df['width'] <= top_range_width) & (metadata_df['height'] >= bottom_range_height) & (metadata_df['height'] <= top_range_height)]
middle_df = metadata_df


string_json = json.load(open(triplets_path, 'r'))
triplets_df = pd.DataFrame(string_json)

non_empty_triplets = triplets_df[triplets_df['Positive'] != '']
valid_filenames = middle_df['filename'].tolist()

encoder = sklearn.preprocessing.OneHotEncoder()
one_hot_encoding = encoder.fit_transform(middle_df['cleaned_artist'].values.reshape(-1,1)).toarray()
num_artists = one_hot_encoding.shape[1]

# No longer need the one-hot encoding since we are no longer performing artist classification
# middle_df['artist_encoding'] = [np.array(a) for a in one_hot_encoding]

min_date = np.min(middle_df['mid-date'].values)
max_date = np.max(middle_df['mid-date'].values)


normalized_middate = (middle_df['mid-date'].values - min_date) / (max_date - min_date)
normalized_start_date = (middle_df['start-date'].replace('', 0).values - min_date) / (max_date - min_date)
normalized_end_date = (middle_df['end-date'].replace('', 0).values - min_date) / (max_date - min_date)

std_devs = np.zeros(len(normalized_start_date))
std_devs[normalized_start_date >= 0.0] = (normalized_middate[normalized_start_date >= 0.0] - normalized_start_date[normalized_start_date >= 0.0]) / 2.0

SAME_DATE_SIGMA = 1.0
std_devs[normalized_start_date < 0.0] = (SAME_DATE_SIGMA / (max_date - min_date))

min_std_dev = np.min(std_devs)
max_std_dev = np.max(std_devs)

middle_df['normalized_midpoint'] = normalized_middate
middle_df['normalized_start'] = normalized_start_date
middle_df['normalized_end'] = normalized_end_date

print(f"Number of Triplets: {len(non_empty_triplets)}")

triplets_within_size = non_empty_triplets[(non_empty_triplets['1'].isin(valid_filenames)) & (non_empty_triplets['2'].isin(valid_filenames)) & (non_empty_triplets['anchor'].isin(valid_filenames))]

print(f"Number of Triplets: {len(triplets_within_size)}")

"""# CREATING TRAIN / TEST SPLIT"""

################################ CREATING TRAIN / TEST SPLIT ##############################
from sklearn.model_selection import train_test_split

data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

split = 1 - TRAIN_TEST_RATIO
torch.manual_seed(0)
np.random.seed(0)
train, test = train_test_split(triplets_within_size, test_size=split)


## This is for training the Base VAE
train_art_dataset = ArtDataset(image_folder, train, middle_df, top_range_height, top_range_width, transform=data_transform)
test_art_dataset = ArtDataset(image_folder, test, middle_df, top_range_height, top_range_width, transform=data_transform)

batch_sz = 12
num_wrkrs = 5 

train_dataset_loader = torch.utils.data.DataLoader(train_art_dataset,
                                             batch_size=batch_sz, shuffle=True,
                                             num_workers=num_wrkrs)

test_dataset_loader = torch.utils.data.DataLoader(test_art_dataset,
                                             batch_size=batch_sz, shuffle=True,
                                             num_workers=num_wrkrs)

print(f"train len: {len(train_dataset_loader)}")
print(f"train len: {len(test_dataset_loader)}")


"""# DEFINING NEURAL NETWORK ARCHITECTURE"""

################################ DEFINING NEURAL NETWORK ARCHITECTURE ##############################

from collections import namedtuple
import torchvision.models as models

# Discriminator that discriminates between the two latent vectors fed in as input

class VAE_latent_artist_discriminator(nn.Module):
  def __init__(self, encoding_length):
        super(VAE_latent_artist_discriminator, self).__init__()

        self.encoding_length = encoding_length
        self.initialize_latent_artist_discriminator()
        self.sigmoid = nn.Sigmoid()  

  ######################################################################
  ######################################################################
  ####################### Network Initialization #######################
  ######################################################################
  ######################################################################

  def initialize_latent_artist_discriminator(self):
      self.latent_artist_discriminate_1 = nn.Linear(2 * self.encoding_length, self.encoding_length * 4)
      self.latent_artist_bn_1 = nn.BatchNorm1d(self.encoding_length * 4)
      self.latent_artist_discriminate_2 = nn.Linear(self.encoding_length * 4, self.encoding_length * 2)
      self.latent_artist_bn_2 = nn.BatchNorm1d(self.encoding_length * 2)
      self.latent_artist_discriminate_3 = nn.Linear(int(self.encoding_length * 2), int(self.encoding_length))
      self.latent_artist_bn_3 = nn.BatchNorm1d(int(self.encoding_length))
      self.latent_artist_discriminate_4 = nn.Linear(int(self.encoding_length), int(self.encoding_length / 2))
      self.latent_artist_bn_4 = nn.BatchNorm1d(int(self.encoding_length / 2))
      self.latent_artist_discriminate_5 = nn.Linear(int(self.encoding_length / 2), 1)

  ######################################################################
  ######################################################################
  ########################### Forward Pass #############################
  ######################################################################
  ######################################################################

  def latent_artist_discriminator_forward(self, x):
      x = F.relu(self.latent_artist_bn_1(self.latent_artist_discriminate_1(x)))
      x = F.relu(self.latent_artist_bn_2(self.latent_artist_discriminate_2(x)))
      x = F.relu(self.latent_artist_bn_3(self.latent_artist_discriminate_3(x)))
      x = F.relu(self.latent_artist_bn_4(self.latent_artist_discriminate_4(x)))
      x = self.latent_artist_discriminate_5(x)
      return self.sigmoid(x)

  def forward(self, x):
      return self.latent_artist_discriminator_forward(x)


# This is the base VAE that predicts time period and reconstructs the input image
class VAE_base(nn.Module):

    def __init__(self, latent_size, artist_latent_size, num_artists):
        super(VAE_base, self).__init__()

        self.latent_size = latent_size
        self.artist_latent_size = artist_latent_size
        self.num_artists = num_artists

        self.initialize_base_vae_enocder()
        self.initialize_base_vae_reparameterization()
        self.initialize_base_vae_decoder()

        self.sigmoid = nn.Sigmoid()


    ######################################################################
    ######################################################################
    ####################### Network Initialization #######################
    ######################################################################
    ######################################################################


    ################ Base VAE Network ####################################

    def initialize_base_vae_enocder(self):
        self.base_encode_conv1 = nn.Conv2d(3, 8, 3, stride=2, padding = 1)
        self.base_conv1_bn = nn.BatchNorm2d(8)
        self.base_encode_conv2 = nn.Conv2d(8, 16, 3, stride=2, padding = 1)
        self.base_conv2_bn = nn.BatchNorm2d(16)
        self.base_encode_conv3 = nn.Conv2d(16, 32, 3, stride=2, padding = 1)
        self.base_conv3_bn = nn.BatchNorm2d(32)
        self.base_encode_conv4 = nn.Conv2d(32, 64, 3, stride=2, padding = 1)
        self.base_conv4_bn = nn.BatchNorm2d(64)
        self.base_encode_conv5 = nn.Conv2d(64, 128, 3, stride=2, padding = 1)
        self.base_conv5_bn = nn.BatchNorm2d(128)
        self.base_encode_conv6 = nn.Conv2d(128, 256, 3, stride=2, padding = 1)
        self.base_conv6_bn = nn.BatchNorm2d(256)

        self.base_encode_conv7 = nn.Conv2d(256, 512, 3, stride=2, padding = 1)
        self.base_conv7_bn = nn.BatchNorm2d(512)

        self.base_encode_conv8 = nn.Conv2d(512, 1024, 3, stride=2, padding = 1)
        self.base_conv8_bn = nn.BatchNorm2d(1024)

    def initialize_base_vae_reparameterization(self):
        self.base_ln_encode_mean = nn.Linear(4096, self.latent_size)
        self.base_ln_encode_variance = nn.Linear(4096, self.latent_size)
        self.base_ln_decode_variance = nn.Linear(self.latent_size, 4096)

    def initialize_base_vae_decoder(self):
        self.base_decode_trans_conv1 = nn.ConvTranspose2d(4096, 512, (5, 4), stride=2)
        self.base_conv1_trans_bn = nn.BatchNorm2d(512)
        self.base_decode_trans_conv2 = nn.ConvTranspose2d(512, 256, 5, stride=2)
        self.base_conv2_trans_bn = nn.BatchNorm2d(256)
        self.base_decode_trans_conv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding = 1)
        self.base_conv3_trans_bn = nn.BatchNorm2d(128)
        self.base_decode_trans_conv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding = 1)
        self.base_conv4_trans_bn = nn.BatchNorm2d(64)
        self.base_decode_trans_conv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding = (0,1))
        self.base_conv5_trans_bn = nn.BatchNorm2d(32)
        self.base_decode_trans_conv6 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding = (0,1))
        self.base_conv6_trans_bn = nn.BatchNorm2d(16)
        self.base_decode_trans_conv7 = nn.ConvTranspose2d(16, 8, 3, stride=2)
        self.base_conv7_trans_bn = nn.BatchNorm2d(8)
        self.base_decode_trans_conv8 = nn.ConvTranspose2d(8, 3, 3, stride=1)
        #self.base_conv8_trans_bn = nn.BatchNorm2d(4)
        self.sigmoid = nn.Sigmoid()

    ######################################################################
    ######################################################################
    ########################### Forward Pass #############################
    ######################################################################
    ######################################################################

    def vae_base_encode(self, x):
        x = F.relu(self.base_conv1_bn(self.base_encode_conv1(x)))
        x = F.relu(self.base_conv2_bn(self.base_encode_conv2(x)))
        x = F.relu(self.base_conv3_bn(self.base_encode_conv3(x)))
        x = F.relu(self.base_conv4_bn(self.base_encode_conv4(x)))
        x = F.relu(self.base_conv5_bn(self.base_encode_conv5(x)))
        x = F.relu(self.base_conv6_bn(self.base_encode_conv6(x)))
        x = F.relu(self.base_conv7_bn(self.base_encode_conv7(x)))
        x = F.relu(self.base_conv8_bn(self.base_encode_conv8(x)))

        mean, log_variance = self.base_ln_encode_mean(x.view(-1,4096)), self.base_ln_encode_variance(x.view(-1,4096))
        std = torch.exp(0.5*log_variance)
        ns = torch.randn_like(std)
        z = ns * std + mean

        return z, mean, log_variance

    def vae_base_decode(self, variance):
        z = self.base_ln_decode_variance(variance)
        z = F.relu(self.base_conv1_trans_bn(self.base_decode_trans_conv1(z.view(-1,4096,1,1))))
        z = F.relu(self.base_conv2_trans_bn(self.base_decode_trans_conv2(z)))
        z = F.relu(self.base_conv3_trans_bn(self.base_decode_trans_conv3(z)))
        z = F.relu(self.base_conv4_trans_bn(self.base_decode_trans_conv4(z)))
        z = F.relu(self.base_conv5_trans_bn(self.base_decode_trans_conv5(z)))
        z = F.relu(self.base_conv6_trans_bn(self.base_decode_trans_conv6(z)))
        z = F.relu(self.base_conv7_trans_bn(self.base_decode_trans_conv7(z)))
        z = self.base_decode_trans_conv8(z)

        return self.sigmoid(z)

    def forward(self, x):
        z, mean, log_variance = self.vae_base_encode(x)
        output_image = self.vae_base_decode(z)

        return output_image, mean, log_variance, z

# Got this code from: https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
'''LossOutput = namedtuple("LossOutput", ["relu3_3"])
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '15': "relu3_3",
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
                break
        return LossOutput(**output)
'''

from collections import namedtuple

import torch
from torchvision import models


class PercepLoss(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(PercepLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        #self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        #for x in range(16, 23):
            #self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        #h = self.slice4(h)
        #h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)
        return out

# Create the two networks
vae = VAE_base(int(LATENT_SIZE), int(LATENT_ARTIST_SIZE), int(num_artists)).to(device)
#discriminator = VAE_latent_artist_discriminator(int(LATENT_ARTIST_SIZE)).to(device)

# generate the pre-trained vgg model so we can calculate the perceptual loss with it
#vgg_model = models.vgg16_bn(pretrained=True).to(device)
loss_network = PercepLoss().to(device)
loss_network = loss_network.eval()

"""# CHECKPOINT LOADING"""

################################ CHECKPOINT LOADING ##############################
def find_latest_checkpoint_in_dir(load_dir, prefix):
  pretrained_models = os.listdir(load_dir)
  if '.ipynb_checkpoints' in pretrained_models:
    pretrained_models.remove('.ipynb_checkpoints')
  max_model_name = ""
  max_model_no = -1
  for pretrained_model in pretrained_models:
    prefix_split = pretrained_model.split(f"{prefix}")
    if len(prefix_split) != 2:
      continue
    model_no = int(prefix_split[1].split(".pt")[0])
    if model_no > max_model_no:
      max_model_no = model_no
      max_model_name = pretrained_model
  epoch = 0
  if max_model_name == "":
    print("Could not find pretrained model.")
    return None
  else:
    print(f"Loaded pretrained model: {max_model_name}")
    epoch = max_model_no
    return epoch, torch.load(f'{load_dir}/{max_model_name}')

print(load_model_folder)
# Checkpoint loading for the latent artist model
if(not find_latest_checkpoint_in_dir(load_model_folder, prefix) is None):
  base_epoch, base_state_dict = find_latest_checkpoint_in_dir(load_model_folder, prefix)
  epoch = base_epoch
  vae.load_state_dict(base_state_dict)
#if(not find_latest_checkpoint_in_dir(LATENT_ARTIST_MODEL_LOAD_PATH, f"{DISCRIMINATOR_PREFIX}_{LATENT_PREFIX}") is None):
  #discrimnator_epoch, discriminator_state_dict = find_latest_checkpoint_in_dir(LATENT_ARTIST_MODEL_LOAD_PATH, f"{DISCRIMINATOR_PREFIX}_{LATENT_PREFIX}")
  #discriminator.load_state_dict(discriminator_state_dict)
# Optimizers for the two networks
base_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
#discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

"""# LOSS FUNCTIONS"""

"""# LOSS FUNCTIONS"""

################################ HELPER METHODS ##############################

def display_first_n_of_batch(batch, n):
  f, axarr = plt.subplots(1, n)
  f.set_figheight(n)
  f.set_figwidth(2 * n)

  for i in range(n):
      imgorg = np.transpose(batch[i].cpu().detach().numpy(), (1,2,0))
      axarr[i].imshow(imgorg)

  plt.show()

################################ LOSS FUNCTIONS ##############################

def loss_function(recon_x, x, mean, logvariance, data_widths, data_heights):
    recon_x = recon_x.cpu()
    x = x.cpu()
    mean = mean.cpu()
    logvariance = logvariance.cpu()
    data_widths = data_widths.cpu()
    data_heights = data_heights.cpu()

    BCE = F.binary_cross_entropy(recon_x, x, reduction='none').cpu()
    mask_matrix = torch.zeros(recon_x.shape)

    # Iterate through each element in the batch
    for i in range(0, recon_x.shape[0]):
      mask_matrix[i,:,0:data_heights[i],0:data_widths[i]] = torch.ones((3, data_heights[i], data_widths[i]))

    # Normalize KL Divergence loss by batch size
    KLD =  0.5 * (torch.sum(mean.pow(2) + logvariance.exp() - logvariance - 1) / recon_x.shape[0])

    if MASK_TYPE == "mask":
      masked_BCE = BCE * mask_matrix
      return (KLD), (torch.sum(masked_BCE / (recon_x.shape[1] * torch.sum(data_widths * data_heights))))
    else:
      return (KLD), (torch.sum(BCE) / (recon_x.shape[0] * recon_x.shape[1] * recon_x.shape[2] * recon_x.shape[3]))

def discriminator_loss(prediction, gt):
  prediction = prediction.cpu()
  gt = gt.cpu()
  BCE_loss = nn.BCELoss()
  return BCE_loss(prediction, gt)

def discriminator_loss_no_reduction(prediction, gt):
  prediction = prediction.cpu()
  gt = gt.cpu()
  BCE_loss = nn.BCELoss(reduction='none')
  return BCE_loss(prediction, gt)

def discriminator_kl_divergence(mean, logvariance):
  mean = mean.cpu()
  logvariance = logvariance.cpu()

  KLD = 0.5 * torch.sum(mean.pow(2) + logvariance.exp() - logvariance - 1)
  return KLD

def artist_prediction_loss(artist_prediction_vector, gt_one_hot_artist, mean, logvariance):
  artist_prediction_vector = artist_prediction_vector.cpu()
  gt_one_hot_artist = gt_one_hot_artist.cpu()
  mean = mean.cpu()
  logvariance = logvariance.cpu()  

  BCE_loss = nn.BCELoss()
  KLD = 0.5 * (torch.sum(mean.pow(2) + logvariance.exp() - logvariance - 1) / artist_prediction_vector.shape[0])
  return ARTIST_PREDICTION_WEIGHT * BCE_loss(artist_prediction_vector.float(), gt_one_hot_artist.float()) + ARTIST_KLD_WEIGHT * KLD

def kl_divergence_two_gaussians(p_mean, p_logvar, q_mean, q_logvar):
  p_mean = p_mean.cpu()
  q_mean = q_mean.cpu()
  p_var = p_logvar.cpu().exp()
  q_var = q_logvar.cpu().exp()

  p = torch.distributions.normal.Normal(p_mean, torch.sqrt(p_var))
  q = torch.distributions.normal.Normal(q_mean, torch.sqrt(q_var))

  return torch.distributions.kl.kl_divergence(p, q)

def kl_divergence_two_gaussians_std(p_mean, p_std, q_mean, q_std):
  p_mean = p_mean.cpu()
  q_mean = q_mean.cpu()
  p_std = p_std.cpu()
  q_std = q_std.cpu()

  p = torch.distributions.normal.Normal(p_mean, p_std)
  q = torch.distributions.normal.Normal(q_mean, q_std)

  return torch.distributions.kl.kl_divergence(p, q)

def artist_kl_divergence(latent_mean, latent_logvar, pretrained_mean, pretrained_logvar):
  return torch.mean(torch.sum(kl_divergence_two_gaussians(latent_mean, latent_logvar, pretrained_mean, pretrained_logvar), dim = 0))

def triplet_loss(a_mean, p_mean, n_mean):
    mse_criterion = nn.MSELoss()
    total_trip_loss = 0
    num_positives = 0
    for i in range(0, a_mean.shape[0]):
        distance = mse_criterion(p_mean[i], a_mean[i]) - mse_criterion(n_mean[i], a_mean[i])
        trip_loss = torch.clamp(distance + TRIPLET_ALPHA, min=0)
        if trip_loss > torch.tensor(0):
            num_positives += 1

        total_trip_loss += trip_loss
    if num_positives == 0:
        return total_trip_loss.cpu()
    else:
        return (total_trip_loss / num_positives).cpu()


#def triplet_loss(p_mean, p_logvar, n_mean, n_logvar, a_mean, a_logvar):
#  positive_divergence = torch.mean(torch.mean(kl_divergence_two_gaussians(a_mean, a_logvar, p_mean, p_logvar), dim=1))
#  negative_divergence = torch.mean(torch.mean(kl_divergence_two_gaussians(a_mean, a_logvar, n_mean, n_logvar), dim=1))

# triplet_loss = torch.clamp(positive_divergence - negative_divergence + TRIPLET_ALPHA, min=0.0)
#  return positive_divergence + torch.mean(negative_divergence)

def calculate_perceptual_loss(recon_x, x):
  # vgg_model = vgg_model.cuda()
  recon_features = loss_network(recon_x)[2].cpu()
  orig_features = loss_network(x.to(device))[2].cpu()
  
  # vgg_model = vgg_model.cpu()
  mse_criterion = nn.MSELoss()
  return mse_criterion(recon_features, orig_features)

def calculate_artist_discriminator_loss(image_batches, triplet_latent_vectors, corresponding_latent_vectors):
  # Here we go through all (6 choose 2) combinations of images and get the discriminator loss for each

  num_same_artists = 0.0
  num_diff_artists = 12 * triplet_latent_vectors[0].shape[0]

  same_artist_disc_loss = 0.0
  diff_artist_disc_loss = 0.0

  # FIRST CASE (3 Comparisons): The images of the same author (we check if they are also the same image, and if so we mask out that loss)
  for image_batch, triplet_latent, corr_latent in zip(image_batches, triplet_latent_vectors, corresponding_latent_vectors):
    should_mask_vector = image_batch["corresponding"]["should_mask"] == 0
    
    num_same_artists += torch.sum(should_mask_vector)

    latent_concat = torch.cat((triplet_latent, corr_latent), dim = 1)
    prediction = discriminator(latent_concat)
    
    disc_label = torch.ones(prediction.shape)

    same_artist_disc_loss += torch.sum(should_mask_vector * discriminator_loss_no_reduction(prediction.squeeze(), disc_label.float().squeeze()))
  
  # SECOND CASE (6 Comparisons): Compare each image from the triplet to each image in the corresponding that are not the same image
  for triplet_idx, triplet_latent in enumerate(triplet_latent_vectors):
    for corr_idx, corr_latent in enumerate(corresponding_latent_vectors):
      if (corr_idx == triplet_idx):
        continue

    latent_concat = torch.cat((triplet_latent, corr_latent), dim = 1)
    prediction = discriminator(latent_concat)
    
    disc_label = torch.zeros(prediction.shape)
    diff_artist_disc_loss += torch.sum(discriminator_loss_no_reduction(prediction.squeeze(), disc_label.float().squeeze()))
  
  # THIRD CASE (6 Comparisons): Compare each image from the same triplet
  latent_vectors_list = [triplet_latent_vectors, corresponding_latent_vectors]
  for latent_vectors_triplet in latent_vectors_list:
    for trip1_idx, trip1_latent in enumerate(latent_vectors_triplet):
      for trip2_idx, trip2_latent in enumerate(latent_vectors_triplet):
        if(trip1_idx == trip2_idx):
          continue
        latent_concat = torch.cat((trip1_latent, trip2_latent), dim = 1)
        prediction = discriminator(latent_concat)

        disc_label = torch.zeros(prediction.shape)
        diff_artist_disc_loss += torch.sum(discriminator_loss_no_reduction(prediction.squeeze(), disc_label.float().squeeze()))



  total_artist_loss = 0.0
  if torch.sum(num_same_artists) == 0:
    total_artist_loss = diff_artist_disc_loss / num_diff_artists
  else:
    total_artist_loss = (same_artist_disc_loss / num_same_artists) + (diff_artist_disc_loss / num_diff_artists)

  return total_artist_loss

"""# MODEL TESTING"""

"""# MODEL TESTING"""

# ################################ MODEL TESTING ##############################

def calculate_base_vae_test(curr_epoch):
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataset_loader):
            print(f"Evaluating batch {batch_idx} for {model_name}")
            end = time.time()
            time_elapsed = end - start_time
            
            if not IS_RUNNING_NOTEBOOK:
              if time_elapsed >= TIMEOUT:
                  exit(0)
              if exit_code is not None:
                  exit(3)

            base_optimizer.zero_grad()

            positive_image_batch = data['positive']
            negative_image_batch = data['negative']
            anchor_image_batch = data['anchor']

            image_batches = [positive_image_batch, negative_image_batch, anchor_image_batch]
            latent_mean = []
            latent_logvar = []

            triplet_latent_vectors = []
            corresponding_latent_vectors = []

            total_kld_loss = 0
            total_recon_loss = 0
            total_artist_loss = 0
            total_perceptual_loss = 0
            #total_trip_loss = 0


            for image_batch in image_batches:
              triplet_keys = ["triplet"]
              for triplet_key in triplet_keys:
                image_metadata = image_batch[triplet_key]

                recon, mean, log_variance, latent_vector = vae(image_metadata['image'].to(device))

                # Extract out just the portion related to the artist
                z_artist = latent_vector[:, 0:LATENT_ARTIST_SIZE]

                if triplet_key == "triplet":
                  latent_mean.append(mean)
                  latent_logvar.append(log_variance)
                  triplet_latent_vectors.append(z_artist)
                else:
                  corresponding_latent_vectors.append(z_artist)

                data_widths = image_metadata['width']
                data_heights = image_metadata['height']

                kld_loss, recon_loss = loss_function(recon, image_metadata['image'].to(device), mean, log_variance, data_widths, data_heights)
                total_kld_loss += kld_loss

                #with torch.no_grad():
                perceptual_loss = calculate_perceptual_loss(recon, image_metadata['image'])

                total_perceptual_loss += perceptual_loss

                total_recon_loss += recon_loss
                #kld_loss.backward()
                #recon_loss.backward()


            
            #total_artist_loss = calculate_artist_discriminator_loss(image_batches, triplet_latent_vectors, corresponding_latent_vectors)

            mean_kld_loss = (total_kld_loss / (len(image_batches)))
            print(f"AVG KLD Loss: {mean_kld_loss}")

            mean_recon_loss = (total_recon_loss / (len(image_batches)))
            #print(f"AVG Recon Loss: {mean_recon_loss}")

            #mean_artist_loss = total_artist_loss / 2.0
            #print(f"AVG Artist Discriminator Loss: {mean_artist_loss}")

            mean_percep_loss = (total_perceptual_loss / (len(image_batches)))
            print(f"AVG Perceptual Loss: {mean_percep_loss}")

            mean_triplet_loss =  triplet_loss(latent_mean[2], latent_mean[0], latent_mean[1])
            print(f"Triplet Loss: {mean_triplet_loss}")

            #loss = (VAE_DIVERGENCE_WEIGHT * mean_kld_loss) + (RECONSTRUCTION_WEIGHT * mean_recon_loss) +  (TRIPLET_LOSS_WEIGHT * mean_triplet_loss)
            loss = (VAE_DIVERGENCE_WEIGHT * mean_kld_loss) + (PERCEP_LOSS_WEIGHT * mean_percep_loss) + (RECONSTRUCTION_WEIGHT * mean_recon_loss) + (TRIPLET_LOSS_WEIGHT * mean_triplet_loss)
            #loss = (VAE_DIVERGENCE_WEIGHT * mean_kld_loss) + (RECONSTRUCTION_WEIGHT * mean_recon_loss) + (LATENT_ARTIST_WEIGHT * mean_artist_loss) + (TRIPLET_LOSS_WEIGHT * mean_triplet_loss)
            print(f"Overall Loss: {loss}")

            if torch.isnan(loss) or loss.item() > MAX_VAE_LOSS_THRESHOLD:
              pickle_name = f"{log_folder}/epoch{epoch}_batch{batch_idx}.pickle"
              print("Loss outside usual range. Skipping and reporting error")
              del loss
              continue
            
            kld_writer.add_scalar(f'Loss/Test_{model_name}', VAE_DIVERGENCE_WEIGHT * mean_kld_loss.item(), curr_epoch)
            recon_writer.add_scalar(f'Loss/Test_{model_name}', RECONSTRUCTION_WEIGHT * mean_recon_loss.item(), curr_epoch) 
            #artist_writer.add_scalar(f'Loss/Test_{model_name}', LATENT_ARTIST_WEIGHT * mean_artist_loss.item(), curr_epoch)
            triplet_writer.add_scalar(f'Loss/Test_{model_name}', TRIPLET_LOSS_WEIGHT * mean_triplet_loss.item(), curr_epoch)
            perceptual_writer.add_scalar(f'Loss/Test_{model_name}', PERCEP_LOSS_WEIGHT * mean_percep_loss, curr_epoch) 
            
            test_loss += loss.item()
            writer.add_scalar(f'Loss/Test_{model_name}', loss.item(), curr_epoch) 


    return test_loss

"""# MODEL TRAINING"""

"""# MODEL TRAINING"""

def train_base_vae(vae, end_iters, results_folder, weights_folder, prefix, start_iters, log_folder):
    #pdb.set_trace()
    loss_array = []
    test_losses = []
    display_interval = 479
    save_interval = 479 * 15
    print(f"start iters: {start_iters}")
    print(f"end iters: {end_iters}")
    for iters in range(start_iters, end_iters, len(train_dataset_loader)):
        train_loss = 0
        NUM_SAMPLES_IN_AVG = 10
        losses = []
        for batch_idx, data in enumerate(train_dataset_loader):
            #pdb.set_trace()
            total_iters = batch_idx + iters
            if total_iters >= end_iters:
                save_filename = f'{prefix}{total_iters}.pt'
                save_path = f'{weights_folder}/{save_filename}'
                torch.save(vae.state_dict(), save_path)
                return
            print(f"Evaluating batch {total_iters} for {model_name}")
            end = time.time()
            time_elapsed = end - start_time
            if not IS_RUNNING_NOTEBOOK:
              if time_elapsed >= TIMEOUT:
                  safe_exit(0)
              if exit_code is not None:
                  print("caught the exception!")
                  save_filename = f'{prefix}{total_iters}.pt'
                  save_path = f'{weights_folder}/{save_filename}'
                  torch.save(vae.state_dict(), save_path)
                  safe_exit(3)
                  return


            base_optimizer.zero_grad()
            #discriminator_optimizer.zero_grad()

            positive_image_batch = data['positive']
            negative_image_batch = data['negative']
            anchor_image_batch = data['anchor']

            image_batches = [positive_image_batch, negative_image_batch, anchor_image_batch]
            latent_mean = []
            latent_logvar = []

            triplet_latent_vectors = []
            corresponding_latent_vectors = []

            total_kld_loss = 0
            total_recon_loss = 0
            total_artist_loss = 0
            total_perceptual_loss = 0
            total_trip_loss = 0
            
            for image_batch in image_batches:
              triplet_keys = ["triplet"]
              for triplet_key in triplet_keys:
                image_metadata = image_batch[triplet_key]

                vae = vae.to(device)
                recon, mean, log_variance, latent_vector = vae(image_metadata['image'].to(device))

                # Extract out just the portion related to the artist
                z_artist = latent_vector[:, 0:LATENT_ARTIST_SIZE]

                if triplet_key == "triplet":
                  latent_mean.append(mean)
                  latent_logvar.append(log_variance)
                  triplet_latent_vectors.append(z_artist)
                else:
                  corresponding_latent_vectors.append(z_artist)

                data_widths = image_metadata['width']
                data_heights = image_metadata['height']

                perceptual_loss = calculate_perceptual_loss(recon, image_metadata['image'])

                kld_loss, recon_loss = loss_function(recon, image_metadata['image'].to(device), mean, log_variance, data_widths, data_heights)
                total_kld_loss += kld_loss
                total_recon_loss += recon_loss
                total_perceptual_loss += perceptual_loss
                #total_trip_loss += VAE_DIVERGENCE_WEIGHT * kld_loss + PERCEP_LOSS_WEIGHT * perceptual_loss
                #trip_loss = VAE_DIVERGENCE_WEIGHT * kld_loss + RECONSTRUCTION_WEIGHT * recon_loss
                #trip_loss.backward()


            
            #total_artist_loss = calculate_artist_discriminator_loss(image_batches, triplet_latent_vectors, corresponding_latent_vectors)

            mean_percep_loss = (total_perceptual_loss / (len(image_batches)))
            print(f"AVG Perceptual Loss: {PERCEP_LOSS_WEIGHT * mean_percep_loss}")

            mean_kld_loss = (total_kld_loss / (len(image_batches)))
            print(f"AVG KLD Loss: {VAE_DIVERGENCE_WEIGHT * mean_kld_loss}")

            mean_recon_loss = (total_recon_loss / (len(image_batches)))
            print(f"AVG Recon Loss: {RECONSTRUCTION_WEIGHT * mean_recon_loss}")

            #mean_artist_loss = total_artist_loss / 2
            #print(f"AVG Artist Discriminator Loss: {LATENT_ARTIST_WEIGHT * mean_artist_loss}")
            # pna -> apn
            mean_triplet_loss = triplet_loss(latent_mean[2],latent_mean[0],latent_mean[1])

            #overall_loss = (total_trip_loss/3.0) + mean_triplet_loss
            #loss = VAE_DIVERGENCE_WEIGHT * mean_kld_loss + PERCEP_LOSS_WEIGHT * mean_percep_loss + TRIPLET_LOSS_WEIGHT * mean_triplet_loss
            loss = (VAE_DIVERGENCE_WEIGHT * mean_kld_loss) + (PERCEP_LOSS_WEIGHT * mean_percep_loss) + (RECONSTRUCTION_WEIGHT * mean_recon_loss) + TRIPLET_LOSS_WEIGHT * mean_triplet_loss
            print(f"Overall Loss: {loss.item()}")

            if batch_idx < NUM_SAMPLES_IN_AVG:
                 losses.append(loss.item())
            else:
                 avg = np.sum(losses) / NUM_SAMPLES_IN_AVG
                 losses.append(loss.item())
                 losses.pop(0)
                 if torch.isnan(loss) or loss.item() > (avg + MAX_VAE_LOSS_THRESHOLD):
                   pickle_name = f"{log_folder}/epoch{epoch}_batch{batch_idx}.pickle"
                   print("Loss outside usual range. Skipping and reporting error")
                   del loss
                   del mean_percep_loss
                   del mean_kld_loss
                   del mean_recon_loss
                   del mean_triplet_loss
                   continue
            
            kld_writer.add_scalar(f'Loss/Train_{model_name}', VAE_DIVERGENCE_WEIGHT * mean_kld_loss, total_iters)
            recon_writer.add_scalar(f'Loss/Train_{model_name}', RECONSTRUCTION_WEIGHT * mean_recon_loss, total_iters)
            #artist_writer.add_scalar(f'Loss/Train_{model_name}', LATENT_ARTIST_WEIGHT * mean_artist_loss.item(), epoch * len(train_dataset_loader) + batch_idx)
            triplet_writer.add_scalar(f'Loss/Train_{model_name}', TRIPLET_LOSS_WEIGHT * mean_triplet_loss, total_iters)
            perceptual_writer.add_scalar(f'Loss/Train_{model_name}', PERCEP_LOSS_WEIGHT * mean_percep_loss, total_iters)
            
            print(f"Step: {total_iters}")
            writer.add_scalar(f'Loss/Train_{model_name}', loss.item(), total_iters)

            loss.backward()
            train_loss += loss.item()

            base_optimizer.step()
            #discriminator_optimizer.step()
            
            if total_iters % display_interval == 0:
              grid = torchvision.utils.make_grid(recon[0:5])
              writer.add_image(f'Images/{model_name}', grid, total_iters)
              ## Generate first 5 images of batch and save them to file
              f, axarr = plt.subplots(1, 5)
              f.set_figheight(5)
              f.set_figwidth(10)

              for i in range(min(5, recon.shape[0])):
                  imgorg = np.transpose(recon[i].cpu().detach().numpy(), (1,2,0))
                  axarr[i].imshow(imgorg)

              image_save_path = f'{results_folder}/{prefix}{epoch}.png'
              plt.savefig(f"{image_save_path}")
              plt.close()

              ## Calculate the test loss
              test_loss = calculate_base_vae_test(total_iters)

              print('====> Iters: {} Average loss: {:.4f}'.format(
                    total_iters, train_loss / len(train_dataset_loader.dataset)))

              ## Checkpointing
              save_filename = f'{prefix}{total_iters}.pt'
              save_path = f'{weights_folder}/{save_filename}'
              torch.save(vae.state_dict(), save_path)
              #disc_save_path = f'{LATENT_ARTIST_MODEL_LOAD_PATH}/{save_filename}'
              #torch.save(discriminator.state_dict, disc_save_path)

              print(f"Saving weights to: {weights_folder}")
              if total_iters % save_interval != 0:
                  if os.path.exists(f'{weights_folder}/{prefix}{total_iters - display_interval}.pt'):
                      os.remove(f'{weights_folder}/{prefix}{total_iters - display_interval}.pt')
                      #os.remove(f'{LATENT_ARTIST_MODEL_LOAD_PATH}/{prefix}{epoch - 1}.pt')
              print(f"Saved images results to: {image_save_path}")

    #return artist_vae


train_base_vae(vae, epochs, results_folder, models_folder, prefix, epoch, log_folder)

if exit_code is not None:
    print(exit_code)
    sys.exit(exit_code)
