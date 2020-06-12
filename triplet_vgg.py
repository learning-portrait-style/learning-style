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

################################ SET UP PATHS ##############################################
# See README.md for more details

mount_filepath = "" # Put the path to the root directory of your project here
model_name_prefix = f"vae_trip_vgg"

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

OUTLIERS_REMOVED = config_args['OUTLIERS']

TRAIN_TEST_RATIO = config_args['TRAIN_TEST_SPLIT']

TIMEOUT = 3600


MASK_TYPE = "mask"

"""# HYPERPARAMETERS"""

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


################################ SETTING UP DIRECTORY STRUCTURE ##############################

## TODO: Update this with a better name? Let's stick with 1024 parameters and 512 for the artist latent vector

# hyperparameter_name = f"kld_{VAE_DIVERGENCE_WEIGHT}_percep_{PERCEP_LOSS_WEIGHT}_recon_{RECONSTRUCTION_WEIGHT}_trip_{TRIPLET_LOSS_WEIGHT}"
#hyperparameter_name = f"nonvaeweight_{NON_VAE_WEIGHT}_artist_{LATENT_ARTIST_SIZE}_latent_{LATENT_SIZE}"

hyperparameter_name = f"TRIPLET_{TRIPLET_LOSS_WEIGHT}" 

model_name = f"{model_name_prefix}_{hyperparameter_name}" # Name of the model generated from the program arguments

writer = SummaryWriter(f'./tensorboard/{model_name}') # Tensorboard writer
# kld_writer = SummaryWriter(f'./tensorboard/{model_name}/kld_loss') # Tensorboard writer
# recon_writer = SummaryWriter(f'./tensorboard/{model_name}/recon_loss') # Tensorboard writer
# artist_writer = SummaryWriter(f'./tensorboard/{model_name}/artist_loss') # Tensorboard writer
triplet_writer = SummaryWriter(f'./tensorboard/{model_name}/triplet_loss') # Tensorboard writer
# perceptual_writer = SummaryWriter(f'./tensorboard/{model_name}/perceptual_loss') # Tensorboard writer

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
print(metadata_df.shape)
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

import pdb
class TRIPLET_VGG(nn.Module):
    def __init__(self):
        super(TRIPLET_VGG, self).__init__()
        self.pretrained_net = models.vgg16(pretrained=True).to(device)
        self.pretrained_net = self.pretrained_net.eval()
        self.fc1 = nn.Linear(4096, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
      with torch.no_grad():
        activations = None
        output = x
        for _, model in enumerate(self.pretrained_net.features):
          output = model(output)

        output = self.pretrained_net.avgpool(output)
        output = output.view(output.shape[0],-1)
        # for _, model in enumerate(pretrained_net.avgpool):
          # output = model(output)
        for layer_id, model in enumerate(self.pretrained_net.classifier):
          output = model(output)
          if layer_id == 3:
            break
      
      output = self.bn1(F.relu(self.fc1(output)))
      output = self.fc2(output)

      return output

triplet_vgg = TRIPLET_VGG()
triplet_vgg.to(device)
base_optimizer = optim.Adam(triplet_vgg.parameters(), lr=1e-3)

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
  triplet_vgg.load_state_dict(base_state_dict)
#if(not find_latest_checkpoint_in_dir(LATENT_ARTIST_MODEL_LOAD_PATH, f"{DISCRIMINATOR_PREFIX}_{LATENT_PREFIX}") is None):
  #discrimnator_epoch, discriminator_state_dict = find_latest_checkpoint_in_dir(LATENT_ARTIST_MODEL_LOAD_PATH, f"{DISCRIMINATOR_PREFIX}_{LATENT_PREFIX}")
  #discriminator.load_state_dict(discriminator_state_dict)
# Optimizers for the two networks
# base_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
#discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

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

"""# MODEL TESTING

# MODEL TRAINING
"""

def test_vgg_triplet(model):
  total_loss = 0
  with torch.no_grad():
    for batch_idx, data in enumerate(test_dataset_loader):
      latents = []
      triplet_keys = ['anchor', 'positive', 'negative']

      for key in triplet_keys:
        triplet_img = data[key]['triplet']['image']
        outputs = triplet_vgg(triplet_img.to(device))
        latents.append(outputs)

      train_loss = 0
      base_optimizer.zero_grad()
      latent_loss = triplet_loss(latents[0], latents[1],latents[2])
      train_loss += latent_loss

      total_loss += train_loss
  
  return total_loss / len(test_dataset_loader)

def train_vgg_triplet(model, start_iters, end_iters, weights_folder, prefix):
  num_epochs = 0
  for iters in range(start_iters, end_iters, len(train_dataset_loader)):
    for batch_idx, data in enumerate(train_dataset_loader):
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
            torch.save(model.state_dict(), save_path)
            safe_exit(3)
            return

      latents = []
      triplet_keys = ['anchor', 'positive', 'negative']

      for key in triplet_keys:
        triplet_img = data[key]['triplet']['image']
        outputs = model(triplet_img.to(device))
        latents.append(outputs)

      train_loss = 0
      base_optimizer.zero_grad()
      latent_loss = triplet_loss(latents[0], latents[1],latents[2])
      train_loss += latent_loss

      train_loss.backward()
      writer.add_scalar(f'Loss/Train_{model_name}', train_loss, total_iters)

      print(f"Loss: {train_loss}")

    num_epochs += 1
    if num_epochs % 5 == 0:
        save_filename = f'{prefix}{total_iters}.pt'
        save_path = f'{weights_folder}/{save_filename}'
        torch.save(model.state_dict(), save_path)

    test_loss = test_vgg_triplet(model)
    triplet_writer.add_scalar(f'Loss/Test_{model_name}', test_loss, total_iters)
    print(f"TEST LOSS: {test_loss} FOR EPOCH: {total_iters}")

train_vgg_triplet(triplet_vgg, epoch, epochs, models_folder, prefix)

