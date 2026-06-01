import torch
import numpy as np
import pandas as pd
import random
import math

from torch.utils.data import TensorDataset, random_split

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import psutil, os, gc
def _rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9
def mem(msg):
    gc.collect()
    print(f"[MEM] {msg}: {_rss_gb():.2f} GB")

# PAD_TOKEN = 0

data_path = "trackml_200to500_40k_pos4.csv"
normalize = False
chunking = False

usecols = ["x","y","z","px","py","pz","q","weight","event_id","particle_id","layer_id","module_id"]

if not chunking:
    sampled_data = pd.read_csv(data_path, header=0, sep=',', usecols=usecols)
    mem("after read_csv")

# Normalize the data if applicable
if normalize:
    for col in ["x", "y", "z", "px", "py", "pz", "q"]:
        mean = sampled_data[col].mean()
        std = sampled_data[col].std()
        sampled_data.loc[:, col] = (sampled_data[col] - mean)/std

# Shuffling the data and grouping by event ID
#shuffled_data = sampled_data.sample(frac=1, random_state=13) # For if batch based padding is desired

#data_grouped_by_event = shuffled_data.groupby("event_id", group_keys=False) # For if event based padding is desired (PADDING LOGIC NOT IMPLEMENTED)

def extract_track_params_data(data):
    data['p'] = np.sqrt(data['px']**2 + data['py']**2 + data['pz']**2)
    data['theta'] = np.arccos(data['pz']/data['p'])
    data['phi'] = np.arctan2(data['py'], data['px'])
    data['sin_phi'] = np.sin(data['phi'])
    data['cos_phi'] = np.cos(data['phi'])
    return data

data_track_params = extract_track_params_data(sampled_data) # For if batch based padding is desired

data = data_track_params[['x', 'y', 'z', 'event_id', 'particle_id', 'q', 'theta', 'phi', 'sin_phi', 'cos_phi', 'p', 'px', 'py', 'weight', 'layer_id', 'module_id']]
mem("after data_track_params")

del sampled_data, data_track_params
gc.collect()

#CALC CLASS_ID
n_bins_phi = 30
n_bins_theta = 30
n_bins_p = 3
n_bins_q = 2

data['p_bin'] = pd.qcut(data['p'], q=n_bins_p, labels=False)
data['theta_bin'] = pd.qcut(data['theta'], q=n_bins_theta, labels=False)
data['phi_bin'] = pd.qcut(data['phi'], q=n_bins_phi, labels=False)
data['q_bin'] = data['q'].apply(lambda x: 0 if x == -1 else 1)

# def check_bins(col, name, n_bins):
#     counts = col.value_counts().sort_index()
    
#     print(f"\n{name} bin counts:")
#     print(counts)
    
#     missing = set(range(n_bins)) - set(counts.index)
#     print(f"Empty bins: {sorted(missing)}")
#     print(f"# empty bins: {len(missing)}")

# check_bins(data['p_bin'], "p", n_bins_p)
# check_bins(data['theta_bin'], "theta", n_bins_theta)
# check_bins(data['phi_bin'], "phi", n_bins_phi)

# ADDING VALUES BY 1 SO THAT I CAN USE 0 FOR PADDING!
data['class_id'] = 1 + data['q_bin'] * n_bins_phi * n_bins_theta * n_bins_p \
                    + data['phi_bin'] * n_bins_theta * n_bins_p \
                    + data['theta_bin'] * n_bins_p \
                    + data['p_bin']


# remove the classes that are empty
# unique_classes = np.sort(data['class_id_many'].unique())
# class_map = {c: i+1 for i, c in enumerate(unique_classes)}
# data['class_id'] = data['class_id_many'].map(class_map)

# mem("after binning/class_id")

# all_possible_classes = set(range(1, 1 + n_bins_q * n_bins_phi * n_bins_theta * n_bins_p))
# existing_classes = set(data['class_id'].unique())

# empty_classes = sorted(all_possible_classes - existing_classes)

# print("Number of empty classes:", len(empty_classes))
# print(existing_classes)
# print(min(existing_classes), max(existing_classes))
# exit(0)

data['hit_id'] = data.index

data['layer_id'] = data['layer_id'] - 2
data['module_id'] = data['module_id'] - 1

data['filter_label'] = (np.sqrt(data['px']**2 + data['py']**2) >= 0.9).astype(int)

r = np.sqrt(data["x"]**2 + data["y"]**2)
data["r"] = r
data["phi"] = np.arctan2(data["y"], data["x"])
theta = np.arctan2(r, data["z"])
data["eta"] = -np.log(np.tan(theta / 2))

n_unique_classes = len(np.unique(data['class_id']))

data['unique_tracks_in_group'] = data.groupby(['event_id', 'class_id'])['particle_id'].transform('nunique')
problematic_mask = data['unique_tracks_in_group'] > 1
problematic_rows = data[problematic_mask]
unique_problematic_tracks = problematic_rows[['event_id', 'particle_id']].drop_duplicates()
wrong_tracks_count = len(unique_problematic_tracks)

total_tracks_count = data[['event_id', 'particle_id']].drop_duplicates().shape[0]
wrong_tracks_percentage = (wrong_tracks_count / total_tracks_count)

mem("after complete class_id")

# import matplotlib.pyplot as plt

# # count unique particles per class
# particles_per_class = (
#     data[['class_id', 'particle_id']]
#     .drop_duplicates()
#     .groupby('class_id')
#     .size()
# )

# print("Particles per class statistics:")
# print("Min:", particles_per_class.min())
# print("Max:", particles_per_class.max())
# print("Mean:", particles_per_class.mean())
# print("Median:", particles_per_class.median())
# print("Number of classes:", len(particles_per_class))

# histogram
# plt.figure(figsize=(12,5))
# plt.bar(particles_per_class.index, particles_per_class.values)
# plt.xlabel("Class ID")
# plt.ylabel("Particles")
# plt.title("Particles per class")
# plt.yscale('log')
# plt.savefig('/projects/0/nisei0750/nadia/part_per_class_highpt.png')

# collision_counts = data[['event_id','class_id','unique_tracks_in_group']].drop_duplicates()

# plt.figure()
# plt.hist(collision_counts['unique_tracks_in_group'], bins=20)
# plt.xlabel("Unique particles in (event_id, class_id)")
# plt.ylabel("Frequency")
# plt.title("Class collisions inside events")
# plt.yscale('log')
# plt.savefig('/projects/0/nisei0750/nadia/event_collisions_highpt.png')

#SUBSET CREATION
save_dir = "trackml_200to500_morepos/"
os.makedirs(save_dir, exist_ok=True)

train_frac, val_frac = 0.6, 0.2 #(test_frac is implicitly 1 - train_frac - val_frac)

unique_event_id = data['event_id'].unique()
n_events = len(unique_event_id)
shuffled_events = np.random.default_rng(seed = 42).permutation(unique_event_id)

n_train = int(train_frac * n_events)
n_val = int(val_frac * n_events)

train_event_ids = shuffled_events[:n_train]
val_event_ids = shuffled_events[n_train:n_train + n_val]
test_event_ids = shuffled_events[n_train + n_val:]

train_set = data[data['event_id'].isin(train_event_ids)].reset_index(drop=True)
val_set = data[data['event_id'].isin(val_event_ids)].reset_index(drop=True)
test_set = data[data['event_id'].isin(test_event_ids)].reset_index(drop=True)

print(f"Train: {train_set.shape[0]} rows ({len(train_event_ids)} events)")
print(f"Val: {val_set.shape[0]} rows ({len(val_event_ids)} events)")
print(f"Test: {test_set.shape[0]} rows ({len(test_event_ids)} events)")

test_set[['hit_id', 'particle_id', 'event_id', 'weight']].to_csv(f"{save_dir}/test_truths.csv", index=False)
mem("after split")

def extract_and_shuffle_sequences(df, feature_cols, label_col, pos_cols, seed=None):
    """Extract per-event sequences of features and labels, and shuffle hits within each event."""
    rng = np.random.default_rng(seed) if seed is not None else None
    hits_list, classes_list, pe_list = [], [], []
    for eid, grp in df.groupby('event_id', sort=False):
        feats = grp[feature_cols].to_numpy(copy=False)
        labs  = grp[label_col].to_numpy(copy=False)
        pe    = grp[pos_cols].to_numpy(copy=False)
        idx = np.arange(labs.shape[0], dtype=np.int32)
        if rng is not None:
            rng.shuffle(idx)
        else:
            np.random.shuffle(idx)
        hits_list.append(feats[idx])
        classes_list.append(labs[idx])
        pe_list.append(pe[idx])
    return hits_list, classes_list, pe_list

def sort_sequences_by_length(features, labels, posenc, descending = False):
    """Sort sequences by length (number of hits) to improve batching efficiency."""
    lengths = np.fromiter((len(seq) for seq in labels), dtype=np.int32)
    order   = np.argsort(lengths)
    if descending:
        order = order[::-1]

    sorted_feats   = [features[i] for i in order]
    sorted_labels  = [labels[i]  for i in order]
    sorted_pos     = [posenc[i] for i in order]
    
    return sorted_feats, sorted_labels, sorted_pos

def save_subset(features, label, posenc, name):
    """Save a dataset split to disk."""
    torch.save((features, label, posenc), f"{save_dir}/data_{name}.pt")

    n_events   = len(features)
    seq_lengths = [len(seq) for seq in features]
    n_rows     = sum(seq_lengths)
    min_len    = min(seq_lengths) if seq_lengths else 0
    max_len    = max(seq_lengths) if seq_lengths else 0

    print(
        f"{name:5} → {n_rows:7d} rows, across {n_events:5d} events\n"
        f"event-lengths = [{min_len} … {max_len}]"
    )

feature_cols = ['x','y','z'] #,'theta','sin_phi', 'cos_phi','q'] #, 'layer_id', 'module_id']
label_col    = 'class_id' #['class_id', 'filter_label'] # add the filter_label column when making hit filtering data
posenc_col   = ['layer_id', 'module_id', 'r', 'eta', 'phi']

train_hits_shuffled_eid, train_classes_shuffled_eid, train_pos_shuffled_eid = extract_and_shuffle_sequences(train_set, feature_cols, label_col, posenc_col, seed=13)
val_hits_shuffled_eid, val_classes_shuffled_eid, val_pos_shuffled_eid = extract_and_shuffle_sequences(val_set, feature_cols, label_col, posenc_col, seed=13)
test_hits_shuffled_eid, test_classes_shuffled_eid, test_pos_shuffled_eid = extract_and_shuffle_sequences(test_set, feature_cols, label_col, posenc_col, seed=13)

train_hits, train_classes, train_pos = sort_sequences_by_length(train_hits_shuffled_eid, train_classes_shuffled_eid, train_pos_shuffled_eid)
val_hits, val_classes, val_pos = sort_sequences_by_length(val_hits_shuffled_eid, val_classes_shuffled_eid, val_pos_shuffled_eid)
test_hits, test_classes, test_pos = sort_sequences_by_length(test_hits_shuffled_eid, test_classes_shuffled_eid, test_pos_shuffled_eid)

mem("after building sequences")

save_subset(train_hits, train_classes, train_pos, "sorted_train")
del train_hits, train_classes, train_pos
save_subset(val_hits, val_classes, val_pos, "sorted_val")
del val_hits, val_classes, val_pos
save_subset(test_hits, test_classes, test_pos, "sorted_test")

### SAVE TEST HELPER
grouped_test = test_set.groupby('event_id', sort=False)
rng = np.random.default_rng(seed=13)

test_hits = []
test_classes = []
test_hit_ids = []
test_event_ids = []
test_pos = []
for event_id, group in grouped_test:
    feature_coords = group[feature_cols].values
    class_ids = group[label_col].values
    pos_enc = group[posenc_col].values
    hit_ids = group['hit_id'].values
    event_ids = group['event_id'].values
    
    indices = np.arange(len(feature_coords))
    #np.random.shuffle(indices)
    rng.shuffle(indices)
    class_ids = class_ids[indices]
    hit_ids = hit_ids[indices]
    event_ids = event_ids[indices]
    pos_enc = pos_enc[indices]

    test_classes.append(class_ids)
    test_hit_ids.append(hit_ids)
    test_event_ids.append(event_ids)
    test_pos.append(pos_enc)

sorted_test_indices = np.argsort([len(seq) for seq in test_classes])
sorted_test_hit_ids = [test_hit_ids[i] for i in sorted_test_indices]
sorted_test_event_ids = [test_event_ids[i] for i in sorted_test_indices]
sorted_test_pos_enc = [test_pos[i] for i in sorted_test_indices]
torch.save((sorted_test_hit_ids, sorted_test_event_ids, sorted_test_pos_enc), f"{save_dir}/test_helper.pt")
