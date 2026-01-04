import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

import gdown
from torch.utils.data import DataLoader

class CustomPushDataset(Dataset):
    def __init__(self, dataset_folder, pred_horizon, obs_horizon, action_horizon):
        """
        Args:
            dataset_folder (str): Path to the folder containing .npy files (e.g., transforms_0.npy, transforms_1.npy).
            pred_horizon (int): Total sequence length for prediction.
            obs_horizon (int): Number of observation frames.
            action_horizon (int): Number of action frames.
        """
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        
        # List all .npy files in the given folder (assuming files are named like transforms_0.npy, transforms_1.npy, etc.)
        dataset_paths = sorted(glob.glob(os.path.join(dataset_folder, "transforms_*.npy")))
        
        # Load all episodes from the .npy files
        self.episodes = [np.load(path, allow_pickle=True).item() for path in dataset_paths]
        self.episode_lengths = [len(ep) for ep in self.episodes]
        self.episode_ends = np.cumsum(self.episode_lengths)
        
        # Flatten data into indices for sampling
        self.indices = self._create_sample_indices()
        
        # Gather all states and actions for normalization
        all_states, all_actions = self._collect_all_data()
        self.stats = self._compute_stats(all_states, all_actions)

    def _create_sample_indices(self):
        indices = []
        for episode_idx, episode_length in enumerate(self.episode_lengths):
            for frame_idx in range(-self.obs_horizon + 1, episode_length - self.action_horizon + 1):
                indices.append((episode_idx, frame_idx))
        return indices

    def _collect_all_data(self):
        lightning_states, thunder_states, lightning_actions, thunder_actions = [], [], [], []
        for episode in self.episodes:
            for frame_data in episode.values():
                # Lightning states and actions
                lightning_states.append(np.concatenate([
                    frame_data['lightning_gripper'],
                    frame_data['lightning_angle']
                ]))
                lightning_actions.append(frame_data['spark_lightning_angle'])

                # Thunder states and actions
                thunder_states.append(np.concatenate([
                    frame_data['thunder_gripper'],
                    frame_data['thunder_angle']
                ]))
                thunder_actions.append(frame_data['spark_thunder_angle'])

        return {
            'lightning_states': np.array(lightning_states),
            'thunder_states': np.array(thunder_states)
        }, {
            'lightning_actions': np.array(lightning_actions),
            'thunder_actions': np.array(thunder_actions)
        }

    def _compute_stats(self, all_states, all_actions):
        stats = {
            'lightning_states': {
                'min': np.min(all_states['lightning_states'], axis=0),
                'max': np.max(all_states['lightning_states'], axis=0)
            },
            'thunder_states': {
                'min': np.min(all_states['thunder_states'], axis=0),
                'max': np.max(all_states['thunder_states'], axis=0)
            },
            'lightning_actions': {
                'min': np.min(all_actions['lightning_actions'], axis=0),
                'max': np.max(all_actions['lightning_actions'], axis=0)
            },
            'thunder_actions': {
                'min': np.min(all_actions['thunder_actions'], axis=0),
                'max': np.max(all_actions['thunder_actions'], axis=0)
            }
        }
        return stats

    def _normalize(self, data, key):
        stats = self.stats[key]
        normalized = (data - stats['min']) / (stats['max'] - stats['min'])
        return normalized * 2 - 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, start_idx = self.indices[idx]
        episode = self.episodes[episode_idx]
        episode_length = len(episode)

        # Collect observation and action sequences
        obs_frames = []
        lightning_actions = []
        thunder_actions = []
        for i in range(start_idx, start_idx + self.pred_horizon):
            frame_idx = np.clip(i, 0, episode_length - 1)
            frame_data = episode[frame_idx]

            # Normalize images to [0, 1]
            normalized_images = {
                'camera_thunder_wrist': frame_data['camera_thunder_wrist'] / 255.0,
                'camera_lightning_wrist': frame_data['camera_lightning_wrist'] / 255.0,
                'camera_both_front': frame_data['camera_both_front'] / 255.0,
            }

            # Collect images and states
            if i < start_idx + self.obs_horizon:
                obs_frames.append({
                    **normalized_images,
                    'lightning_state': np.concatenate([
                        frame_data['lightning_gripper'],
                        frame_data['lightning_angle']
                    ]),
                    'thunder_state': np.concatenate([
                        frame_data['thunder_gripper'],
                        frame_data['thunder_angle']
                    ])
                })

            # Collect actions
            lightning_actions.append(frame_data['spark_lightning_angle'])
            thunder_actions.append(frame_data['spark_thunder_angle'])

        # Normalize states and actions
        for frame in obs_frames:
            frame['lightning_state'] = self._normalize(frame['lightning_state'], 'lightning_states')
            frame['thunder_state'] = self._normalize(frame['thunder_state'], 'thunder_states')
        lightning_actions = self._normalize(np.array(lightning_actions), 'lightning_actions')
        thunder_actions = self._normalize(np.array(thunder_actions), 'thunder_actions')

        # Format output
        images = {
            'camera_thunder_wrist': np.stack([frame['camera_thunder_wrist'] for frame in obs_frames]),
            'camera_lightning_wrist': np.stack([frame['camera_lightning_wrist'] for frame in obs_frames]),
            'camera_both_front': np.stack([frame['camera_both_front'] for frame in obs_frames])
        }
        lightning_states = np.stack([frame['lightning_state'] for frame in obs_frames])
        thunder_states = np.stack([frame['thunder_state'] for frame in obs_frames])

        # Ensure the final outputs match the expected format for the dataset
        return {
            'images': images,
            'lightning_states': lightning_states,
            'thunder_states': thunder_states,
            'lightning_actions': lightning_actions,
            'thunder_actions': thunder_actions
        }


dataset_path = "dataset"

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset = CustomPushDataset(dataset_path, pred_horizon, obs_horizon, action_horizon)

stats = dataset.stats

dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

batch = next(iter(dataloader))
print("batch['images']['camera_thunder_wrist'].shape:", batch['images']['camera_thunder_wrist'].shape)
print("batch['images']['camera_lightning_wrist'].shape:", batch['images']['camera_lightning_wrist'].shape)
print("batch['images']['camera_both_front'].shape:", batch['images']['camera_both_front'].shape)
print("batch['lightning_states'].shape:", batch['lightning_states'].shape)
print("batch['thunder_states'].shape:", batch['thunder_states'].shape)
print("batch['lightning_actions'].shape:", batch['lightning_actions'].shape)
print("batch['thunder_actions'].shape:", batch['thunder_actions'].shape)