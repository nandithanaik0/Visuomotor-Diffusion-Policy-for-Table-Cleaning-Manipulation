# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import glob
# import cv2


# # transform the dataset

# import os
# import glob
# import numpy as np
# original_data_folder = 'dataset'
# # transformed_data_folder = 'transformed_dataset'
# # os.makedirs(transformed_data_folder, exist_ok=True)
# # def _load_as_dict(path):
# #     """Load the data as a dictionary instead of a NumPy array"""
# #     data = np.load(path, allow_pickle=True).item()
# #     return {i: data[i] for i in range(len(data))}

# # npy_file_paths = sorted(glob.glob(os.path.join(original_data_folder, "transforms_*.npy")))
# # episodes = [_load_as_dict(path) for path in npy_file_paths]
# # sum =0
# # print(len(episodes))
# # for i in range(0,len(episodes)):
# #     sum = sum + len(episodes[i])
    
# # print(sum)
# # print(sum-(2*len(episodes)))
# # print(len(episodes[0]))
# class TransformData:
#     def __init__(self, original_data_folder):
#         # Create the transformed dataset directory
#         self.transformed_data_folder = 'transformed_dataset'
#         os.makedirs(self.transformed_data_folder, exist_ok=True)

#         # Read in the original dataset
#         self.original_data_folder = original_data_folder
#         npy_file_paths = sorted(glob.glob(os.path.join(original_data_folder, "transforms_*.npy")))
#         self.episodes = [self._load_as_dict(path) for path in npy_file_paths]

#         # Calculate episode lengths and the cumulative episode ends
        

#         # Create directories for each topic (camera images, states, and actions)
#         self.topics = [
#             'camera_thunder_wrist', 'camera_lightning_wrist',
#             'lightning_angle',  'pred_lightning_angle'
#         ]
#         for topic in self.topics:
#             os.makedirs(os.path.join(self.transformed_data_folder, topic), exist_ok=True)

#         # Process and store data
#         self._transform_and_store_data()

#         # Save episode ends as a numpy file
#         np.save(os.path.join(self.transformed_data_folder, 'episode_ends.npy'), self.episode_ends)

#     def _load_as_dict(self, path):
#         """Load the data as a dictionary instead of a NumPy array"""
#         data = np.load(path, allow_pickle=True).item()
#         return {i: data[i] for i in range(len(data))}

#     def _transform_and_store_data(self):
#         # Initialize a counter for naming the files
#         frame_counter = 0

#         # Process each episode and its frames
#         for episode_idx, episode in enumerate(self.episodes):
#             # Process episode as a dictionary
#             keys = list(episode.keys())  # Avoid modifying during iteration
#             for i in range(len(keys)):
#                 if (i + 2) >= len(keys):
#                     episode.pop(keys[i])  # Remove the key from the dictionary
#                 else:
#                     episode[keys[i]]['spark_lightning_angle'] = episode[keys[i + 2]]['lightning_angle']
#                     # episode[keys[i]]['spark_thunder_angle'] = episode[keys[i + 2]]['thunder_angle']

#             # Convert the processed dictionary back to a NumPy array
#             self.episodes[episode_idx] = np.array([episode[k] for k in sorted(episode.keys())], dtype=object)

#             # Save each frame in the processed episode
#             for frame_idx, frame_data in enumerate(self.episodes[episode_idx]):
#                 # Save the data for each topic as individual .npy files
#                 self._save_data('camera_thunder_wrist', frame_data['camera_thunder_wrist'], frame_counter)
#                 self._save_data('camera_lightning_wrist', frame_data['camera_lightning_wrist'], frame_counter)

#                 # Process and save states
#                 self._save_data('lightning_angle', frame_data['lightning_angle'], frame_counter)
#                 # self._save_data('thunder_angle', frame_data['thunder_angle'], frame_counter)

#                 # Process and save actions
#                 self._save_data('pred_lightning_angle', frame_data['spark_lightning_angle'], frame_counter)
#                 # self._save_data('pred_thunder_angle', frame_data['spark_thunder_angle'], frame_counter)

#                 # Increment the frame counter
#                 frame_counter += 1
#         self.episode_lengths = [len(ep) for ep in self.episodes]
#         self.episode_ends = np.cumsum(self.episode_lengths) - 1

#     def _save_data(self, topic, data, frame_counter):
#         """Helper function to save data into the respective folder"""
#         topic_folder = os.path.join(self.transformed_data_folder, topic)
#         np.save(os.path.join(topic_folder, f'{frame_counter}.npy'), data)

# # Example usage:
# original_data_folder = 'dataset'  # Replace with your original dataset path
# transformer = TransformData(original_data_folder)
# print(transformer.episode_ends)



# # transformed_data_folder = 'transformed_dataset'  # Path to the transformed dataset folder

# # image_folder = os.path.join(transformed_data_folder, 'camera_both_front')
# # image_files = sorted(
# #     glob.glob(os.path.join(image_folder, "*.npy")),
# #     key=lambda x: int(os.path.splitext(os.path.basename(x))[0])  # Extract the numeric part
# # )
# # print(image_files)
# # images = [np.load(file) for file in image_files]

# # for image in images:
# #     cv2.imshow('image', image)
# #     if cv2.waitKey(100) & 0xFF == ord('q'):
# #         break
# # cv2.destroyAllWindows()




# # class tablecleanerDataset(Dataset):
# #     def __init__(self, dataset_folder, pred_horizon,obs_horizon, action_horizon):
        
# #         self.episode_ends = np.load(os.path.join(dataset_folder, 'episode_ends.npy'))
# #         self.episode_starts = [0] + [end + 1 for end in self.episode_ends[:-1]]
# #         self.episode_ranges = list(zip(self.episode_starts, self.episode_ends))

# #         self.obs_horizon = obs_horizon
# #         self.pred_horizon = pred_horizon
# #         self.action_horizon = action_horizon

# #         self.sample_indices = torch.arange(0, self.episode_ends[-1] + 1)


# #     def __len__(self):
# #         return len(self.sample_indices)
    
    
# #     def __getitem__(self, idx):

# #         # identify which episode the index belongs to
# #         episode = np.where((np.array(self.episode_starts) <= idx) & (idx <= np.array(self.episode_ends)))[0]
# #         if len(episode) == 0:
# #             raise ValueError(f"Index {idx} is out of range of provided episodes.")
            
# #         episode_start = self.episode_starts[episode[0]]
# #         episode_end = self.episode_ends[episode[0]]

# #         '''
# #         Compute the indices for Observation Horizon
# #         '''
# #         obs_start = max(idx - self.obs_horizon + 1, episode_start)
# #         obs_indices = list(range(obs_start, idx + 1))
        
# #         # Pad with the episode start if needed
# #         while len(obs_indices) < self.obs_horizon:
# #             obs_indices.insert(0, episode_start)

# #         '''
# #         Compute the indices for Prediction Horizon
# #         '''
# #         pred_end = min(idx + self.pred_horizon-1, episode_end)
# #         pred_indices = list(range(idx, pred_end + 1))
        
# #         # Pad with the episode end if needed for prediction horizon
# #         while len(pred_indices) < self.pred_horizon:
# #             pred_indices.append(episode_end)


# #         '''
# #         Create the Data Item
# #         '''
# #         # camera_both_front = []
# #         camera_lightning_wrist = []
# #         camera_thunder_wrist = []

# #         # lightning_gripper = []
# #         lightning_angle = []

# #         # thunder_gripper = []
# #         # thunder_angle = []


# #         pred_lightning_angle = []
# #         # spark_thunder_angle = []

# #         for index in obs_indices:
# #             # camera_both_front.append(np.load(os.path.join('transformed_dataset', 'camera_both_front', f'{index}.npy')))
# #             camera_lightning_wrist.append(np.load(os.path.join('transformed_dataset', 'camera_lightning_wrist', f'{index}.npy')))
# #             camera_thunder_wrist.append(np.load(os.path.join('transformed_dataset', 'camera_thunder_wrist', f'{index}.npy')))

# #             # lightning_gripper.append(np.load(os.path.join('transformed_dataset', 'lightning_gripper', f'{index}.npy')))
# #             lightning_angle.append(np.load(os.path.join('transformed_dataset', 'lightning_angle', f'{index}.npy')))

# #             # thunder_gripper.append(np.load(os.path.join('transformed_dataset', 'thunder_gripper', f'{index}.npy')))
# #             # thunder_angle.append(np.load(os.path.join('transformed_dataset', 'thunder_angle', f'{index}.npy')))

# #         for index in pred_indices:
# #             pred_lightning_angle.append(np.load(os.path.join('transformed_dataset', 'pred_lightning_angle', f'{index}.npy')))
# #             # pred_thunder_angle.append(np.load(os.path.join('transformed_dataset', 'spark_thunder_angle', f'{index}.npy')))

# #         return {
# #             # 'camera_both_front': np.array(camera_both_front),
# #             # 'camera_lightning_wrist': np.array(camera_lightning_wrist),
# #             'camera_thunder_wrist': np.moveaxis(np.array(camera_thunder_wrist), -1,1),

# #             # 'lightning_gripper': np.array(lightning_gripper),
# #             'lightning_angle': np.array(lightning_angle),

# #             # 'thunder_gripper': np.array(thunder_gripper),
# #             # 'thunder_angle': np.array(thunder_angle),

# #             'pred_lightning_angle': np.array(pred_lightning_angle),
# #             # 'spark_thunder_angle': np.array(spark_thunder_angle)
# #         }


# # # dataset_folder = '/home/rpmdt05/Code/the-real-bartender/transformed_dataset'  # Replace with your actual folder path
# # # obs_horizon = 50  # Define the observation horizon (how many time steps to observe)
# # # pred_horizon = 5  # Define the prediction horizon (how many time steps to predict)
# # # action_horizon = 5  # Define the action horizon (this can be used if needed for future predictions)

# # # # Instantiate the BartenderDataset
# # # dataset = tablecleanerDataset(
# # #     dataset_folder=dataset_folder,
# # #     obs_horizon=obs_horizon,
# # #     pred_horizon=pred_horizon,
# # #     action_horizon=action_horizon
# # # )

# # # # Create a DataLoader for batching the dataset
# # # batch_size = 4  # Define the batch size
# # # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # # # Get and print one batch from the DataLoader
# # # batch_data = next(iter(dataloader))

# # # print("One batch:")
# # # # print("camera_both_front shape:", batch_data['camera_both_front'].shape)
# # # print("camera_lightning_wrist shape:", batch_data['camera_lightning_wrist'].shape)
# # # print("camera_thunder_wrist shape:", batch_data['camera_thunder_wrist'].shape)
# # # # print("lightning_gripper shape:", batch_data['lightning_gripper'].shape)
# # # print("lightning_angle shape:", batch_data['lightning_angle'].shape)
# # # # print("thunder_gripper shape:", batch_data['thunder_gripper'].shape)
# # # # print("thunder_angle shape:", batch_data['thunder_angle'].shape)
# # # print("spark_lightning_angle shape:", batch_data['pred_lightning_angle'].shape)
# # # # print("spark_thunder_angle shape:", batch_data['spark_thunder_angle'].shape)


import os
import numpy as np
import glob

class TransformData:
    def __init__(self, original_data_folder):
        # Create the transformed dataset directory
        self.transformed_data_folder = 'transformed_dataset'
        os.makedirs(self.transformed_data_folder, exist_ok=True)

        # Read in the original dataset
        self.original_data_folder = original_data_folder
        npy_file_paths = sorted(glob.glob(os.path.join(original_data_folder, "transforms_*.npy")))

        # Validate and load data
        self.episodes = self.load_npy_as_dict(npy_file_paths)

        # Print invalid file details
        # if self.invalid_files:
        #     print("Invalid files detected:")
        #     for file, error in self.invalid_files:
        #         print(f"File: {file} - Error: {error}")

        # Create directories for each topic (camera images, states, and actions)
        self.topics = [
            'camera_thunder_wrist', 'camera_lightning_wrist',
            'lightning_angle', 'pred_lightning_angle'
        ]
        for topic in self.topics:
            os.makedirs(os.path.join(self.transformed_data_folder, topic), exist_ok=True)

        # Process and store data
        self._transform_and_store_data()

        # Save episode ends as a numpy file
        np.save(os.path.join(self.transformed_data_folder, 'episode_ends.npy'), self.episode_ends)

    def load_npy_as_dict(self, file):
        data = [] 
        for file_path in file:
            try:
                # Load the .npy file
                d = np.load(file_path, allow_pickle=True).item()
                # Ensure the data is a dictionary
                # if not isinstance(data, dict):
                #     raise TypeError(f"Expected a dictionary, but got {type(data)}")
                data.append({i: d[i] for i in range(len(d))})
                
                print(f"Successfully loaded dictionary from {file_path}")
                
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
            except ValueError:
                print("Error: File format is not a valid .npy file or the file is corrupted.")
            except TypeError as te:
                print(f"Error: {te}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        return data


    def _transform_and_store_data(self):
        # Initialize a counter for naming the files
        frame_counter = 0

        # Process each episode and its frames
        for episode_idx, episode in enumerate(self.episodes):
            # Process episode as a dictionary
            keys = list(episode.keys())  # Avoid modifying during iteration
            for i in range(len(keys)):
                if (i + 2) >= len(keys):
                    episode.pop(keys[i])  # Remove the key from the dictionary
                else:
                    episode[keys[i]]['spark_lightning_angle'] = episode[keys[i + 2]]['lightning_angle']

            # Convert the processed dictionary back to a NumPy array
            self.episodes[episode_idx] = np.array([episode[k] for k in sorted(episode.keys())], dtype=object)

            # Save each frame in the processed episode
            for frame_idx, frame_data in enumerate(self.episodes[episode_idx]):
                # Save the data for each topic as individual .npy files
                self._save_data('camera_thunder_wrist', frame_data['camera_thunder_wrist'], frame_counter)
                self._save_data('camera_lightning_wrist', frame_data['camera_lightning_wrist'], frame_counter)
                self._save_data('lightning_angle', frame_data['lightning_angle'], frame_counter)
                self._save_data('pred_lightning_angle', frame_data['spark_lightning_angle'], frame_counter)

                # Increment the frame counter
                frame_counter += 1

        self.episode_lengths = [len(ep) for ep in self.episodes]
        self.episode_ends = np.cumsum(self.episode_lengths) - 1

    def _save_data(self, topic, data, frame_counter):
        """Helper function to save data into the respective folder"""
        topic_folder = os.path.join(self.transformed_data_folder, topic)
        np.save(os.path.join(topic_folder, f'{frame_counter}.npy'), data)

# Example usage
original_data_folder = 'data'  # Replace with your original dataset path
transformer = TransformData(original_data_folder)
print("Valid episodes processed:", len(transformer.episodes))
print("Episode ends:", transformer.episode_ends)









