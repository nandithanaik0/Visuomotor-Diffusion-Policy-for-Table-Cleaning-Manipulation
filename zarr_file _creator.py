import os
import zarr
import numpy as np

# pusht_dataset_path = 'pusht/pusht_cchi_v7_replay.zarr'
# pusht_dataset = zarr.open(pusht_dataset_path, mode='r')
# print("PushT dataset tree: \n", pusht_dataset.tree())
# print(type(pusht_dataset))

## Read the data from the directory
raw_data_directory = 'transformed_dataset'

num_of_data_points = len(os.listdir(os.path.join(raw_data_directory, 'camera_thunder_wrist')))
print("Number of data points: ", num_of_data_points)
N = num_of_data_points
image_shape = np.load("transformed_dataset/camera_thunder_wrist/0.npy").shape
H, W, C = image_shape


## Create a new Zarr Group
zarr_data_file_path = 'table_cleaner.zarr'
root = zarr.open_group(zarr_data_file_path, mode='w')

## Create data group
data_group = root.create_group('data')

# data_group.create_dataset('img_front', shape=(N, H, W, C), chunks=(1, H, W, C), dtype='float32')
data_group.create_dataset('camera_thunder_wrist', shape=(N, H, W, C), chunks=(1, H, W, C), dtype='float32')
data_group.create_dataset('camera_lightning_wrist', shape=(N, H, W, C), chunks=(1, H, W, C), dtype='float32')
data_group.create_dataset('lightning_angle', shape=(N,6), chunks=(1,6), dtype='float32')    ## UR angles
data_group.create_dataset('pred_lightning_angle', shape=(N,6), chunks=(1,6), dtype='float32')   ## Spark Angles

meta_group = root.create_group('meta')

## Fill the data
episode_ends_raw = np.load("transformed_dataset/episode_ends.npy")
meta_group.create_dataset('episode_ends', shape=episode_ends_raw.shape, chunks=episode_ends_raw.shape, dtype='int64')
meta_group['episode_ends'][:] = episode_ends_raw

## Front Camera Files.
for i in range(N):
    # image_front = np.load(f"transformed_dataset/camera_both_front/{i}.npy")
    lightning_wrist = np.load(f"transformed_dataset/camera_lightning_wrist/{i}.npy")
    thunder_wrist = np.load(f"transformed_dataset/camera_thunder_wrist/{i}.npy")
    # data_group['img_front'][i] = image_front/255.0
    data_group['camera_lightning_wrist'][i] = lightning_wrist/255.0
    data_group['camera_thunder_wrist'][i] = thunder_wrist/255.0
    
    lightning_angles = np.load(f"transformed_dataset/lightning_angle/{i}.npy")
    # lightning_gripper = np.load(f"transformed_dataset/lightning_gripper/{i}.npy")

    # thunder_angles = np.load(f"transformed_dataset/thunder_angle/{i}.npy")
    # thunder_gripper = np.load(f"transformed_dataset/thunder_gripper/{i}.npy")

    data_group['lightning_angle'][i] = lightning_angles

    spark_lightning_angles = np.load(f"transformed_dataset/pred_lightning_angle/{i}.npy")
    # print(spark_lightning_angles)
    # spark_thunder_angles = np.load(f"transformed_dataset/spark_thunder_angle/{i}.npy")

    data_group['pred_lightning_angle'][i] = spark_lightning_angles