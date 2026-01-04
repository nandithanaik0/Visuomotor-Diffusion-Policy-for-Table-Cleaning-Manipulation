# import rosbag
import cv2
# from cv_bridge import CvBridge
import numpy as np

import os
import glob
import numpy as np
# import_data = np.load('data/transforms_0.npy', allow_pickle=True).item()


## save as npz
# np.savez_compressed('dataset/transforms_0_mhhit_shit.npz', import_data)

# # Initialize CvBridge
# bridge = CvBridge()

# # Path to the bag file
# path = 'bag_files/initial_no_sync.bag'

# # Open the bag file
# bag = rosbag.Bag(path)

# for topic, msg, t in bag.read_messages():
#     # print the type of the message
#     print(topic)


# file1 = "transformed_dataset/camera_thunder_wrist"
# file2 = "transformed_dataset/camera_lightning_wrist"


# data = np.load(file, allow_pickle=True).item()

# # print(data[0]['camera_thunder_wrist'].shape)
# # image = data[20]['camera_thunder_wrist']
# # cv2.imshow('image', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# for key in data:
#     # print(key)
#     # camera_thunder_wrist = data[key]['thunder_angle']
#     # camera_lightning_wrist = data[key]['lightning_angle']
#     # print(camera_lightning_wrist)
#     camera_thunder_wrist = cv2.cvtColor(camera_thunder_wrist, cv2.COLOR_BGR2RGB)
#     camera_lightning_wrist = cv2.cvtColor(camera_lightning_wrist, cv2.COLOR_BGR2RGB)
#     cv2.imshow('image', camera_thunder_wrist)
#     cv2.imshow('image2', camera_lightning_wrist)
#     if cv2.waitKey(50) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
count =0

transformed_data_folder = 'transformed_dataset'  # Path to the transformed dataset folder

image_folder = os.path.join(transformed_data_folder, 'camera_thunder_wrist')

image_files = sorted(
    glob.glob(os.path.join(image_folder, "*.npy")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])  # Extract the numeric part
)
# print(image_files)
images = [np.load(file) for file in image_files]

for image in images:
    cv2.imshow('image', image)
    # print(count)
    # count = count +1
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
