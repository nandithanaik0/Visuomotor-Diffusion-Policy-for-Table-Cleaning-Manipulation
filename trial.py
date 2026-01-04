import numpy as np
import cv2
def load_npy_as_dict(file_path):
    """
    Loads a .npy file that contains a dictionary.
    
    Parameters:
        file_path (str): Path to the .npy file.
    
    Returns:
        dict: The dictionary stored in the .npy file.
    """
    try:
        # Load the .npy file
        data = np.load(file_path, allow_pickle=True).item()
        
        # Ensure the data is a dictionary
        if not isinstance(data, dict):
            raise TypeError(f"Expected a dictionary, but got {type(data)}")
        
        print(f"Successfully loaded dictionary from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except ValueError:
        print("Error: File format is not a valid .npy file or the file is corrupted.")
    except TypeError as te:
        print(f"Error: {te}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "transforms_456.npy"  # Replace with the path to your .npy file
    dictionary = load_npy_as_dict(file_path)
    episodes = np.array([dictionary[k] for k in sorted(dictionary.keys())], dtype=object)

            # Save each frame in the processed episode
    for frame_idx, frame_data in enumerate(episodes):
        # Save the data for each topic as individual .npy files
        cv2.imshow("image",frame_data['camera_lightning_wrist'])
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # print(count)
    # count = count +
    # if cv2.waitKey(1000) & 0xFF == ord('q'):
    #     break

    if dictionary is not None:
        # Print the dictionary contents
        print("Dictionary contents:")
        for key, value in dictionary.items():
            print(f"{key}: {value}")
