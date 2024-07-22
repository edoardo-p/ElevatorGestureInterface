# Elevator Gesture Interface

This project is a proof of concept for a gesture interface for an elevator. The interface is controlled by a camera which captures the user's hand gestures. The gestures are then classified by a neural network and the gesture is shown in the application window. Required libraries are listed inside the `requirements.txt` file.

## Files

- `main.py`: Entry point for the project. Opens a window which simulates the elevator interface
- `buffer.py`: Contains the Buffer class which is used to store the gesture data and provides the callback function for MediaPipe
- `net.py`: Contains the GestureNet class which is used to classify the gestures. The `infer` method is called after the buffer is full the the main script and returns the predicted gesture
- `generate_data.py`: Contains utility functions for creating the dataset from the images inside the `data/images` directory:
  - `take_picture`: Will take a picture from the camera every time a hot key is pressed. Hot keys are listed at the top of the file and represent the gestures. The image is then saved in the appropriate directory
  - `convert_image_to_numpy`: Runs the HandLandmarker model on every image and stores all coordinates and the label in a single numpy array called `hands.npy`. Also deletes images which the model could not find hands in.
  - `print_data_description`: Prints useful information on the array such as size and class distribution
- `train.py`: Contains the training loop for the model. The models are saved in the `experiments` directory alongside the performance metrics: Accuracy, Precision, F1 Score in one file, confusion matrices in a separate file.
- `results.ipynb`: A Jupyter notebook which loads the models and metrics and graphs the results

## Experiments

Inside the `experiments` directory, there are multiple subdirectories, one for each experiment. Each subdirectory contains a `hands.npy` file which contains the data for the experiment, five `GestureNet_x.pt` files which contain the parameters from each of the model runs, a `metrics.npy` file which contains the performance metrics for each of the model runs and a `cms.npy` file containing the confusion matrix data.

The models are saved in the `models` directory and the performance metrics are saved in the `metrics` directory. The metrics are saved in a `.csv` file and contain the following columns:
