import os
import cv2
import numpy as np
import version_number as v


version = v.vm.version

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

int_classifications = []  # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end
# declare empty numpy array, we will use this to write to file later
# zero rows, enough cols to hold all image data
npa_flattened_images = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))


trained_folder = 'knn'


# Classify a digit
def train_file(file_path, char):
    global npa_flattened_images, int_classifications, npaFlattenedImage

    img = cv2.imread(file_path)
    imgROIResized = cv2.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage
    imgGray = cv2.cvtColor(imgROIResized, cv2.COLOR_BGR2GRAY)  # get grayscale image
    imgThreshCopy = imgGray.copy()

    int_classifications.append(ord(char))
    npaFlattenedImage = imgThreshCopy.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
    npa_flattened_images = np.append(npa_flattened_images, npaFlattenedImage, 0)  # add current flattened image numpy array to list of flattened image numpy arrays


def main():
    training_dir = "training/training_7.2"

    for fname in os.listdir(training_dir):
        path = os.path.join(training_dir, fname)
        if os.path.isdir(path):
            print('Training ' + fname)
            tfiles = os.listdir(path)
            for tfile in tfiles:
                if not tfile.startswith('.'):
                    train_file(path + '/' + tfile, fname)


    # Save the classifications
    fltClassification = np.array(int_classifications, np.float32)  # convert classifications list of ints to numpy array of floats
    npa_Classifications = fltClassification.reshape((fltClassification.size, 1))  # flatten numpy array of floats to 1d so we can write to file later
    np.savetxt(trained_folder + "/classifications" + version + ".txt", npa_Classifications)
    np.savetxt(trained_folder + "/flattened_images" + version + ".txt", npa_flattened_images)  # write flattened images to file

    print(npa_Classifications.shape)
if __name__ == '__main__':
    main()

