import os
import cv2
import numpy as np
import version_number as v
from ImageProcessing import FrameProcessor, ProcessingVariables
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from yellowbrick.classifier import ClassificationReport


version = v.vm.version

erode = ProcessingVariables.erode
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = ProcessingVariables.blur

RESIZED_IMAGE_WIDTH = 24
RESIZED_IMAGE_HEIGHT = 42

label = []
npa_flattened_images = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))


# Mengklasifikasikan angka
def test(file_path, char):
    global npa_flattened_images, label, npaFlattenedImage

    # Pra-pengolahan digit
    img = cv2.imread(file_path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # get grayscale image
    imgGaussian = cv2.GaussianBlur(imgGray, (blur, blur), 0)
    imgThreshold = cv2.adaptiveThreshold(imgGaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold, adjustment)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
    imgEroded = cv2.erode(imgThreshold, kernel, iterations=iterations)
    imgErodedCopy = imgEroded.copy()

    imgROIResized = cv2.resize(imgErodedCopy, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

    label.append(ord(char))  # ord: mengembalikan bil.integer dari karakter 01234
    npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
    npa_flattened_images = np.append(npa_flattened_images, npaFlattenedImage, 0)  # add current flattened image numpy array to list of flattened image numpy arrays


def main():
    y_train = np.loadtxt("knn/classifications" + version + ".txt", np.float32)  # read in training classifications
    x_train = np.loadtxt("knn/flattened_images" + version + ".txt", np.float32)  # read in training images

    y_train = y_train.reshape(y_train.size)  # reshape numpy array to 1d, necessary to pass to call to train

    test_dir = "tests/Digit/Digit Label"
    for fname in os.listdir(test_dir):
        path = os.path.join(test_dir, fname)
        if os.path.isdir(path):
            # print('Test ' + fname)
            tfiles = os.listdir(path)
            for tfile in tfiles:
                if not tfile.startswith('.'):
                    test(path + '/' + tfile, fname)

    fltClassification = np.array(label, np.float32)
    y_test = fltClassification.reshape((fltClassification.size))
    x_test = np.float32(npa_flattened_images)

    # Apply PCA to extract features
    pca = PCA(n_components=0.99).fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')  # Create KNN Classifier, buat model
    knn.fit(x_train, y_train)  # Train the model using the training sets
    y_pred = knn.predict(x_test)

    # print("x_train: ", x_train.shape)
    # print("y_train: ", y_train.shape)
    # print("x_test: ", x_test.shape)
    # print("y_test: ", y_test.shape)

    kategori = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    ConfusionMatrixDisplay(conf_matrix, display_labels=kategori).plot(cmap=plt.cm.binary)
    plt.grid(False)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)




    # Classification Report
    print("Classification report for - \n{}:\n{}\n".format(knn, metrics.classification_report(y_test, y_pred)))

    # result = knn.score(x_test, y_test)
    # print("Accuracy: %.2f%%" % (result*100.0))
    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
    print("Precision: ", precision_score(y_test, y_pred, average="macro") * 100)
    print("Recall: ", recall_score(y_test, y_pred, average="macro") * 100)
    print("F1 Score: ", f1_score(y_test, y_pred, average="macro") * 100)

    # plt.show()


    # Accuracy
    # test_acc = metrics.accuracy_score(y_test, y_pred) * 100
    # print("Testing accuracy: " + str(test_acc))



if __name__ == '__main__':
    main()

