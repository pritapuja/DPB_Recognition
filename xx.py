import os
import cv2
import numpy as np
import imutils
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

file_name = 'tests/tasbih_lcd/2021.jpg'
image = cv2.imread(file_name, cv2.COLOR_BGR2RGB)  # lokasi gambar

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.show()