import os
import cv2
import numpy as np
import imutils
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
CROP_DIR = 'save'


class FrameProcessor:
    def __init__(self, height, version, debug=False, write_digits=False):
        self.debug = debug
        self.version = version
        self.height = height
        self.file_name = None
        self.image = None
        self.width = 0
        self.original = None
        self.ori = None
        self.write_digits = write_digits
        self.knn = self.train_kNN(self.version)
        self.pca_knn = self.train_PCA_kNN(self.version)


    def set_image(self, file_name):
        if not os.path.isfile(file_name):
            raise IOError("File tidak ada:", file_name)
        self.file_name = file_name
        self.image = cv2.imread(file_name)  # lokasi gambar
        self.original, self.width = self.resize_to_height(self.height)

    def resize_to_height(self, height):
        # menghitung rasio tinggi dan membangun dimensi
        r = self.image.shape[0] / float(height)
        dim = (int(self.image.shape[1] / r), height)
        img = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)  # memperkecil image
        return img, dim[0]

    def inverse_colors(self, img):
        img = (255 - img)
        return img

    def sort_contours(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "become-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        # menangani jika mengurutkan koordinat y daripada koordinat x dari kotak pembatas
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to bottom
        # buat daftar kotak pembatas dan urutkan dari atas ke bawah
        bounding_boxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                             key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        # return daftar kontur yang diurutkan dan kotak pembatas
        return cnts, bounding_boxes

    def train_kNN(self, version):

        y_train = np.loadtxt("knn/classifications" + version + ".txt", np.float32)  # read in training classifications
        X_train = np.loadtxt("knn/flattened_images" + version + ".txt", np.float32)  # read in training images

        y_train = y_train.reshape(y_train.size)  # reshape numpy array to 1d, necessary to pass to call to train

        # Create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

        # Train the model using the training sets
        knn.fit(X_train, y_train)

        return knn

    def train_PCA_kNN(self, version):
        global pca

        y_train = np.loadtxt("knn/classifications" + version + ".txt", np.float32)  # read in training classifications
        X_train = np.loadtxt("knn/flattened_images" + version + ".txt", np.float32)  # read in training images

        y_train = y_train.reshape(y_train.size)  # reshape numpy array to 1d, necessary to pass to call to train

        pca = PCA(n_components=0.99)
        pca.fit(X_train)
        X_pca = pca.transform(X_train)

        # Create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

        # Train the model using the training sets
        knn.fit(X_pca, y_train)

        return knn


    def crop(self, nama_file, image, blur, threshold, adjustment, erode, iterations):

        image = imutils.resize(image, width=500)  # mengubah ukuran gambar menjadi lebar 500px unutk memudahkan pemrosesan di tahapan selanjutnya

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # konversi ke RGB

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Blur to reduce noise
        img_blurred = cv2.GaussianBlur(gray, (blur, blur), 0)

        cropped = img_blurred

        # cv2.imwrite('binary.jpg', cropped)

        # Image binary
        cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  threshold, adjustment)
        # cv2.imwrite('binary.jpg', cropped_threshold)

        # Operasi Erosi untuk mencegah adanya noise pada citra setelah melalui proses threshold.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))  # kernel untuk diterapkan pada morfologi
        eroded = cv2.erode(cropped_threshold, kernel, iterations=iterations)

        # Reverse
        inverse = self.inverse_colors(eroded)


        # Temukan kontur u/mendapatkan semua kontur yang terdapat pada image biner
        cnts = cv2.findContours(inverse.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]  #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)

        # Setiap kontur didekati membentuk poligon dan jika kontur berbentuk segiempat (memiliki 4 sisi), maka diprediksi menjadi LCD Tasbih Digital
        # dan kontur digambar menggunakan metode cv2.drawContours()
        TasbihCnt = None  # Saat ini kita tidak memiliki kontur Tasbih

        # Looping kontur (cnts) u/mendapatkan kontur yg sesuai
        for c in cnts:
            # Perkiraan kontur
            peri = cv2.arcLength(c, True)  # Menghitung keliling kontur
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Untuk mendekati bentuk

            # Jika perkiraan kontur memiliki 4 sisi, maka dapat di asumsikan bahwa Tasbih Digital telah ditemukan
            if len(approx) == 4:  # Pilih kontur dengan 4 sisi
                TasbihCnt = approx  # Ini kira-kira kontur Tasbih Digital

                # Dapatkan lokasi x, y, nilai width, height dari setiap kontur Tasbih
                x, y, w, h = cv2.boundingRect(c)
                self.ROI = img[y:y + h, x:x + w]
                break


        if TasbihCnt is not None:
            # Menggambar kontur yang dipilih pada image asli
            cv2.drawContours(image, [TasbihCnt], -1,
                             (0, 255, 0),  # green
                             3)

        self.drawROI = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if nama_file.find("lurus") != -1:
            angle = 0
            pixel_crop = 10
        elif nama_file.find("miring_atas") != -1:
            angle = -3
            pixel_crop = 15
        elif nama_file.find("miring_bawah") != -1:
            angle = 4
            pixel_crop = 15
        else:
            angle = 0
            pixel_crop = 10

        try:

            height, width = self.ROI.shape[:2]  #dividing height and width by 2 to get the center of the image
            center = (width/2, height/2)  #get the center coordinates of the image to create the 2D rotation matrix
            rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)  #using cv2.getRotationMatrix2D() to get the rotation matrix
            rotated_mat = cv2.warpAffine(self.ROI, rotation_mat, (width, height), flags=cv2.INTER_LINEAR)  # rotate the image using cv2.warpAffine

            self.skewness = rotated_mat
            # cv2.imwrite('rotate_sejajar.jpg', self.skewness)

            self.crop_image = rotated_mat[pixel_crop:rotated_mat.shape[0] - pixel_crop,
                              pixel_crop:rotated_mat.shape[1] - pixel_crop]  # memotong pixel image

            # print("height [0]: ", self.crop_image.shape[0])
            # print("width [1]: ", self.crop_image.shape[1])

            r = self.crop_image.shape[0] / float(90)
            dim = (int(self.crop_image.shape[1] / r), 90)  # dimensi
            img_r = cv2.resize(self.crop_image, dim, interpolation=cv2.INTER_AREA)

            # print("height_new [0]: ", img_r.shape[0])
            # print("width_new [1]: ", img_r.shape[1])

            # plt.imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
            # plt.title('Resize')
            # plt.show()


            return img_r
        except:
            return img


    def process_image(self, blur, threshold, adjustment, erode, iterations):

        crop_image = self.crop(self.file_name, self.image, blur, threshold, adjustment, erode, iterations)

        self.img_crop = crop_image
        debug_images = []
        debug_images.append(('Original Crop', self.img_crop))
        cv2.imwrite('img_crop.jpg', self.img_crop)

        # Convert to grayscale
        self.img2gray = cv2.cvtColor(self.img_crop, cv2.COLOR_BGR2GRAY)
        debug_images.append(('Grayscale', self.img2gray))


        # Blur to reduce noise
        self.img_blurred = cv2.GaussianBlur(self.img2gray, (blur, blur), 0)
        debug_images.append(('Blurred', self.img_blurred))

        cropped = self.img_blurred

        # Image binary
        self.cropped_threshold = cv2.adaptiveThreshold(cropped,  # input image
                                                       255,  # make pixels that pass the threshold full white
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       # use gaussian rather than mean, seems to give better results
                                                       cv2.THRESH_BINARY,
                                                       # invert so foreground will be white, background will be black
                                                       threshold,
                                                       # size of a pixel neighborhood used to calculate threshold value
                                                       adjustment)  # constant subtracted from the mean or weighted mean
        debug_images.append(('Cropped Threshold', self.cropped_threshold))


        # Erode the lcd digits to make them continuous for easier contouring
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
        self.eroded = cv2.erode(self.cropped_threshold, kernel, iterations=iterations)
        debug_images.append(('Eroded', self.eroded))


        # Reverse the image to so the white text is found when looking for the contours
        self.inverse = self.inverse_colors(self.eroded)
        debug_images.append(('Inversed', self.inverse))

        # Try out close
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.closed = cv2.morphologyEx(self.inverse, cv2.MORPH_CLOSE, kernel1)
        debug_images.append(('Closed', self.closed))


        # Find the lcd digit contours
        contours, _ = cv2.findContours(self.closed.copy(),
                                       # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                       cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                       cv2.CHAIN_APPROX_NONE)  # get contours

        # Assuming we find some, we'll sort them in order left -> right
        if len(contours) > 0:
            contours, _ = self.sort_contours(contours)

        potential_digits = []
        total_digit_height = 0
        total_digit_y = 0

        # Aspect ratio for all non 1 character digits
        desired_aspect = 0.6
        # Aspect ratio for the "1" digit
        digit_one_aspect = 0.25
        # The allowed buffer in the aspect when determining digits
        aspect_buffer = 0.15
        # aspect_buffer = 0.2  # yang benar

        # Loop over all the contours collecting potential digits
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            aspect = float(w) / h
            size = w * h

            # If it's small and it's not a square, kick it out
            if size < 2000 and (aspect < 1 - aspect_buffer and aspect > 1 + aspect_buffer):
                continue

            # Ignore any rectangles where the width is greater than the height
            if w > h:
                if self.debug:
                    cv2.rectangle(self.img_crop,  # draw rectangle on original self.img_crop
                                  (x, y),  # upper left corner
                                  (x + w, y + h),  # lower right corner
                                  (0, 0, 255),  # red
                                  2)  # thickness
                continue

            # If the contour is of decent size and fits the aspect ratios we want, we'll save it
            if ((size > 2000 and aspect >= desired_aspect - aspect_buffer and aspect <= desired_aspect + aspect_buffer) or
                (size > 1000 and aspect >= digit_one_aspect - aspect_buffer and aspect <= digit_one_aspect + aspect_buffer)):
                # Keep track of the height and y position so we can run averages later
                total_digit_height += h
                total_digit_y += y
                potential_digits.append(contour)
            else:
                if self.debug:
                    cv2.rectangle(self.img_crop, (x, y), (x + w, y + h),
                                  (0, 255, 0),  # green
                                  2)

        avg_digit_height = 0
        avg_digit_y = 0
        potential_digits_count = len(potential_digits)


        # Calculate the average digit height and y position so we can determine what we can throw out
        if potential_digits_count > 0:

            avg_digit_height = float(total_digit_height) / potential_digits_count
            avg_digit_y = float(total_digit_y) / potential_digits_count
            if self.debug:
                print("Average Digit and Y: " + str(avg_digit_height) + " and " + str(avg_digit_y))

        output = ''
        ix = 0

        # Loop over all the potential digits and see if they are candidates to run through KNN to get the digit
        for pot_digit in potential_digits:
            [x, y, w, h] = cv2.boundingRect(pot_digit)

            # # Apakah kontur ini cocok dengan rata-rata
            if h <= avg_digit_height * 1.2 and h >= avg_digit_height * 0.2 and y <= avg_digit_height * 1.2 and y >= avg_digit_y * 0:
                # Crop the contour off the eroded image
                cropped = self.eroded[y:y + h, x:x + w]
                cropped2 = self.img_crop[y:y + h, x:x + w]

                # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                # plt.title('Digit')
                # plt.show()

                # print("w: ", w)
                # print("h: ", h)

                # Draw a rect around it
                cv2.rectangle(self.img_crop, (x, y), (x + w, y + h), (255, 0, 0), 2)

                debug_images.append(('digit' + str(ix), cropped))

                # Call into the KNN to determine the digit
                digit = self.predict_digit(cropped)
                if self.debug:
                    print("Digit: " + digit)
                output += digit

                # Helper code to write out the digit image file for use in KNN training
                if self.write_digits:
                    __, full_file = os.path.split(self.file_name)
                    file_name = full_file.split('.')
                    # crop_file_path = CROP_DIR + '/' + digit + '_' + file_name[0] + '_crop_' + str(ix) + '.png'
                    crop_file_path = CROP_DIR + '/' + '_' + file_name[0] + '_crop_' + str(ix) + '.png'
                    cv2.imwrite(crop_file_path, cropped2)

                ix += 1

            else:
                if self.debug:
                    cv2.rectangle(self.img_crop, (x, y), (x + w, y + h),
                                  (66, 146, 244),  # orange
                                  2)


        # Log some information
        if self.debug:
            print("Potential Digits: " + str(len(potential_digits)))
            print("String: " + output)

        return debug_images, output

    def process_image_manual(self, blur, threshold, adjustment, erode, iterations):

        self.img = self.original.copy()

        # plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        # plt.title('Resize')
        # plt.show()

        debug_images = []

        debug_images.append(('Original', self.original))

        # Convert to grayscale
        self.img2gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        debug_images.append(('Grayscale', self.img2gray))

        # Blur to reduce noise
        self.img_blurred = cv2.GaussianBlur(self.img2gray, (blur, blur), 0)
        debug_images.append(('Blurred', self.img_blurred))

        cropped = self.img_blurred

        # Image binary
        self.cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, threshold, adjustment)
        debug_images.append(('Cropped Threshold', self.cropped_threshold))

        # Erode the lcd digits to make them continuous for easier contouring
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
        self.eroded = cv2.erode(self.cropped_threshold, kernel, iterations=iterations)
        debug_images.append(('Eroded', self.eroded))

        # Reverse the image to so the white text is found when looking for the contours
        self.inverse = self.inverse_colors(self.eroded)
        debug_images.append(('Inversed', self.inverse))

        # Try out close
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.closed = cv2.morphologyEx(self.inverse, cv2.MORPH_CLOSE, kernel1)
        debug_images.append(('Closed', self.closed))

        # Find the lcd digit contours
        contours, _ = cv2.findContours(self.closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

        # Assuming we find some, we'll sort them in order left -> right
        if len(contours) > 0:
            contours, _ = self.sort_contours(contours)

        potential_digits = []
        total_digit_height = 0
        total_digit_y = 0

        # Aspect ratio for all non 1 character digits
        desired_aspect = 0.6
        # Aspect ratio for the "1" digit
        digit_one_aspect = 0.25
        # Buffer yang diizinkan dalam aspek saat menentukan digit
        # aspect_buffer = 0.15
        aspect_buffer = 0.2  # yang benar

        # Loop over all the contours collecting potential digits
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            aspect = float(w) / h
            size = w * h

            # If it's small and it's not a square, kick it out
            if size < 2000 and (aspect < 1 - aspect_buffer and aspect > 1 + aspect_buffer):
                continue

            # Ignore any rectangles where the width is greater than the height
            if w > h:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                continue

            # # If the contour is of decent size and fits the aspect ratios we want, we'll save it
            if ((size > 2000 and aspect >= desired_aspect - aspect_buffer and aspect <= desired_aspect + aspect_buffer) or
                    (size > 1000 and aspect >= digit_one_aspect - aspect_buffer and aspect <= digit_one_aspect + aspect_buffer)):
                # Pantau ketinggian dan posisi y sehingga kita dapat menjalankan rata-rata nanti
                total_digit_height += h
                total_digit_y += y
                potential_digits.append(contour)
            else:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        avg_digit_height = 0
        avg_digit_y = 0
        potential_digits_count = len(potential_digits)


        # Hitung tinggi rata-rata digit dan posisi y sehingga kita dapat menentukan apa yang dapat kita buang
        if potential_digits_count > 0:
            avg_digit_height = float(total_digit_height) / potential_digits_count
            avg_digit_y = float(total_digit_y) / potential_digits_count
            if self.debug:
                print("Average Digit and Y: " + str(avg_digit_height) + " and " + str(avg_digit_y))

        output = ''
        ix = 0

        # Loop over all the potential digits and see if they are candidates to run through KNN to get the digit
        for pot_digit in potential_digits:
            [x, y, w, h] = cv2.boundingRect(pot_digit)

            # Does this contour match the averages
            if h <= avg_digit_height * 1.2 and h >= avg_digit_height * 0.2 and y <= avg_digit_height * 1.2 and y >= avg_digit_y * 0:
                # Crop the contour off the eroded image
                cropped = self.eroded[y:y + h, x:x + w]
                cropped1 = self.img[y:y + h, x:x + w]

                # Draw a rect around it
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                debug_images.append(('digit' + str(ix), cropped))

                # Call into the KNN to determine the digit
                digit = self.predict_digit(cropped)
                if self.debug:
                    print("Digit: " + digit)
                output += digit

                # Helper code to write out the digit image file for use in KNN training
                if self.write_digits:
                    __, full_file = os.path.split(self.file_name)
                    file_name = full_file.split('.')
                    # crop_file_path = CROP_DIR + '/' + digit + '_' + file_name[0] + '_crop_' + str(ix) + '.png'
                    crop_file_path = CROP_DIR + '/' + '_' + file_name[0] + '_crop_' + str(ix) + '.png'
                    cv2.imwrite(crop_file_path, cropped1)

                ix += 1
            else:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (66, 146, 244), 2)


        # Log some information
        if self.debug:
            print("Potential Digits: " + str(len(potential_digits)))
            print("String: " + output)

        return debug_images, output

    def process_image_digit(self, blur, threshold, adjustment, erode, iterations):

        self.img = self.image

        debug_images = []

        debug_images.append(('Original', self.img))

        # Convert to grayscale
        self.img2gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        debug_images.append(('Grayscale', self.img2gray))

        # Blur to reduce noise
        self.img_blurred = cv2.GaussianBlur(self.img2gray, (blur, blur), 0)
        debug_images.append(('Blurred', self.img_blurred))

        cropped = self.img_blurred

        # Image binary
        self.cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, threshold, adjustment)
        debug_images.append(('Cropped Threshold', self.cropped_threshold))

        # Erode the lcd digits to make them continuous for easier contouring
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
        self.eroded = cv2.erode(self.cropped_threshold, kernel, iterations=iterations)
        debug_images.append(('Eroded', self.eroded))

        # Call into the KNN to determine the digit
        digit = self.predict_digit(self.eroded)
        if self.debug:
            print("Digit: " + digit)
        output = digit

        return debug_images, output

    # Predict the digit from an image using KNN
    def predict_digit(self, digit_tasbih):
        # Resize the image
        imgROIResized = cv2.resize(digit_tasbih, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  #resize gambar

        # plt.imshow(cv2.cvtColor(imgROIResized, cv2.COLOR_BGR2RGB))
        # plt.title('24x42')
        # plt.show()

        # print("digit tasbih: ", imgROIResized)

        # Reshape the image
        X_test = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        # Convert it to floats
        X_test = np.float32(X_test)

        X_test = pca.transform(X_test)

        # y_pred = self.knn.predict(X_test)
        y_pred = self.pca_knn.predict(X_test)
        predict_digit = str(chr(int(y_pred)))

        return predict_digit




