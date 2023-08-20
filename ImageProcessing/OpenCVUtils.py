import cv2

def inverse_colors(img):
    img = (255-img)
    return img

def sort_contours(cnts, method="left-to-right"):
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

