import cv2  # library untuk memproses gambar
import time  #

import version_number as v  # import versi pelatihan
from ImageProcessing import FrameProcessor
from DisplayUtils.TileDisplay import show_img, reset_tiles


window_name = 'Citra Tasbih Digital'

# file_name = 'tests/tasbih_asli/miring_atas/'
# file_name = 'tests/tasbih_asli/miring_bawah/20000.jpeg'
file_name = 'tests/tasbih_asli/lurus/10892.jpeg'


# Versi pelatihan saat ini
version = v.vm.version

erode = 2
threshold = 37
adjustment = 11
iterations = 3
blur = 3


std_height = 90


frameProcessor = FrameProcessor(std_height, version, True)

def process_image():
    reset_tiles()
    start_time = time.time()
    debug_images, output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)


    for image in debug_images:
        show_img(image[0], image[1])

    print("Processed image in %s seconds" % (time.time() - start_time))

    cv2.imshow(window_name, frameProcessor.drawROI)
    cv2.moveWindow(window_name, 600, 600)

def main():
    img_file = file_name
    frameProcessor.set_image(img_file)
    process_image()
    cv2.waitKey()

if __name__ == "__main__":
    main()






