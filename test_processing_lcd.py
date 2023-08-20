import os
import time

from DisplayUtils.Colors import bcolors
from ImageProcessing import FrameProcessor, ProcessingVariables
import version_number as v

version = v.vm.version
std_height = 90

erode = ProcessingVariables.erode
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = ProcessingVariables.blur


test_folder = 'tests/tasbih_lcd'

frameProcessor = FrameProcessor(std_height, version, False, write_digits=False)

def test_img(path, expected, show_result=True):
    frameProcessor.set_image(path)
    (debug_images, calculated) = frameProcessor.process_image_manual(blur, threshold, adjustment, erode, iterations)

    if expected == calculated:
        if show_result:
            print(bcolors.OKBLUE + 'Testing: ' + path + ', Expected ' + expected + ' and got ' + calculated + bcolors.ENDC)
        return True
    else:
        if show_result:
            print(bcolors.FAIL + 'Testing: ' + path + ', Expected ' + expected + ' and got ' + calculated + bcolors.ENDC)
        return False



def get_expected_from_filename(filename):
    expected = filename.split('.')[0]
    expected = expected.replace('Z', '')
    return expected

def run_test(show_result=True):
    count = 0
    correct = 0

    # start_time = time.time()
    for file_name in os.listdir(test_folder):
        # Lewati file tersembunyi
        if not file_name.startswith('.'):
            count += 1
            expected = get_expected_from_filename(file_name)
            is_correct = test_img(test_folder + '/' + file_name, expected, show_result)
            if is_correct:
                correct += 1



    print("\nFile diuji: " + str(count))
    print("File benar: " + str(correct))
    acc = round(float(correct) / count * 100, 2)
    print("Uji parameter - erode: " + str(erode) + ", blur: " + str(blur) + ", adjust: " +
          str(adjustment) + ", thres: " + str(threshold) + ", iterations: " + str(iterations))
    print("Uji Akurasi: " + bcolors.BOLD + str(acc) + '%' + bcolors.ENDC)
    return acc

def main():
    start_time = time.time()
    acc = run_test()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()





