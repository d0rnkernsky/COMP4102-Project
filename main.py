import stitcher as st
import object_removal as rm
import cv2 as cv
import os
import sys

stitch_path = 'stitch-result.jpg'
result_path = 'result.jpg'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Specify example by passing folder name. Ex: python main.py Example_5')
        exit(-1)

    example_folder = f'Images/{sys.argv[1]}'
    if not os.path.isdir(example_folder) or not os.path.exists(example_folder):
        print(f'Invalid example folder: {example_folder}')
        exit(-2)

    if os.path.exists(stitch_path):
        os.unlink(stitch_path)
    if os.path.exists(result_path):
        os.unlink(result_path)

    images = []
    # Read sample images from local directory
    for i in range(1, 5):
        images.append(cv.imread(f'{example_folder}/{i}.jpg'))

    # Build panorama using sample images
    stitch_result = st.build_panorama(images)

    # Convert RGB image to BGR (for opencv compatibility)
    result = cv.cvtColor(stitch_result, cv.COLOR_RGB2BGR)

    # Save panorama to file
    cv.imwrite(stitch_path, result)
    rm.remove(stitch_path)
