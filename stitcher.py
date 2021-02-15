# Imports
import numpy as np                       # Matrix operations
import imutils                           # Image processing functions
import cv2                               # OpenCV
from matplotlib import pyplot as plt     # Graph plotting

# Function for building panorama from array of images
def build_panorama(images):

    # Create dictionary for images info
    imageDict = {}

    # Iterate through images, adding info to nested dictionary
    for i in range(1, len(images) + 1):

        # Name image dictionary based on index
        index = "image_" + str(i)

        # Create nested dictionary for current image
        imageDict[index] = {}

        # Create variables for current image to simplify code
        thisImage = imageDict[index]
        thisImage["image"] = images[i-1]

        # Convert original image to RGB (for matplotlib compatibility)
        thisImage["image"] = cv2.cvtColor(thisImage["image"], cv2.COLOR_BGR2RGB)

        # Convert RGB image to greyscale to detect keypoints
        thisImage["grey"] = cv2.cvtColor(thisImage["image"], cv2.COLOR_RGB2GRAY)

        # Find keypoints using SIFT
        feature_finder = cv2.SIFT_create()
        thisImage["keypoints"] = feature_finder.detect(thisImage["grey"], None)

        # Find features using SIFT
        thisImage["keypoints"] = feature_finder.compute(thisImage["grey"],
                                                        thisImage["keypoints"])[0]
        thisImage["features"] = feature_finder.compute(thisImage["grey"],
                                                        thisImage["keypoints"])[1]

    # Iterate through images to build panorama
    for i in range(1, len(images) + 1):

        # Get current image dictionary based on index
        index = "image_" + str(i)

        # Create variable for current image dictionary to simplify code
        thisImage = imageDict[index]

        # Build panorama on right neighbour (if exists)
        if (i != len(images)):
            
            # Create variable for right neighbour to simplify code
            rightImage = imageDict["image_" + str(i+1)]

            # Create descriptor matcher
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            
            # Find 2 best matches for each descriptor
            matches = matcher.knnMatch(thisImage["features"],
                                       rightImage["features"],
                                       2)

            # Create array to store good matches
            good_matches = []
            
            # Iterate through each descriptor's matches
            for m in matches:

                # Check that there are two best matches for descriptor
                if (len(m) == 2):

                    # Ensure the distance between each point is within a 
                    # ratio of 0.75
                    if (m[0].distance < (m[1].distance * 0.75)):
                        good_matches.append((m[0].trainIdx, m[0].queryIdx))

def main():

    # Create array to hold sample images
    images = []

    # Read sample images from local directory
    images.append(cv2.imread(r'Images\Example_1\1.jpg'))
    images.append(cv2.imread(r'Images\Example_1\2.jpg'))
    images.append(cv2.imread(r'Images\Example_1\3.jpg'))
    images.append(cv2.imread(r'Images\Example_1\4.jpg'))
    images.append(cv2.imread(r'Images\Example_1\5.jpg'))
    images.append(cv2.imread(r'Images\Example_1\6.jpg'))

    # Build panorama using sample images
    build_panorama(images)

if __name__ == "__main__":
    main()
