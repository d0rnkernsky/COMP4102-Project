## COMP4102 - Final Project

#### Project team:
Erica Warriner 101002942  
Daniil Kulik 101138752

#### Title:  
Panorama creation with subsequent object removal

#### Summary: 
The ultimate goal of this project is to create a tool that can produce panoramas from 4-5 images. Subsequently, a user will be able to choose an object to remove from the panorama. This project will work with two computer vision problems: (1) image stitching to produce a panorama and (2) inpainting. 

#### Background:

#### The Challenge:
The task of image stitching to produce a panorama (see Figure 1) can be completed with moderate difficulty using OpenCV functions. This task is made more complex with the project requirement of producing panoramas from 4-5 images, rather than using only 2 source images. With these additions, there will be several "middle" images that will need to be matched to descriptors on both its left and right neighbours. Not only will the added images increase the complexity of image stitching, but they will also create a higher potential for error due to slight variations between the features of each image (e.g. brightness, rotation, etc.). The process of creating a panorama for 4-5 images will require more time than the traditional 2-image panorama, to ensure a seamless result.

<br>
<p align="center">
  <img width="460" height="300" src="https://media.discordapp.net/attachments/801117503542525993/806290415699361802/image_stitching_opencv_header.png">
</p>
<p align="center">
  <i>Figure 1</i>. Image stitching of 3 images from the same scene to form a panorama. From "Image Stitching with OpenCV and Python" by Adrian Rosebrock, 2018, https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
</p>
<br>

Inpainting of small regions within a single image (see Figure 2) is fairly simple with the use of pre-existing OpenCV functions. The challenge for this task will be to combine the methods of image stitching and inpainting to support object removal on a multi-image panorama. The object removal feature will be implemented with a desktop application to allow the user to manually draw a contour of the object. This introduces another challenge of building an interactive application that can adequately display contours and integrate the user input with OpenCV object removal.

<br>
<p align="center">
  <img width="460" height="300" src="https://media.discordapp.net/attachments/801117503542525993/806290515372539935/inpainting.png">
</p>
<p align="center">
  <i>Figure 2</i>. Inpainting after removing a subject from an image. From "Texture Synthesis and Hole-Filling" by Derek Hoiem, 2019, https://courses.engr.illinois.edu/cs445/fa2019/lectures/Lecture%2007%20-%20Texture%20Synthesis%20-%20CP%20Fall%202019.pdf
</p>

By completing this project, we hope to learn how to:
- detect interest points and extract descriptors,
- match descriptors between 4+ images,
- compute homography to align 4+ images,
- create an interactive application for object removal, and
- perform inpainting with OpenCV.

#### Goals and Deliverables: 
The ultimate goal of this project is to build a program that is capable of producing a panorama image by combining several images (4-5) of the same scene. Additionally, the program will provide the user with a feature for object removal from the scene. The user will be able to upload several images of a scene into the program, create a panorama, and manually select a contour of the object to be removed from the scene. The removed object spot will be inpainted with a suitable background. 

Extra goals include extending panorama generation in more than one direction. Traditional panorama creation tools allow horizontal or vertical panoramas. If time permits, diagonal panoramas will be explored. Another possible extension is to create a mobile application with the same functionality that can work with limited computational resources.

// How realistic is it for your team to get what it needs to get done within theallotted time?  Remember you only have a few weeks to get this projectcompleted

The project evaluation will be based on several criteria:
1) Correct and seamless panorama creation
2) Correct and seamless object removal
3) Correct background inpainting at the removed object spot

#### Schedule:

March 31 - demo program is ready
