## COMP4102 - Final Project

#### Project team:
Erica Warriner 101002942  
Daniil Kulik 101138752

#### Title:  
Panorama creation with subsequent manual object removal

#### Summary: 
The ultimate goal of this project is to create a tool that can produce panoramas from 4-5 images. Subsequently, a user will be able to choose an object to remove from the panorama. This project will work with two computer vision problems: (1) image stitching to produce a panorama and (2) inpainting. 

#### Background:

#### The Challenge:
The task of image stitching to produce a panorama (see Figure 1) can be completed with moderate difficulty using OpenCV functions. This task is made more complex with the project requirement of producing panoramas from 4-5 images, rather than using only 2 source images. With these additions, there will be several "middle" images that will need to be matched to descriptors on both its left and right neighbours. Not only will the added images increase the complexity of image stitching, but they will also create a higher potential for error due to slight variations between the features of each image (e.g. brightness, rotation, etc.). The process of creating a panorama for 4-5 images will require more time than the traditional 2-image panorama, to ensure a seamless result.

<center><src="https://media.discordapp.net/attachments/801117503542525993/806290415699361802/image_stitching_opencv_header.png"></center>

Inpainting of small regions within a single image is fairly simple with the use of pre-existing OpenCV functions. The challenge for this task will be to combine the methods of image stitching and inpainting to support object removal on a multi-image panorama. The object removal feature will be implemented with a desktop application to allow the user to manually draw a contour of the object. This introduces another challenge of building an interactive application that can adequately display contours and integrate the user input with OpenCV object removal.

By completing this project, we hope to learn how to:
- detect interest points and extract descriptors,
- match descriptors between 4+ images,
- compute homography to align 4+ images,
- create an interactive application for object removal, and
- perform inpainting with OpenCV.

#### Goals and Deliverables: 
The ultimate goal of this project is to build a program that is capable of producing a panorama image by combining several images (4-5) of the same scene. Additionally, the program will provide users with a feature for object removal from the scene. The users will be able to upload several images of a scene into the program, create a panorama, and manually select a contour of the object to be removed from the scene. The removed object spot will be inpainted with a suitable background. 

Extra goals include extending panorama generation in more than one direction. Traditional panorama creation tools allow horizontal or vertical panoramas. If time permits, diagonal panoramas will be explored. Another possible extension is to create a mobile application with the same functionality that can work with limited computational resources.

// How realistic is it for your team to get what it needs to get done within theallotted time?  Remember you only have a few weeks to get this projectcompleted

The project evaluation will be based on several criteria:
1) Correct and seamless panorama creation
2) Correct and seamless object removal
3) Correct background inpainting at the removed object spot

#### Schedule:

March 31 - demo program is ready
