# CV-Final-Project
### Portrait relighting on a new scene
* Our goal is to achieve background replacement while taking the lighting condition into account.  
  
  There are many applications that can replace the background of the original image with a new background. It seems that the person is photographed in the new scene. 
  However, if we only apply background replacement, we will find that the lighting condition between the source image and the target image is different.  
  
  Therefore, we want to extract the lighting feature of the target image and render it in the final result. By doing so, we hope that we can make the result more
  consistent with the conditions of the target image.

### Usage 
* Note： all input images must store in data\test\images, you can check results in result\trained  
`python relight.py --source_image [src_image] --light_image [target_image] --srcBG [src_background_image] --lightBG [target_background_image]`
  ex：`python relight.py --source_image [src.jpg] --light_image [target_light.jpg] --srcBG [src_bgr.jpg] --lightBG [target_bgr.jpg]`

