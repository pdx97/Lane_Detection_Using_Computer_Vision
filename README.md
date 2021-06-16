## Lane Detection Using Computer Vision algorithms via Hand Engineering. 
<p>This was the Project 1 of the 3 part project we did to understand how The Self Driving Cars Work. </p>
<br>
We Used various Computer Vision techniques to see how the Driverless Vehicle sees an Image and Processes it to navigate it self to be inside certain Region Of interest so Following the methods and steps  we Used </br>
<ul>
  <li><b>Step 1</b>: We Captured the Image using a Webcam  and  collected the image</li>
  <li><b>Step 2</b>:We Grayscaled  the Image using OpenCv as color increases complexity of the model. A grayscale image consists of colors white and black and their varying intensities.Brightness, contrast, edges, shapes, contours, texture,perspective and shadows can be worked upon without addressing color. Grayscaling is also a necessary pre-
processing step before we can run more powerful algorithms to isolate lines.. </li>
  <li><b>Step 3</b>: Applied Various Smoothening and Filtering Methods on it so that we can reduce the Noise on the Image </li>
  <li><b>Step 4</b>: Applied Edge Detection to identify edges in an image and discard all other data. </li>
  <li><b>Step 5</b> : Specified the Region Of Interest as helps us discard any lines outside of our desired region. One crucial assumption is that the camera remains in the same place across all images and the lanes are flat, therefore we can identify the critical region we are interested in.</li>
  <li><b>Step 6</b>: Applied Hough Lines Transformation which helped to extract lines and draw its own lines on them by identifying all points that lie on them </li>
  <li><b>Step 7</b> : We need to identify left and right lanes so that we can calculate the slope and calculate the angle at which the servo motor should rotate and we were able to identify our Lanes Properly .</li>
</ul>
![stack Overflow](http://lmsotfy.com/so.png)
  
  

