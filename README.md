# Face-Recognition-Attendance-System
This Project is an Application of Deep Neural Networks ( CNN Based dlib and OpenCV modules ) in recognizing Faces with high accuracy and marking Attendance. We will explore all the important concepts required for this projects and use those concepts to execute our System.

But before that...Have you noticed that Facebook has developed an uncanny ability to recognize your friends in your photographs? In the old days, Facebook used to make you to tag your friends in photos by clicking on them and typing in their name. Now as soon as you upload a photo, Facebook tags everyone for you like magic:
This technology is called face recognition. Facebook’s algorithms are able to recognize your friends’ faces after they have been tagged only a few times.

This technology is called face recognition. Facebook’s algorithms are able to recognize your friends’ faces after they have been tagged only a few times. It’s pretty amazing technology — Facebook can recognize faces with 98% accuracy which is pretty much as good as humans can do!

## What is Face Recognition?
Face recognition is the task of identifying an already detected object as a known or unknown face. Often the problem of face recognition is confused with the problem of face detection. Face Recognition on the other hand is to decide if the "face" is someone known, or unknown, using for this purpose a database of faces in order to validate this input face.

![image](https://user-images.githubusercontent.com/107324616/176882807-f2c48b99-2ff8-4c33-88f8-8b486e997726.png)

## Different Methods for Face Recognition

![image](https://user-images.githubusercontent.com/107324616/176885598-76f55146-baa7-475b-a873-3b3a869fc184.png)

But Before all these, we should have basic understnding of Machine Learning & Deep Learning Concepts!

## What is Machine Learning?

Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed. ML is one of the most exciting technologies that one would have ever come across. As it is evident from the name, it gives the computer that makes it more similar to humans: The ability to learn. Machine learning is actively being used today, perhaps in many more places than one would expect.

**Supervised learning** is when the model is getting trained on a labelled dataset. A labelled dataset is one that has both input and output parameters. In this type of learning both training and validation, datasets are labelled.

**Unsupervised learning** is the training of a machine using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance. Here the task of the machine is to group unsorted information according to similarities, patterns, and differences without any prior training of data.

![image](https://user-images.githubusercontent.com/107324616/176885992-928d2320-50ae-4fa6-921b-509ac938ab8e.png)

 ## An Introduction with Deep Neural Networks
 
Neural networks are the core of deep learning, a field that has practical applications in many different areas. Today neural networks are used for image classification, speech recognition, object detection, etc.
Deep structured learning or hierarchical learning or deep learning in short is part of the family of machine learning methods which are themselves a subset of the broader field of Artificial Intelligence.

**Deep learning is a class of machine learning algorithms that use several layers of nonlinear processing units for feature extraction and transformation. Each successive layer uses the output from the previous layer as input.**
Deep neural networks, deep belief networks and recurrent neural networks have been applied to fields such as computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, and bioinformatics where they produced results comparable to and in some cases better than human experts have.

![image](https://user-images.githubusercontent.com/107324616/176886618-87bd8e37-5db6-49bb-871a-45700e3e1511.png)

**1. Artificial Neural Network (ANN)**

![image](https://user-images.githubusercontent.com/107324616/176887400-13228a4d-1ebc-42cd-930b-903d5ca08580.png)

**2. Convolutional Neural Networkss(CNN)**

![image](https://user-images.githubusercontent.com/107324616/176887906-6f4a4552-473e-41b8-9375-1ddd3507496c.png)

**3. Recurrent Neural Network (RNN)**

![image](https://user-images.githubusercontent.com/107324616/176888371-f3dd94bf-6593-4876-bc0a-ff4d8db8db76.png)

## Convolutional Neural Networks (Structure & Working)

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.

### Why CNN?

![image](https://user-images.githubusercontent.com/107324616/176891095-c2a65c87-1d85-445c-8d7e-da74a28b0fda.png)

A ConvNet is able to successfully capture the Spatial and Temporal dependencies in an image through the application of relevant filters. The role of the ConvNet is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction.The different layers of a CNN. There are four types of layers for a convolutional neural network: **the convolutional layer, the pooling layer, the ReLU correction layer and the fully-connected layer**.


![image](https://user-images.githubusercontent.com/107324616/176891756-d1abbaed-d8b4-418d-951d-adbe570b40ba.png)

### Some Important Standard CNNs

**1. LeNet-5**

![image](https://user-images.githubusercontent.com/107324616/176892127-04b7eb5c-7535-48d1-93f2-e04dfbf03ad7.png)

**2. AlexNet**

![image](https://user-images.githubusercontent.com/107324616/176892274-2b344f4c-1f3b-4536-978b-2e339f59faba.png)

**3. VGG-16**

![image](https://user-images.githubusercontent.com/107324616/176892417-f45aaeda-4d0e-49eb-a7b1-dacdececc58f.png)

**4. ResNet**

![image](https://user-images.githubusercontent.com/107324616/176892724-f5f394e3-2b59-4ff3-93ab-d2896a4c618c.png)

## dlib Library

The dlib library is arguably one of the most utilized packages for face recognition. A Python package appropriately named face_recognition wraps dlib’s face recognition functions into a simple, easy to use API. Dlib is an open source library of machine learning, which contains many algorithms of machine learning and is very convenient to use.

This includes two face detection methods built into the library:

1. A HOG(Histogram of Oriented Gradients) + Linear SVM face detector that is accurate and computationally efficient. (Applies dlib’s HOG + Linear SVM face detector)
2. A Max-Margin (MMOD) CNN face detector that is both highly accurate and very robust, capable of detecting faces from varying viewing angles, lighting conditions, and occlusion. (Utilizes dlib’s MMOD CNN face detector)

### Which method should I choose?

The HOG + Linear SVM face detector will be faster than the MMOD CNN face detector but will also be less accurate as HOG + Linear SVM does not tolerate changes in the viewing angle rotation.

For more robust face detection, use dlib’s MMOD CNN face detector. This model requires significantly more computation (and is thus slower) but is much more accurate and robust to changes in face rotation and viewing angle.

Furthermore, if you have access to a GPU, you can run dlib’s MMOD CNN face detector on it, resulting in real-time face detection speed. The MMOD CNN face detector combined with a GPU is a match made in heaven — you get both the accuracy of a deep neural network along with the speed of a less computationally expensive model.

## Finding all Faces

To find faces in an image, we’ll start by making our image black and white because we don’t need color data to find faces. Then we’ll look at every single pixel in our image one at a time. For every single pixel, we want to look at the pixels that directly surrounding it. Our goal is to figure out how dark the current pixel is compared to the pixels directly surrounding it. Then we want to draw an arrow showing in which direction the image is getting darker. If you repeat that process for every single pixel in the image, you end up with every pixel being replaced by an arrow. These arrows are called **gradients** and they show the flow from light to dark across the entire image.

This might seem like a random thing to do, but there’s a really good reason for replacing the pixels with gradients. If we analyze pixels directly, really dark images and really light images of the same person will have totally different pixel values. But by only considering the direction that brightness changes, both really dark images and really bright images will end up with the same exact representation. That makes the problem a lot easier to solve!

But saving the gradient for every single pixel gives us way too much detail. We end up missing the forest for the trees. It would be better if we could just see the basic flow of lightness/darkness at a higher level so we could see the basic pattern of the image.

![image](https://user-images.githubusercontent.com/107324616/176897297-ef387c14-388b-4a5a-b1d9-ecc5107180f2.png)


To do this, we’ll break up the image into small squares of 16x16 pixels each. In each square, we’ll count up how many gradients point in each major direction (how many point up, point up-right, point right, etc…). Then we’ll replace that square in the image with the arrow directions that were the strongest.

The end result is we turn the original image into a very simple representation that captures the basic structure of a face in a simple way.

## Posing and Projecting Faces

To do this, we are going to use an algorithm called face landmark estimation. The basic idea is we will come up with 68 specific points (called landmarks) that exist on every face — the top of the chin, the outside edge of each eye, the inner edge of each eyebrow, etc. Then we will train a machine learning algorithm to be able to find these 68 specific points on any face.

![image](https://user-images.githubusercontent.com/107324616/176899120-a0733703-20c7-41e0-b861-4d29385395e2.png)


Now that we know were the eyes and mouth are, we’ll simply rotate, scale and shear the image so that the eyes and mouth are centered as best as possible. We won’t do any fancy 3d warps because that would introduce distortions into the image. We are only going to use basic image transformations like rotation and scale that preserve parallel lines (called Affine Transformations).

![image](https://user-images.githubusercontent.com/107324616/176899151-f9cd76cc-0d3c-401c-a2f1-11b1cc11e94e.png)

Now no matter how the face is turned, we are able to center the eyes and mouth are in roughly the same position in the image. This will make our next step a lot more accurate.

## Encoding Faces

It turns out that the measurements that seem obvious to us humans (like eye color) don’t really make sense to a computer looking at individual pixels in an image. Researchers have discovered that the most accurate approach is to let the computer figure out the measurements to collect itself. Deep learning does a better job than humans at figuring out which parts of a face are important to measure.

The solution is to train a Deep Convolutional Neural Network (just like we did in Part 3). But instead of training the network to recognize pictures objects like we did last time, we are going to train it to generate 128 measurements for each face.

The training process works by looking at 3 face images at a time:

1. Load a training face image of a known person
2. Load another picture of the same known person
3. Load a picture of a totally different person

After repeating this step millions of times for millions of images of thousands of different people, the neural network learns to reliably generate 128 measurements for each person. Any ten different pictures of the same person should give roughly the same measurements.

Machine learning people call the 128 measurements of each face an embedding. The idea of reducing complicated raw data like a picture into a list of computer-generated numbers comes up a lot in machine learning (especially in language translation).

This process of training a convolutional neural network to output face embeddings requires a lot of data and computer power. Even with an expensive NVidia Telsa video card, it takes about 24 hours of continuous training to get good accuracy.

But once the network has been trained, it can generate measurements for any face, even ones it has never seen before! So this step only needs to be done once. So all we need to do ourselves is run our face images through their pre-trained network to get the 128 measurements for each face. Here’s the measurements for our test image:
![image](https://user-images.githubusercontent.com/107324616/176899726-b7c5dc9d-1a17-46d4-b345-f69a3b682c81.png)

## Finding the person’s name from the encoding

This last step is actually the easiest step in the whole process. All we have to do is find the person in our database of known people who has the closest measurements to our test image.

You can do that by using any basic machine learning classification algorithm. No fancy deep learning tricks are needed. We’ll use a simple linear SVM classifier, but lots of classification algorithms could work.

All we need to do is train a classifier that can take in the measurements from a new test image and tells which known person is the closest match. Running this classifier takes milliseconds. The result of the classifier is the name of the person!

### Let’s review the steps we followed:

1. Encode a picture using the HOG algorithm to create a simplified version of the image. Using this simplified image, find the part of the image that most looks like a generic HOG encoding of a face.
2. Figure out the pose of the face by finding the main landmarks in the face. Once we find those landmarks, use them to warp the image so that the eyes and mouth are centered.
3. Pass the centered face image through a neural network that knows how to measure features of the face. Save those 128 measurements.
4. Looking at all the faces we’ve measured in the past, see which person has the closest measurements to our face’s measurements. That’s our match!

# Execution of our Program

## Required Dependencies

1. [numpy](https://github.com/numpy/numpy)
2. [OpenCV](https://github.com/opencv/opencv)
3. [cmake](https://github.com/Kitware/CMake)
4. [dlib](https://github.com/davisking/dlib)
5. [face_recognition](https://github.com/ageitgey/face_recognition)

## Working of the System

### Step 1: Importing Images
As we have imported before we can use the same face_recognition.load_image_file() function to import our images. But when we have multiple images, importing them individually can become messy. Therefore we will write a script to import all images in a given folder at once. For this we will need the os library so we will import that first. We will store all the images in one list and their names in another.

### Step 2: Compute Encodings
Now that we have a list of images we can iterate through those and create a corresponding encoded list for known faces. To do this we will create a function. As earlier we will first convert it into RGB and then find its encoding using the face_encodings( ) function. Then we will append each encoding to our list. 
Now we can simply call the function with the images list as the input arguments.

### Step 3: The While loop
The while loop is created to run the webcam. But before the while loop we have to create a video capture object so that we can grab frames from the webcam. 
1. First we will read the image from the webcam and then resize it to quarter the size. This is done to increase the speed of the system. Even though the image being used is 1/4 th of the original, we will still use the original size while displaying. Next we will convert it to RGB.
2. Once we have the webcam frame we will find all the faces in our image. The face_locations function is used for this purpose. Later we will find the face_encodings as well.
3.  Now we can match the current face encodings to our known faces encoding list to find the matches. We will also compute the distance. This is done to find the best match in case more than one face is detected at a time. Once we have the list of face distances we can find the minimum one, as this would be the best match. Now based on the index value we can determine the name of the person and display it on the original Image.

### Step 4: Marking Attendance
Lastly we are going to add the automated attendance code. We will start by writing a function that requires only one input which is the name of the user. First we open our Attendance file which is in csv format. Then we read all the lines and iterate through each line using a for loop. Next we can split using comma ‘,’. This will allow us to get the first element which is the name of the user. If the user in the camera already has an entry in the file then nothing will happen. On the other hand if the user is new then the name of the user along with the current time stamp will be stored. We can use the datetime class in the date time package to get the current time.

## Results Obtained

[Dataset Used for training](https://drive.google.com/open?id=1MoqRPI1BS-DZQkRhYRitLiCI6iJTYacL)

https://drive.google.com/open?id=1Pr9jMLD-2BYENbe_t-2btUgb9G2cYe-O

https://drive.google.com/open?id=1Ut4Jxw5xZrg4REHKuOZMiK96mwgyilOI

https://drive.google.com/open?id=1GAHlRr9J2YYbnl6KVuQv7jSYkyeoeYgh

https://drive.google.com/open?id=1AO1nANSWt_68OiUGeW209TBup9O3Qexs

https://drive.google.com/open?id=1PCEFNe576go8v-rgLLdPiN24w68Kee91

https://drive.google.com/open?id=1qNOCGsXUALdj7WNmEtgRM6uJ1k7k23lY

### Auto Update in Database:

[Screenshot of CSV File](https://drive.google.com/open?id=1BMlvzuECEk_UYGZzS1Ipb29ELX9Lz4oA)

### Excel Record for Attendance

[Screenshot of Excel Record](https://drive.google.com/open?id=1MoqRPI1BS-DZQkRhYRitLiCI6iJTYacL)

# References

1. [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning, by Adam Geitgey](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
2. [Face recognition with OpenCV, Python, and deep learning, by Adrian Rosebrock](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
3. [Face detection with dlib (HOG and CNN), by Adrian Rosebrock](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)
4. [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way, by Sumit Saha](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
5. A Face Recognition Library using Convolutional Neural Networks, Paperwork by Leonardo Blanger (URI University, Erechim - Brazil) and Alison R. Panisson ( Pontifical University of Rio Grande do Sul, Porto Alegre - Brazil)
6. [An improved face recognition algorithm and its application in attendance management system, by Serign Modou Bah and Fang Ming](https://www.sciencedirect.com/science/article/pii/S2590005619300141#!)















































