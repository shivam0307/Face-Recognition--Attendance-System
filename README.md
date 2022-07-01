# Face-Recognition-Attendance-System
Application of Deep Neural Networks ( CNN Based dlib and OpenCV modules ) in recognizing Faces with high accuracy and marking Attendance.

Have you noticed that Facebook has developed an uncanny ability to recognize your friends in your photographs? In the old days, Facebook used to make you to tag your friends in photos by clicking on them and typing in their name. Now as soon as you upload a photo, Facebook tags everyone for you like magic:
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

1. A HOG + Linear SVM face detector that is accurate and computationally efficient. (Applies dlib’s HOG + Linear SVM face detector)
2. A Max-Margin (MMOD) CNN face detector that is both highly accurate and very robust, capable of detecting faces from varying viewing angles, lighting conditions, and occlusion. (Utilizes dlib’s MMOD CNN face detector)

### Which method should I choose?

The HOG + Linear SVM face detector will be faster than the MMOD CNN face detector but will also be less accurate as HOG + Linear SVM does not tolerate changes in the viewing angle rotation.

For more robust face detection, use dlib’s MMOD CNN face detector. This model requires significantly more computation (and is thus slower) but is much more accurate and robust to changes in face rotation and viewing angle.

Furthermore, if you have access to a GPU, you can run dlib’s MMOD CNN face detector on it, resulting in real-time face detection speed. The MMOD CNN face detector combined with a GPU is a match made in heaven — you get both the accuracy of a deep neural network along with the speed of a less computationally expensive model.














































