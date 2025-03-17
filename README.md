# Age-Gender-detection-using-AI-ML

## Introduction

In this Python project, we will use Deep Learning to accurately identify the gender and age of a person from a single image of a face. We will use models trained by Tal Hassner and Gil Levi. The predicted gender may be one of **Male** or **Female**, and the predicted age may fall into one of the following ranges:

- **(0 – 2)**
- **(4 – 6)**
- **(8 – 12)**
- **(15 – 20)**
- **(25 – 32)**
- **(38 – 43)**
- **(48 – 53)**
- **(60 – 100)**

*(8 nodes in the final softmax layer)*

Accurately guessing an exact age from a single image is challenging due to factors such as makeup, lighting, obstructions, and facial expressions. Therefore, we formulate this as a classification problem rather than a regression task.

## The CNN Architecture

The convolutional neural network for this project consists of:

- **3 Convolutional Layers:**
  - **Layer 1:** 96 nodes, kernel size 7
  - **Layer 2:** 256 nodes, kernel size 5
  - **Layer 3:** 384 nodes, kernel size 3

- **2 Fully Connected Layers:**
  - Each with 512 nodes

- **Final Output Layer:**
  - Softmax type layer for classification

## Project Workflow

To implement the project, we will:

1. **Detect faces**
2. **Classify into Male/Female**
3. **Classify into one of the 8 age ranges**
4. **Overlay the results on the image and display it**

## The Dataset

For this project, we will use the **Adience dataset**. This dataset:

- Is available in the public domain.
- Serves as a benchmark for face photos under various real-world conditions such as noise, lighting, pose, and appearance.
- Contains **26,580 photos** of **2,284 subjects** across eight age ranges.
- Is approximately **1GB** in size.

The models employed have been trained on this dataset. The images, collected from Flickr albums, are distributed under the Creative Commons (CC) license.

## Prerequisites

Before running the project, ensure you have the following installed:

- **OpenCV (cv2):**

  ```bash
  pip install opencv-python
  ```

- Other required packages include `math` and `argparse`, which are part of the standard Python library.

## Steps for Practicing Gender and Age Detection

### 1. Setup

- **Download** the project zip file.
- **Unzip** the file and place its contents in a directory (e.g., `gad`).

#### Contents of the Zip File

- `opencv_face_detector.pbtxt`
- `opencv_face_detector_uint8.pb`
- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`
- A few sample pictures for testing

> **Note:**  
> - The `.pb` file is a protobuf file containing the graph definition and trained weights for face detection.  
> - The `.pbtxt` file holds the same in text format (TensorFlow).  
> - For age and gender detection, the `.prototxt` files define the network configuration, and the `.caffemodel` file contains the trained model weights.

### 2. Argument Parsing

- Use the `argparse` library to create an argument parser.
- Parse the command-line argument for the image path to classify gender and age.

### 3. Initialize Models

- Initialize the protocol buffers and load the models for face, age, and gender detection.

### 4. Set Mean Values and Labels

- Set the mean values for preprocessing.
- Define the lists for age ranges and gender labels.

### 5. Load the Networks

- Use the `readNet()` method to load the networks:
  - The first parameter provides the trained weights.
  - The second parameter contains the network configuration.

### 6. Capture Video Stream (Optional)

- Capture a video stream if you wish to classify live input from a webcam.
- Set the padding to **20**.

### 7. Process the Video Stream

- Continuously read frames until a key is pressed.
- Use `waitKey()` from OpenCV (`cv2`) to handle frame processing.

### 8. Face Detection

- Call the `highlightFace()` function with the `faceNet` model and the current frame.
- The function returns:
  - `resultImg`: The image with detected faces highlighted.
  - `faceBoxes`: A list containing coordinates of detected faces.

**Detection Process:**

- Create a shallow copy of the frame.
- Retrieve the image’s dimensions.
- Create a blob from the image copy.
- Set the blob as input and perform a forward pass through the network.
- Loop over potential detections (e.g., indices 0 to 127) and, if the confidence (between 0 and 1) exceeds **0.7**, record the face coordinates (`x1, y1, x2, y2`) and append them to `faceBoxes`.
- Draw rectangles on the image for each set of coordinates.

### 9. Gender and Age Classification

- For each detected face in `faceBoxes`:
  - Extract the face region and create a 4-dimensional blob (with scaling, resizing, and mean subtraction).
  - Feed the blob to the gender detection network and choose the class with the higher confidence as the predicted gender.
  - Repeat the process for age detection.

### 10. Display Results

- Overlay the predicted gender and age on the image.
- Display the final image using `imshow()` from OpenCV.

## Additional Insights

> **DO YOU KNOW:**  
> Data scientists require a good knowledge of Python.  
> **What are you waiting for? Start learning Python now!!**

## Python Project Examples for Gender and Age Detection

Try out the gender and age classifier on your own images using the command prompt. Below are some example commands and their outputs:

### Example 1

```bash
python project_ideas.py --image path/to/image.jpg
```
![image](https://github.com/user-attachments/assets/3635f1c7-5e18-4fee-b67c-da0e8435550b)


**Output:**
![image](https://github.com/user-attachments/assets/5aa58f8a-ac36-496f-b4e6-0b3a3d7b15ba)

```
python open source project
```

### Example 2

```bash
python projects_in_python.py --image path/to/image.jpg
```
![image](https://github.com/user-attachments/assets/29136137-dbfa-4eeb-9d20-f42b93fff46f)

**Output:**
![image](https://github.com/user-attachments/assets/12286e0e-e21b-4d5c-8a3d-0bd0f9f1042f)

```
python projects for practice
```

### Example 3

```bash
python open_source_project.py --image path/to/image.jpg
```
![image](https://github.com/user-attachments/assets/91b8f777-8e4c-43f1-88e3-ceb03d084b50)

**Output:**
![image](https://github.com/user-attachments/assets/2e70f37a-db9f-4142-b271-383371f4436a)

```
Python project ideas
```

### Example 4

```bash
python projects_for_practice.py --image path/to/image.jpg
```
![image](https://github.com/user-attachments/assets/aaec3753-f60c-4594-b028-3682558bfe85)

**Output:**
![image](https://github.com/user-attachments/assets/31f4bf0e-e834-44e4-be71-b97d52c1ea90)

```
interesting python project example
```

### Example 5

```bash
python open_source_project_example.py --image path/to/image.jpg
```
![image](https://github.com/user-attachments/assets/97862757-bf3a-4e97-b253-efb9ac7661fc)



**Output:**
![image](https://github.com/user-attachments/assets/6f299d08-0518-43f5-ab30-d8e04d3172d9)

```
learning python projects
```


## Summary

In this project, we implemented a CNN to detect gender and age from a single face image. Did you complete the project? Try running it on your own pictures and explore more exciting Python projects with source code published by DataFlair.

**Happy Learning! 😊**
```
