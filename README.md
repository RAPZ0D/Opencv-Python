# Opencv-Python
Using the OpenCV Library using Python 
# OpenCV using Python
- [Installation Instructions](#installation-instructions)
- [Libraries Used](#libraries-used)
- [Visual](#visual)
- [Pytorch YOLOV5](#pytorch-yolov5)

This project utilizes OpenCV, NumPy, and Matplotlib for image processing and computer vision tasks in Python.

## Installation Instructions

### Installing OpenCV, NumPy, and Matplotlib with pip

1. **OpenCV**: Install OpenCV using pip, preferably within a virtual environment:
   ```bash
   pip install opencv-python

   pip install numpy

   pip install matplotlib

### Installation within a Virtual Environment. Create a Virtual Environment (Optional but recommended):
### Create a virtual environment using venv or virtualenv

1. **Using venv (Python 3.x)**
``bash
  python3 -m venv myenv

2.  **Activate the virtual environment (Linux or MacOS/Windows)**

  (MacOS / Linux)
  
  source myenv/bin/activate
  
  (Windows)
  
  myenv\Scripts\activate

## Libraries Used 
1. OpenCV
2. Numpy
3. Matplotlib
# OpenCV using Python

This project utilizes OpenCV, NumPy, and Matplotlib for image processing and computer vision tasks in Python.

## Libraries Overview

### OpenCV (Open Source Computer Vision Library)
- **Purpose**: OpenCV is an open-source computer vision and machine learning software library, primarily used for real-time image and video processing. 
- **Functionalities**: It offers a wide range of tools and algorithms for tasks such as object detection, facial recognition, image segmentation, and more.
- **Key Features**: OpenCV provides a robust set of functions to manipulate images and perform various computer vision operations, making it a popular choice in the field.

### NumPy (Numerical Python)
- **Purpose**: NumPy is a fundamental package for scientific computing in Python, particularly for numerical operations and handling multidimensional arrays.
- **Functionalities**: It provides high-performance multidimensional array objects and tools for working with these arrays, enabling mathematical operations on arrays efficiently.
- **Key Features**: NumPy facilitates numerical computations, including linear algebra, Fourier analysis, random number generation, and more, serving as a foundation for many scientific computing tasks in Python.

### Matplotlib
- **Purpose**: Matplotlib is a plotting library for Python and its numerical mathematics extension, NumPy. 
- **Functionalities**: It enables the creation of static, interactive, and animated visualizations in Python, facilitating the generation of various types of plots, charts, histograms, etc.
- **Key Features**: Matplotlib provides a wide range of plotting functions to visualize data, making it useful for data exploration, analysis, and presentation purposes in scientific computing and data science.

These libraries are widely used in various domains of data science, machine learning, computer vision, and scientific computing due to their rich functionalities and capabilities. They form the backbone of many Python-based projects involving numerical computations, data manipulation, and visualization.


## Visual

![opencv](https://github.com/RAPZ0D/Opencv-Python/assets/100001521/8789c454-57c2-4cfd-9f5d-b0c3db35d286)

![o](https://github.com/RAPZ0D/Opencv-Python/assets/100001521/d42555bd-a34d-407a-bdab-da3c03209967)


## Pytorch YOLOV5
YOLOv5 is an object detection model series known for its efficiency in real-time object detection tasks. It's built on the You Only Look Once (YOLO) architecture and implemented using the PyTorch deep learning framework.

 **Key Features**

- **Unified Model**: Predicts bounding boxes and class probabilities in a single pass through the network.
- **Grid-Based Approach**: Divides the image into a grid and predicts objects within each grid cell.
- **Anchor Boxes**: Utilizes predefined anchor boxes for accurate bounding box predictions.
- **Efficiency**: Offers real-time inference speeds for object detection tasks.

**YOLOv5 Overview**

Features of YOLOv5

- **PyTorch Implementation**: Built using the PyTorch framework for flexibility and ease of use.
- **Improved Architecture**: Enhanced architecture for better accuracy and speed.
- **Variants**: Different model sizes (s, m, l, x) catering to varying computational requirements.
- **Pretrained Models**: Provides pretrained models on COCO dataset for general object detection.
- **Transfer Learning**: Allows fine-tuning on custom datasets for specific object detection tasks.

### Object Detection with YOLOv5

- **Usage**: Detects multiple objects within an image, providing bounding boxes and class probabilities.
- **Applications**: Used in various fields including autonomous vehicles, surveillance, and robotics.

### Getting Started 
[Pytorch Link](https://pytorch.org/hub/ultralytics_yolov5/)
## Installation

To use the YOLOv5 model and related utilities, you need to install the Ultralytics library. Run the following command in your terminal or command prompt:

## Installation

1. **To use the YOLOv5 model and related utilities, you need to install the Ultralytics library. Run the following command in your terminal or command prompt:** 

   ```bash 
    pip install -U ultralytics


## Using YOLOv5 for Object Detection

1. **To perform object detection using YOLOv5, you can utilize the following code:**
   ```python
   import torch

   # Load the YOLOv5 model
   model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

   # Define a batch of images (e.g., image URLs)
   imgs = ['https://ultralytics.com/images/zidane.jpg']

   # Obtain object detection results
   results = model(imgs)

   # Display results in console
   results.print()

   # Save or display the results
   results.save()  # Save results to a directory
   # or
   results.show()  # Display results

## VISUALS/ OUTPUTS

![u](https://github.com/RAPZ0D/Opencv-Python/assets/100001521/c6cdda09-6dbb-43eb-84cc-7db0be86f3d9)


![j](https://github.com/RAPZ0D/Opencv-Python/assets/100001521/bda5ca7a-7bc3-43c4-853a-c1d71f0cebaa)
