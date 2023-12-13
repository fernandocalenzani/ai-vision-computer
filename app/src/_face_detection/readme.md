# COMPUTER VISION - FACE DETECTION

## Important Topics

Here are some of the key topics I will be covering during this study project:
```
- Image Processing
- Object Detection
- Image Classification
- Feature Extraction
- Convolutional Neural Networks (CNNs)
- OpenCV
- Deep Learning for Computer Vision
```
### Face Detection
```
1. Haarcascade and OpenCV
2. Eyes, Face, People and Clock
3. HOG and Dlib
4. Face detection using CNN using Dlib
5. Face detection using webcam
```
#### 1 IMAGE AND PIXEL
```
1. Pixel: least information available -> l x c = total pixels | 32 x 32 = 1024 pixels
2. RGB: (red, green, blue) -> (0-255,0-255,0-255) -> total = 3 x 1024 = 3072 values
3. RGB to Gray -> the values R,G and B are always the same values = [(10,10,10), (100,100,100)] or 1024 values saved in memory
```
#### 2 HAARCASCADE
```
- Two database: positive images and negative images
- AdaBoost: Training
- Feature Selection
- sum(white pixels - black pixels) each element
- resulting matrix
  [ 2 | 3 | 7 ]
  [ 1 | 5 | 1 ]
  [ 0 | 3 | 6 ]
- Then, send the image to many specific location classifiers such as: eyes, eyebrows, nose. Each one will be a classifier: IMG -> C1 -> C2 -> ... -> Cn
- in the end, the detection is done, if all the classifiers detected the characteristics. Then, the face is detected correctly.
```
#### 2 HOG - Histograms of Oriented Gradients
```
- Derivation: calculate how change the colors in image: zero derivate, small derivate, high derivate
- High derivate: there are the big variations between the environment (different object). Different colors
- Zero derivate: there are not variations between the envoironment (the same object). The same colors
- small derivate: there are a small variations between the environment. Small differences between colors

- Gradient Vector: the direction that the values exchange
- What direction the values are increasing or decreasing

- Gradient Magnitude
- Gradient Direction

- Using Gradient Matrix, we can build a histogram. The histogram show how many times the range value appears in the matrix
```
