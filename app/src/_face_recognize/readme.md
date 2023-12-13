# COMPUTER VISION - FACE RECOGNIZE

## Important Topics
```
Here are some of the key topics I will be covering during this study project:

- Image Processing
- Object Detection
- Image Classification
- Feature Extraction
- Convolutional Neural Networks (CNNs)
- OpenCV
- Deep Learning for Computer Vision
```
### Face Recognize
```
1.  Face Detection
2.  Face Recognize
    2.1. Face recognize using LBPH algorithm and OpenCV
    2.2. Face recognition using Dlib, CNN and distance calculate algorithm
```
#### LBPH: Local Binary Pattern Histograms
```
    1.1. The matrix representation:
    M (9 pixels) =
    [12 15 18]
    [05 08 03]
    [08 01 02]

    condition:
    {if [others element] >= [selected element] } -> 1
    {if [others element] < [selected element] } -> 0

    Then,
    getting element M2x2 = [8]
    if 12 >= 8 ? yes, so M1x1 = 1
    if 15 >= 8 ? yes, so M1x2 = 1
    if 18 >= 8 ? yes, so M1x3 = 1
    if 5 >= 8 ? yes, so M2x1 = 0
    if 8 >= 8 ? yes, so M2x2 = 8
    if 3 >= 8 ? yes, so M2x3 = 0
    if 8 >= 8 ? yes, so M3x1 = 1
    if 1 >= 8 ? yes, so M3x2 = 0
    if 2 >= 8 ? yes, so M4x3 = 0

    Result:
    -> -> ->
    [1 1 1] |
    [0 8 0] |
    [1 0 0] |
    <- <- <-

    binary number: 11100010 (226 decimal)

    1.2. Good to night images
    2.3. then, the image is partitioned in blocks, then, calculate each histogram, each
    block have your own histogram. Each person have different histograms and each face
    characteristics (eyes, mouth...) have different histograms
    2.4. Comparing each histogram with the real person and you can get the result
```
