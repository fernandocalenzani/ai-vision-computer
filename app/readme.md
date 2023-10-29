# COMPUTER VISION - FACE DETECTION

## Author

- [Fernando Calenzani](fernando@arisetechnology.com.br)

## 0. Requisitos:

- Linux Ubuntu
- Python 3 (`sudo apt-get install python3`)
- jq (`sudo apt-get install jq`)

### 1. Scripts:

Os comandos para instalar, fazer a build ou iniciar algum script são:

```bash
1. sh run.sh --build
2. sh run.sh --main
3. sh run.sh --install py_package
4. sh run.sh --uninstall py_package
```

obs: Cadastre o script em project.json e inclua o caminho em que está a pasta .py

## License

This project is licensed under the MIT License - see the [MIT](LICENSE) file for details.

## Overview

This repository is dedicated to my studies in computer vision, a fascinating field that focuses on teaching machines to interpret and understand visual data. In this project, I'll be exploring various important topics related to computer vision.

## Important Topics

Here are some of the key topics I will be covering during this study project:

- Image Processing
- Object Detection
- Image Classification
- Feature Extraction
- Convolutional Neural Networks (CNNs)
- OpenCV
- Deep Learning for Computer Vision

### Face Detection

1. Haarcascade and OpenCV
2. Eyes, Face, People and Clock
3. HOG and Dlib
4. Face detection using CNN using Dlib
5. Face detection using webcam

#### 1 IMAGE AND PIXEL

1. Pixel: least information available -> l x c = total pixels | 32 x 32 = 1024 pixels
2. RGB: (red, green, blue) -> (0-255,0-255,0-255) -> total = 3 x 1024 = 3072 values
3. RGB to Gray -> the values R,G and B are always the same values = [(10,10,10), (100,100,100)] or 1024 values saved in memory

#### 2 HAARCASCADE

- Two database: positive images and negative images
- AdaBoost: Training
- Feature Selection
- sum(white pixels - black pixels) each element
- resulting matrix  [ 2 | 3 | 7 ]
                    [ 1 | 5 | 1 ]
                    [ 0 | 3 | 6 ]
- Then, send the image to many specific location classifiers such as: eyes, eyebrows, nose. Each one will be a classifier: IMG -> C1 -> C2 -> ... -> Cn
- in the end, the detection is done, if all the classifiers detected the characteristics. Then, the face is detected correctly.
