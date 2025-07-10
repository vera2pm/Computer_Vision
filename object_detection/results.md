# Object Detection

- Take 2 or more objects (which are not presented in COCO/OpenImages)
- Take a photo of them
- Label them using CVAT or VGG VIA
- Train an object detection model

## Dataset

Hyppopotamus and Rhinoceros were chosen as objects.

I prepared the Dataset: found images, labeled them with Roboflow, split for train, validation and test. 

## Model YOLO parameters:
[opt.yaml](opt.yaml)

## Metrics Yolo

### Confusion matrix:
![confusion_matrix.png](confusion_matrix.png)

### Precision-Recall:
![PR_curve.png](PR_curve.png)


## Examples of batches:

### Train:
![train_batch0.jpg](train_batch0.jpg)

### Validation:
![val_batch0_labels.jpg](val_batch0_labels.jpg)