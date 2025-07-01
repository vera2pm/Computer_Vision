# Segmentation

Task: 
- Train a model which can count number of rods in photo
- Model: UNET + blob detection for counting
- Metric: MAPE + Visualization of mask

Code: `src/segmentation_task.py` and used model with pytorch lightner classes are in `src/segmentation/`

Loss for training - **Dice loss**

## Training model:
### Without augmentation
Size of the image (224, 224), precision = float32

Training loss:

<img src="results_data/Unet_no_aug_train_val.png" alt="Unet_no_aug_train_val.png" width="400"/>


### With augmentation
For this experiment augmentation was added: A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.GaussianBlur(3).

Moreover, size of the image was changed to (512, 512), but precision was reduced to "16-mixed", that is equal to float16.


Training loss:

<img src="results_data/Unet_aug_train_val.png" alt="Unet_aug_train_val.png" width="400"/>


## Results

Here are model outputs - masks - from the test dataset.

| model               | images                                                                                                                                                                                                                                                                                           |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| original image      | <img src="results_data/Photo10.JPG" alt="Photo10.JPG" width="200"/>, <img src="results_data/Photo40.JPG" alt="Photo40.JPG" width="200"/> , <img src="results_data/Photo122.JPG" alt="Photo122.JPG" width="200"/> , <img src="results_data/Photo167.JPG" alt="Photo167.JPG" width="200"/>                                                             |
| oririgal mask       | <img src="results_data/mask_Photo10.JPG" alt="mask_Photo10.JPG" width="200"/>, <img src="results_data/mask_Photo40.JPG" alt="mask_Photo40.JPG" width="200"/> , <img src="results_data/mask_Photo122.JPG" alt="mask_Photo122.JPG" width="200"/> , <img src="results_data/mask_Photo167.JPG" alt="mask_Photo167.JPG" width="200"/>                     |
| Unet no augmentation | <img src="results_data/test_mask_woaug0.png" alt="test_mask_woaug0.png" width="200"/>, <img src="results_data/test_mask_woaug1.png" alt="test_mask_woaug1.png" width="200"/>, <img src="results_data/test_mask_woaug2.png" alt="test_mask_woaug2.png" width="200"/>, <img src="results_data/test_mask_woaug3.png" alt="test_mask_woaug3.png" width="200"/> |
| Unet w/ augmentation | <img src="results_data/test_mask_aug0.png" alt="test_mask_aug0.png" width="200"/>, <img src="results_data/test_mask_aug1.png" alt="test_mask_aug1.png" width="200"/>, <img src="results_data/test_mask_aug2.png" alt="test_mask_aug2.png" width="200"/>, <img src="results_data/test_mask_aug3.png" alt="test_mask_aug3.png" width="200"/>           |



### Blob detection examples:

<img src="results_data/test_blob_3.jpg" alt="test_blob_3.jpg" width="200"/>
<img src="results_data/test_blob_10.jpg" alt="test_blob_10.jpg" width="200"/>