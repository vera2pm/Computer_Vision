# Metric Learning or Similarity Search

Task: Build a system which can find a visually similar product in the catalog by a photo-query

Code: `src/similarity_search.py`

Graphics of training:

![similarity_best.png](similarity_best.png)

## Add online sampling
When I added online sampling, i.e. during training the farthest positive and the closest negative images are chosen to make margin between them bigger to improve loss.


## Metrics comparison
1. Model with offline sampling with pretrained weights
    - 100 epochs - **Top1** = 69%

2. Model with online sampling with pretrained weights
    - 1 epoch - **Top1** = 80% 
    - 124 epoch - **Top1** = 64%

3. Model with online sampling without weights 
   - 1 epoch - **Top1** = 39% 
   - 83 epoch - **Top1** = 28%

