# Metric Learning or Similarity Search

Task: Build a system which can find a visually similar product in the catalog by a photo-query

Code: `src/similarity_search.py`

## Baseline

Metric on test data -- **Top1** = 69%

Graphics of training:

![similarity_best.png](similarity_best.png)

## Add online sampling
When I added online sampling, i.e. during training the farthest positive and the closest negative images are chosen to make margin between them bigger to improve loss.

Metric on test data -- **Top1** = 69%

Graphics of training:
