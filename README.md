# GridRep

GridRep is a feature transformation tool that enables more efficient BSCAN clustering. 

It is effective for large volumes of low-cardinality input data containing multiple repeating unique sets of feature values.

## preprocess.FeaturesTransformer

The GridRep transformer generates a representative input subset based on DBSCAN's _min_samples_ parameter that participates in the clustering procedure. The generated labels can then be re-mapped back to the original input data.

If the input data does not appear to be of low-cardinality, the GridRep transformer also allows to eliminate suspected false precision (e.g. lots of meaningless decimals after standardisation) by simply ppassing a _rounding_decimals_ parameter value.

<img src="subsampling.png" width=400>
<!-- ![Representatives](files/subsampling.png) -->

## cluster.ClippedDBSCAN

ClippedDBSCAN wraps the FeaturesTransformer around sklearn's DBSCAN, in a sklearn.pipeline compatible Estimator.


```python

```
