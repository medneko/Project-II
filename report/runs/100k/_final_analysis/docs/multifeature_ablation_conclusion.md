# Multi-feature Ablation Conclusion

The best full-100k experimental model is `text_pca64_lexical + minibatch_k32`. Its global silhouette on the diagnostic sample is 0.109388; the bounded summary reports silhouette 0.109388 and DBI 3.381518.

Lexical-only augmentation is the cleanest positive signal. Publisher/stock-heavy features are useful for context but can dominate or blur semantic clustering, especially for GMM and HDBSCAN.

CLARA true was tested and remains below the best MiniBatch result.
