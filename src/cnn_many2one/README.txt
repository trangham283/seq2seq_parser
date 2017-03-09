Changes from both2parse:
- change of buckets and their sizes
- change of naming convention for model saving directory
- add convolutional attention in attention_decoder
- add math-appropriate attention mechanism in decoder
- reverse speech frames in each word in get_batch step; though this shouldn't matter since conv filters are symmetric, and I'm pooling max over word anyway; modified mainly for "principled"-ness
