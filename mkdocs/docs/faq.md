## Do you support \[feature | encoder | decoder\] in Ludwig?

The list of encoders for each feature type is specified in the [User Guide](user_guide.md).
We plan to add additional feature types and additional encoders and decoders for all feature types. For instance, we plan to add speech / audio features, the [Transformer](https://arxiv.org/abs/1706.03762), [ELMo](https://arxiv.org/abs/1802.05365) and [BERT](https://arxiv.org/abs/1810.04805) for text features, and [DenseNet](https://arxiv.org/abs/1608.06993) and [FractalNet](https://arxiv.org/abs/1605.07648) for images.
If you want to help us implementing your favourite feature or model please take a look at the [Developer Guide](developer_guide.md) to see how to contribute.


## Do all datasets need to be loaded in memory?

At the moment it depends on the type of feature: image features can be dynamically loaded from disk from an opened hdf5 file, while other types of features (that usually take need less memory than image ones) are loaded entirely in memory for speed. We plan to add an option to load also other features from disk in future releases and to also support more input file types and more scalable solutions like [Petastor](https://github.com/uber/petastorm).


## Who are the authors of Ludwig?

- Piero Molino is the main designer and maintainer
- Yaroslav Dudin is a key contributor
- Sai Sumanth Miryala contributed all the testing and polishing.


## Who else helped developing Ludwig?

- Yi Shi who implemented the time series encoding
- Ankit Jain who implemented the bag feature encoding
- Pranav Subramani who contributed documentation
- Alex Sergeev and Felipe Petroski Such who helped with distributed training
