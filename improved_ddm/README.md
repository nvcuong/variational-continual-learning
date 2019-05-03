# Improving and Understanding Variational Continual Learning

Link to paper (on arXiv): [Improving and Understanding Variational Continual Learning]()

Requirement: Tensorflow 1.12.0.

**To run the Permuted MNIST experiment:**

	python run_permuted.py

**To run the Split MNIST experiment:**

	python run_multihead_split.py

**To plot weights from trained models on both experiments:**

	python plot_weights.py

The printed results are matrices where each row i contains the test set accuracies on all previously observed tasks after seeing task i. The average accuracy will also be plotted and saved to [model_storage/](model_storage/). They should be similar to the figures in Appendix A in the paper.
