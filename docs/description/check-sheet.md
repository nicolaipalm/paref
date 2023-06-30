## Check sheet for proper use of Surrogate based optimization

[] Are the components of input and output within the same scale? Action: Normalization of input and output data
[] Do you have enough training data (Rule of thumb: 3 dots per input dimension; Ex: 10 input dimensions and 30 samples)? Action: Extend initial (LH) sampling
[] Was the training successful? Action: inspect trainingsprocess if error converged; if not raise the number of training iteration (Rule of thumb: at least 2000 training iterations)


also for
- impl bbf (store evals)
- do dim match? **Action:**
