## Quora Question Pairs Can you identify question pairs that have the same intent?


@ImportantPoints
- Finishing the assignment (30%)
- Neural network architecture selection and implementation (30%)
- Code quality (20%)
- Data preprocessing/augmentation (10%)
- Baseline accuracy of 80% (10%)


## Links
https://gist.github.com/prats226/4ba1856a91664671dd7ef9bf9e821ff9

## Resources
- https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/deepnet.py
- https://github.com/bradleypallen/keras-quora-question-pairs/blob/master/keras-quora-question-pairs.py
- https://github.com/facebookresearch/poincare-embeddings
- https://github.com/facebookresearch/fastText
- https://pdfs.semanticscholar.org/b31e/447edb0af6ab5ddd4fc0ce3d4a8c6c70882e.pdf?_ga=2.53550959.631894776.1517046815-1803697881.1517046815
- https://github.com/bradleypallen/keras-quora-question-pairs


## Work Process
- I have basically used Embedding layers (word embeddings),  LSTM's (As these are good in holding memory for long time) and in some models BiDirectional LSTM with Attention mechanicism (http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/).
- GloVe embeddings (https://nlp.stanford.edu/projects/glove/) were used to initialize the embedding weights. Words which are not present in glove were represented with zeros. The embedding layers were mostly freezed during training. Training them is leading to over-fitting on dataset.
- Used BatchNormalization between dense layers and drop opt to generalize the network and avoid over fitting and speed up the training progress
- Relu activation functions are used throughout the network. The output layer uses Sigmoid.
- Binary cross entropy is used to evaluate the loss and adam optimizer to compute the gradients.
- Basic feature processing is done after reading the code and comments mentioned in this kaggle kernals. (https://www.kaggle.com/currie32/the-importance-of-cleaning-text) (This improved the model performance)


## Further Work
- Bilateral Multi-Perspective Matching for Natural language Sentences. (https://arxiv.org/pdf/1702.03814.pdf)
- This paper achieves an accuracy of 87% and uses some intitutive architecture
  - Given two sentences P and Q, model first encodes them wiha BiLSTM encoder
  - Next, we match tow encoded senctences in two directions P against Q and Q against P.
  - In each matching direction, each time step of one setence is matched against all time-steps of the other sentence from multiple perspectives.
  - Another BiLSTM layer is utilized to aggregate the matching results into a fixed-length matching vector.
  - Based on the matching vector a decision is made through a fully connected layer.
- keras Implementation: https://github.com/ijinmao/BiMPM_keras


## Packages used
- tensorflow==1.3.0
- keras==2.1.2
- pandas==0.20.3
- numpy==1.13.3
- tqdm
- nltk==3.2.4
- re==2.2.1

## Results and code
- train_test_split.ipynb (splits the dataset into train(90%) and test(10%))
- Basemodel3 - 80% accuracy
- Data_processing_base_model6 - 81% accuracy
- Bidirectional_lstm_with_attention-base_model5 - [Training in Progress]
