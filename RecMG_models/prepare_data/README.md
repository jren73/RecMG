1. Preprocessing Data
   We use dataset-16 to reproduce model performance. We set cache size as 20% of total trace length,  we use 200000000 indices from dataset-16, which is about 10% of the entire dataset
   It requires about 4H to finish data preprocessing
   $ python optgen.py 0.2 200000000 /path/to/fbgemm_t856_bs65536_15.pt 

2. Train ML model for caching
    We set input sequence length as 150 and output sequence length as 10. The total number of training steps is 12K
    It requires about 4H to train a model
    $ python3 seq2seq_caching.py /path/to/fbgemm_t856_bs65536_15.pt 150 10

3. Train ML model for prefetching
    $ python3 train_prediction.py --config example_seq2seq.json --traceFile /path/to/fbgemm_t856_bs65536_15.pt 


