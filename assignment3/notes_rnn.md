---
DATE:2018/06/23
---


## RNN for image captioning
RNN网络layers
### Vanilla RNN: forward/backward
- Vanilla RNN: step forward
- Vanilla RNN: step backward
### Word embedding: forward/backward
因为word index的缘故,forward和backward的计算有些特别
### Temporal Affine layer   
### temporal_softmax_loss
- 有mask的softmax的loss计算/dx计算

### CaptioningRNN
计算图是什么样子的
#### (1) Use an affine transformation to compute the initial hidden state from the image features. This should produce an array of shape (N, H) 
- h0 = np.dot(features, W_proj) + b_proj
#### (2) Use a word embedding layer to transform the words in captions_in from indices to vectors, giving an array of shape (N, T, W).        
- captions_in shape (N, T)
- W_embed shape is (vocab_size, wordvec_dim)
- x shape (N, T, W)
- x, cache_embedding = word_embedding_forward(captions_in, W_embed)
#### (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to process the sequence of input word vectors and produce hidden state vectors for all timesteps, producing an array of shape (N, T, H).    
**时间步**主要在rnn_forward/rnn_backward中计算
- x shape (N, T, W), h shape (N, T, H)
- h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)

#### (4) Use a (temporal) affine transformation to compute scores over the vocabulary at every timestep using the hidden states, giving an array of shape (N, T, V).
- h shape (N,T,H), scores (N,T,M)
- W_vocab, b_vocab IS Weight and bias for the **hidden-to-vocab transformation**.
- scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)

#### (5) Use (temporal) softmax to compute loss using captions_out, ignoring the points where the output word is <NULL> using the mask above.
- loss, dscores = temporal_softmax_loss(scores, captions_out, mask, verbose=False)     