This txt file is going to be used to note down all the information about the minGPT.py file we have here

This goes over everything that we are implementing in this code and look into depth about how all of this has been implemented

It will also go into the things we have done in the original file, marking down all the information about different functions and their usage.


# GPT-2 model
So the GPT-2 model follows very similar things to the GPT-1 and the decoder part of the transformer architecture. 
But one of the big changes that the GPT-2 model does is it uses a pre-normalization technique over post-normalization done in GPT-1. 

This was done because pre-normalzation was found to be a better alternative as it:
- stablized the gradients
- led to faster convergence 
- avoided the exploding gradient problem

So the architecture is very similar to GPT-1 except all the normalizations are now done before the sub-layers and not after. They also put a normalization after the feed-forward NN, before passing the model to the linear head to calculate logits and loss.

They also integrated many dropouts, this was done to improve the model performance as this would help it make it work well on unseen data during training.


## Layer Normalization[class LayerNorm](../LLM/nanoGPT.py#L13)

LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False 
- This function is used to create the layer normalization in nanoGPT
- In this we define the weights and bias initially weight = 1, bias = 0


## Causal/Masked Self- Attention[class CausalSelfAttention](../LLM/nanoGPT.py#L25)

Function for masked self-attention, one of the most important parts of a decoder-only architecture model.

- In this first we define the attention part that will have all the projections of key, query and value vectors for all heads, but in a batch.
- We define regularization terms like attention_dropout and residual_dropout.
- Then we create a forward propogation where we define the query, key and value vectors. 
- Query(q), Key(k), Value(v) vectors all are defined in the same dimension of (B, nh, T, hs), which is basically (Batch_size x num_heads x seq_len x head_size).
- if the flash works, then we can directly call the scaled_dot_prodcut_attention from nn.functional, otherwise we define the attention implementation 
- The attention implementation is the same we have learned in the past, where we defined the attention, masked the future tokens, apply softmax on the attention and then mutlipled that softmax value with the value vector to find the output.
- Lastly we need to convert the (B, nh, T, hs) into (B, T, C). We do this by inter-changing the num_heads and block_size and then making num_heads x head_size into C. 


## Multi-layer Perceptron[class MLP](../LLM/nanoGPT.py#L69)
This is used to allow the model to transform and mix the information in the embeddings after the self-attention step, helping it learn more complex representations

- This is achieved by projecting the embeddings into a higher dimension, applying activation function (GELU in this case) that adds non-linearity, then reverting back to the same shape, and applying dropout


## Defining the block[class Block](../LLM/nanoGPT.py#L85)
This is basically defining the block of the model architecture

- First we apply layer norm 1 on the input to stabilize the graidents of the input before passing it through the rest of the architecture
- Then we add the masked multi-head attention to get a better understanding of the input
**Residual**
- After passing through multi-head attention, we apply layer norm 2 to stabilize the gradients and prevent the inverted covariant shift.
- Then we pass the input through Multi-Layer Perceptron, which transforms the mode and help the model understand more complex representations, by projecting the embeddings into a higer dimension and applying non-linearity to make the model better.
**Residual**

- The residual that are mentioned here are basically residual connections added in the model to improve the gradient flow and keep the model stable due this being a deep model. 
- It also helps massively during the back propogation steps when we need to update the weights, as it creates direct gradient-paths through the network and prevent vanishing gradient problem during back propogation. It also lets the model map identity mappings when needed, improving training stability and speed.

## configuring the GPT model parameters[class GPTConfig](../LLM/nanoGPT.py#L100)
- block_size = 1024
- vocab_size = 50304 
- n_layer = 12
- n_head = 12
- n_embd = 768
- dropout = 0.0
- bias = True


## Defining the GPT class[class GPT](../LLM/nanoGPT.py#L109)
This class is defining the whole flow of the model, here we will define everything

- first we assert that the vocab_size and the block_size is not None
- Then we start with defining the token and position embeddings of shape (vocab_size x n_embd), making essentially a lookup table for every token in the vocabulary and its score according to the dimensionality embeddings
- we will define the heads as a hyperparameter, essentially making it from the range of 1 to n_layer(set to 12)
- then we define the final proejection layer of the model (lm_head), which takes the output of the model and turns into a score of each token taken from the vocabulary.
- then we apply weight typing on it, using the same emebedding matrix from the token embeddings.
- We apply the initialized weights for all layers recursively
- After this we apply a special scaled initialization, as per the GPT-2 paper on the residual projections. This is done to normalize the variance we prevent the model from becoming too slow and making sure the training remains stable

Then we created a function for calculating the number of parameters in the model[get_num_params](../LLM/nanoGPT.py#L138). This function excludes the position embeddings as we only take non-embedding count. We can also exclude the token embeddings, but as it is weight tyed with the weights in the final layer, we include them.

Initializing the weights using the [_init_weights](../LLM/nanoGPT.py#L150). where the weights for Linear and Embedding modules are normalized

Then we define the the forward function[def forward(self, idx, targets=None)](../LLM/nanoGPT.py#L158), where we forward the GPT function itself.
- - Here we define the size of the token and position embeddings and add them to create the input for the model, dropout is applied on this to prevent overfitting and improve generalization.
- - Then we define 2 cases, training(with targets) and inference(without targets). 

### CASE 1: Training(target is not None)
`logits = self.lm_head(x)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)`
- self.lm_head(x) projects the model output x to vocabulary logits.
- .view(-1, logits.size(-1)) flattens the logits and targets to match shape: [batch_size * seq_len, vocab_size] vs. [batch_size * seq_len].
- F.cross_entropy(...) computes the token-wise loss.
- ignore_index=-1 means tokens with value -1 in targets are skipped in loss calculation (e.g., padding).

### CASE 2: Inference(targets is None)
`logits = self.lm_head(x[:. [-1],:])
loss = None`
- Only the last token's representation x[:, [-1], :] is passed to lm_head, saving compute.
- [-1] keeps the time dimension as [batch_size, 1, hidden_size].
- No loss is computed, since we're generating, not training.


Lastly we create a function in the case we want to reduce the block_size from the existing 1024 block_size of GPT-2[def crop_block_size](../LLM/nanoGPT.py#L184).

From this point on the code is a bit tricky, so keep in mind most of the things might not be too easy to do, we will go step by step on each of these to understand what are we doing

Now we define the function from which we can call thr pretrained model[def from_pretrained](../LLM/nanoGPT.py#L197)
- We will then initialize a hugging face/transformer model from the GPT2LMHeadModel, where the model type are {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-x1'}
- Then we configure the number of layers, number of heads and dimensionality embeddings according to the model chosen, keeping the vocab_size, block_size and bias fixed. We can override the dropout rate if needed.
- We will import GPT-2 model pre-trained weights from hugging face.
- We will then configure our model called minGPT, which is initializing the weights of GPT-2 while discarding/ignoring the buffers
- We will transpose some of the layers in hugging face implementation as they use the Conv1D of shape [out, in] and we use Linear [in, out]
- `for k in sd_keys_hf:
    if any(k.endswith(w) for w in transposed):
        # special treatment for the Conv1D weights we need to transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k].t())`
This code snippet does the transfering of weights, while also doing transposing where needed, to match the shape. 

We will then configure the optimizers[def configure_optimizers](../LLM/nanoGPT.py#L254)
- We will first take all the candidate parameters and filter out those that do not require grad
- Then we will create optimization groups, where any parameter that is 2D will be weight decayed, i.e. all weight tensors in matmuls + embeddings decay, other ones like biases and layernorm will not be decayed(Hence the decay_params and nodecay_params)
- Then we will initialize the AdamW optimizer, it is basically the Adam optimizer but better because it seperates the weight decay from gradient upgrades, this allows more control over regularization and prevents overfitting in deep model like LLMs

Then we create a function to estimate the MFU(Model FLOPs Utilization)[def estimate_mfu](../LLM/nanoGPT.py#L280)
- This is used to calculate the how much GPU's theoretical power is the model using.

Then finally, we create the generate function[def gneerate](../LLM/nanoGPT.py#L297)
- This block is simply the running of the model architecture of the GPT-2 model, where we 
- - will pass the input (crop if not of block_size) 
- - get logits from the model for the current input 
- - select next token and and apply desired temperature.
- - Then optionally crop the logits to only the top_k options(done to prevent the model to over-generalize every time and let us have other probable tokens)
- - Then apply softmax on the logits to find the probabilities
- - Sample next token from the probability distribution. 
- - Append the new token into the input sequence and pass it again in the model. This is done for max_new_token steps
- Here we apply torch.no_grad as this is the inferencing step and we do not need to calculate the loss, therefore no back-propogation.








