
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam
import math

import heapq
class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        T,d=X.shape[1],X.shape[2]
        pos=torch.arange(T)
        result=torch.zeros_like(X)
        for pos in range (T):
            for i in range (d//2):
                result[:, pos, 2*i]=np.sin(pos/(10000**(2*i/d)))
                result[:, pos, 2*i+1]=np.cos(pos/(10000**((2*i)/d)))
        sum=X+result
        return sum


class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """
        dk=self.out_dim
        outputQ=self.linear_Q(query_X)
        outputK=self.linear_K(key_X)
        outputV=self.linear_V(value_X)
        attention_weights=torch.bmm(outputQ, outputK.transpose(1,2))
        if mask!=None:
            transformed_mask=(mask-1).abs().bool()
            attention_weights=attention_weights.masked_fill(transformed_mask, float(-1e32))
        attention_weights=self.softmax(attention_weights/np.sqrt(dk))
        attention_output=attention_weights@outputV
        return (attention_output,attention_weights)

class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return self.linear(outputs), attention_weights
        
class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """  
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)

class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """

        padding_locations = torch.where(X == self.vocab_size, torch.zeros_like(X, dtype=torch.float64),
                                        torch.ones_like(X, dtype=torch.float64))
        padding_mask = torch.einsum("bi,bj->bij", (padding_locations, padding_locations))

        X = self.embedding_layer(X)
        X = self.position_encoding(X)
        for block in self.blocks:
            X = block(X, padding_mask)
        return X, padding_locations

class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """  
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)
        
        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)

        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights

class Decoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()
        
        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1)

        self.vocab_size = vocab_size

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        mask=torch.ones(seq_length, seq_length)
        mask=torch.tril(mask)
        return mask


    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """
        
        # Lookahead mask
        seq_length = target.shape[1]
        mask = self._lookahead_mask(seq_length)

        # Padding masks
        target_padding = torch.where(target == self.vocab_size, torch.zeros_like(target, dtype=torch.float64), 
                                     torch.ones_like(target, dtype=torch.float64))
        target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, target_padding))
        mask1 = torch.multiply(mask, target_padding_mask)

        source_target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, source_padding))

        target = self.embedding_layer(target)
        target = self.position_encoding(target)

        att_weights = None
        for block in self.blocks:
            target, att = block(encoded_source, target, mask1, source_target_padding_mask)
            if att_weights is None:
                att_weights = att

        y = self.linear(target)
        return y, att_weights


class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)
        self.softmax = Softmax(-1)


    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)

    def predict(self, source: List[int], beam_size=1, max_length=12) -> List[int]:
        """
        Given a sentence in the source language, you should output a sentence in the target
        language of length at most `max_length` that you generate using a beam search with
        the given `beam_size`.

        Note that the start of sentence token is 0 and the end of sentence token is 1.

        Return the final top beam (decided using average log-likelihood) and its average
        log-likelihood.

        Hint: The follow functions may be useful:
            - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
            - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
        """
        self.eval() # Set the PyTorch Module to inference mode (this affects things like dropout)
        if not isinstance(source, torch.Tensor):
            source_input = torch.tensor(source).view(1, -1)
        else:
            source_input = source.view(1, -1)

        encoded_source, source_padding=self.encoder(source_input)
        beams = [([0], 0.0)] #sequence containing start sequence, log likelyhood
        for i in range (max_length-1):
            candidates=[]
            candidate_prob=[]
            for (seq, prob) in beams:
                if seq[-1]==1:
                    candidates.append((seq, prob))
                    candidate_prob.append(prob/len(seq))
                    continue
                pred, att_weights=self.decoder(encoded_source, source_padding,  torch.tensor(seq).view(1, -1))
                last_token=pred[:, -1, :]
                logit=torch.log(torch.softmax(last_token, dim=-1))
                #take top k first
                topPred=torch.topk(logit, beam_size)
                for (val, idx) in zip(topPred.values[0], topPred.indices[0]):
                    newSeq=seq+[idx.item()]
                    newProb=prob+val.item()
                    candidates.append((newSeq,newProb))
                    candidate_prob.append(newProb/len(newSeq))
                    
            probs=torch.tensor(candidate_prob)
            _,topk=torch.topk(probs, beam_size)
            beams=[candidates[i] for i in topk]

        bestSeq, bestProb=max(beams,key=lambda x: x[1]/len(x[0]))
        return bestSeq, bestProb/len(bestSeq)

def flip_elements(input_list):
    return [elem[::-1] for elem in input_list]

def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('./data/translation_data.bin', 'rb') as f:
        corpus,french_vocab, eng_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:10000]
        train_sentences = flip_elements(train_sentences)
        test_sentences = flip_elements(test_sentences)
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(french_vocab), len(eng_vocab), len(train_sentences),
                                              len(test_sentences)))
        

        return train_sentences, test_sentences, eng_vocab, french_vocab

def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):
    
    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        if len(source) >= max_length:
            continue
        if len(target) >= max_length:
            continue
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)

def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)

def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray,
                        i, j):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 1:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 1:
        target_length += 1

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    name = f"attention_{i}_{j}.png"
    plt.savefig(f'attention_plot_{i}_{j}.png')  # Save the plot with a unique filename

    # Clear the plot so future plots do not overlap
    plt.close(fig)

    

def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")
    return epoch_train_loss, epoch_test_loss




def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    *** For students in 10-617 only ***
    (Students in 10-417, you can leave `raise NotImplementedError()`)

    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (0), EOS (1), and padding (anything after EOS)
    from the predicted and target sentences.
    
    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """

    def stripSeq(seq):
        res=[]
        i=0
        while seq[i]!=1:
            if seq[i]==0:
                i=i+1
                continue
            else:
                res.append(seq[i])
                i=i+1
            
        return res
    predStrip=stripSeq(predicted)
    targetStrip=stripSeq(target)

    if len(predStrip)<N or len(targetStrip)<N:
        return 0

    def computePk(predicted, target, k):
        count=0
        def getDict(seq, k):
            res=dict()
            for i in range(len(seq)-k+1):
                currGram=tuple(seq[i: i+k])
                if currGram not in res:
                    res[currGram]=1
                else:
                    res[currGram]+=1
            return res
    
        predDict=getDict(predicted, k)
        targetDict=getDict(target, k)
        for key in predDict:
            if key not in targetDict:
                continue
            else:
                count+=min(predDict[key], targetDict[key])

        
        num_ngram=len(predicted)-k+1

        res=count/num_ngram
        return res
    res=1
    for i in range (1, N+1, 1):
        currScore=computePk(predStrip, targetStrip, i)**(1/N)
        res=res*currScore
    brevity_penalty=min(1, np.exp(1-len(targetStrip)/len(predStrip)))
    return res*brevity_penalty
        

if __name__ == "__main__":
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()
    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), 12)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), 12)

    embed_dim=256
    target_vocab_size=len(target_vocab)
    source_vocab_size=len(source_vocab)
    epochs = list(range(1, 31))

    #model 2a
    model2a=Transformer(source_vocab_size, target_vocab_size, embed_dim, 1, 1, 1)
    train_loss2a, test_loss2a= train(model2a, train_source, train_target, test_source, test_target, target_vocab_size)
    torch.save(model2a.state_dict(), "model2a.pkl")
    print("model2a result:\n")
    print(train_loss2a)
    print(test_loss2a)
    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss2a, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss2a, label='Test Loss', marker='o')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over 30 Epochs 2a')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig("train_test_loss_plot2a.png")  # Save as PNG file

    model2b=Transformer(source_vocab_size, target_vocab_size, embed_dim, 1, 1, 2)
    train_loss2b, test_loss2b= train(model2b, train_source, train_target, test_source, test_target, target_vocab_size)
    torch.save(model2b.state_dict(), "model2b.pkl")
    print("model2b result:\n")
    print(train_loss2b)
    print(test_loss2b)
    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss2b, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss2b, label='Test Loss', marker='o')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over 30 Epochs 2b')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig("train_test_loss_plot2b.png")  # Save as PNG file
    
    model2c=Transformer(source_vocab_size, target_vocab_size, embed_dim, 2, 1, 1)
    train_loss2c, test_loss2c= train(model2c, train_source, train_target, test_source, test_target, target_vocab_size)
    torch.save(model2c.state_dict(), "model2c.pkl")
    print("model2c result:\n")
    print(train_loss2c)
    print(test_loss2c)
    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss2c, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss2c, label='Test Loss', marker='o')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over 30 Epochs 2c')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig("train_test_loss_plot2c.png")  # Save as PNG file
    plt.show()


    model2d=Transformer(source_vocab_size, target_vocab_size, embed_dim, 2, 2, 2)
    train_loss2d, test_loss2d= train(model2d, train_source, train_target, test_source, test_target, target_vocab_size)
    torch.save(model2d.state_dict(), "model2d.pkl")
    print("model2d result:\n")
    print(train_loss2d)
    print(test_loss2d)
    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss2d, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss2d, label='Test Loss', marker='o')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over 30 Epochs 2d')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig("train_test_loss_plot2d.png")  # Save as PNG file
    plt.show()

    model2e=Transformer(source_vocab_size, target_vocab_size, embed_dim, 2, 3, 4)
    train_loss2e, test_loss2e= train(model2e, train_source, train_target, test_source, test_target, target_vocab_size)
    torch.save(model2e.state_dict(), "model2e.pkl")
    print("model2e result:\n")
    print(train_loss2e)
    print(test_loss2e)
    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss2e, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss2e, label='Test Loss', marker='o')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over 30 Epochs 2e')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig("train_test_loss_plot2e.png")  # Save as PNG file
    plt.show()


    def q3Sol():
        model=Transformer(source_vocab_size, target_vocab_size, embed_dim, 2, 3, 4)
        model.load_state_dict(torch.load("model2e.pth")) 
        
        for i in range(6):
            target_seq=target_seq[i]
            source_seq=test_source[i]
            prediction, avg_Log=model.predict(source_seq, beam_size=3, max_length=12)
            sourceSentence=decode_sentence(source_seq, source_vocab)
            targetSentence=decode_sentence(target_seq, target_vocab)
            predictedSentence=decode_sentence(prediction, target_vocab)
            with open("q3output.txt", "w") as file:
                file.write("Source Sentence: " + sourceSentence + "\n")
                file.write("Target Sentence: " + targetSentence + "\n")
                file.write("Predicted Sentence: " + predictedSentence + "\n")
                file.write("Average Log-Likelihood: " + str(avg_Log) + "\n")
                file.write("\n")
            
    q3Sol()

    def q4Sol():
        for i in range(3):
            currTrainSource=train_source[i]
            currTrainTarget=train_target[i]
            model=Transformer(source_vocab_size, target_vocab_size, embed_dim, 2, 3, 4)
            model.load_state_dict(torch.load("model2e.pth")) 
            _, attentionMatrix=model(currTrainSource, currTrainTarget)
            print(attentionMatrix)
            visualize_attention(currTrainSource, currTrainTarget, target_vocab, attentionMatrix, )

    # def q5Sol():

    # Q3(test_source, test_target, source_vocab, target_vocab)