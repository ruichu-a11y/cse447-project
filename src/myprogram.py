#!/usr/bin/env python
import os
import string
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# This is a mix of things we remember from CSE 446 as well as
# some stuff from A1

# First, we'll need a separate class to help take text and convert it into
# ngrams that we can actually use
class NgramDataset(Dataset):
    def __init__(self, data, n, char_to_idx):
        self.n = n
        self.char_to_idx = char_to_idx
        self.examples = []

        # From here, we'll go through the data and
        # convert it into ngrams (i.e. if we see "the", we'd have "th" -> "e" if n=2)
        for line in data:
            for i in range(len(line) - n):
                # We have n characters of context
                context = line[i:i+n]
                # Then one target character right after
                target = line[i+n]

                # Now we need to get the indices for these characters
                context_idx = [char_to_idx.get(c, char_to_idx['<unk>']) for c in context]
                target_idx = char_to_idx.get(target, char_to_idx['<unk>'])
                self.examples.append((context_idx, target_idx))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # We'll return torch tensors
        context, target = self.examples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# Now we'll make a basic NN model that will take in n characters of context
# and try to predict the next character
class NgramModel(nn.Module):
    """ Basic n-gram model, vocab_size is number of unique chars, embedding_dim is a hyperparam
        for how large embedding vectors should be, hidden_dim is the number of nodes in the hidden layer,
        and n is the number of characters of context we are using (e.g. n=2 for bigrams) """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n):
        super(NgramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Each vector going into this layer will be the flattened concatenation of the n embedding vectors
        self.hdn_layer = nn.Linear(embedding_dim * n, hidden_dim)

        self.relu = nn.ReLU()  # we want to add nonlinearity to ensure we can capture complex patterns

        # We will apply a softmax to the output of this layer to get probabilities
        self.out_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # This one's simple, just pass through the layers!
        # x will be of shape (batch_size, n)
        embeds = self.embeddings(x)  # (batch_size, n, embedding_dim)

        # Concatenating the embedding vectors!
        embeds = embeds.view(embeds.size(0), -1)  # (batch_size, n * embedding_dim)

        # Pass through the hidden layer and apply ReLU
        out = torch.relu(self.hdn_layer(embeds))  # (batch_size, hidden_dim)

        # Now the last linear layer!
        out = self.out_layer(out)  # (batch_size, vocab_size)
        return out

# As a note for any human graders, we haven't started on the implementation yet,
# but this code currently runs without errors, so we are leaving it as is for this
# submission, as we will begin work this week.
class MyModel:
    """
    Might make this comment more descriptive later, but for now, this model uses a
    basic n-gram NN to predict the top 3 most likely next characters with n characters of context.
    """
    def __init__(self, n=2):
        self.n = n
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
    
    # We'll need a function to build our vocab just like in A1
    def build_vocab(self, data):
        # We want to iterate through string in the data and dump them into a set for unique characters
        chars = set()
        for line in data:
            chars.update(line)  # apparently the update method will iterate through the string for us!
        
        # Now we'll sort before we make the dicts
        chars = sorted(list(chars))

        # Last we fill in char_to_idx and idx_to_char
        # We'll use 0 for unknown chars
        self.char_to_idx = {'<unk>': 0}
        self.idx_to_char = {0: '<unk>'}
        for i, c in enumerate(chars):
            self.char_to_idx[c] = i + 1
            self.idx_to_char[i + 1] = c
        
        # All done! We'll return the vocab size for convenience
        return len(self.char_to_idx)

    @classmethod
    def load_training_data(cls, fname):
        # Honestly if loading data fails it's fine if we crash, so
        # we won't worry about error handling and just hope that everything actually works
        data = []
        with open(fname) as f:
            for line in f:
                data.append(line.strip())
        return data

    @classmethod
    def load_test_data(cls, fname):
        # Same as trainig data
        data = []
        with open(fname) as f:
            for line in f:
                data.append(line.strip())
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir, verbose=False):
        # First we need to build our vocab
        vocab_size = self.build_vocab(data)

        # Now we can make our model!
        # We're using arbitrary magic numbers for now, but we'll do a hyperparam search later
        self.model = NgramModel(vocab_size, 16, 128, self.n)

        # Now we need to make our dataset and dataloader
        dataset = NgramDataset(data, self.n, self.char_to_idx)
        # Another magic number for batch size that we'll tune later
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # Note for later: we'll figure out moving the model to GPU at some point,
        # but for now CPU is fine to make sure we have something working
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # We'll use CE loss since we are doing multiclass classification,
        # and Adam because they wouldn't let me use it in 446 (and also it's a fine default)
        ce_loss= nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # more magic!

        # Now for the main training loop! We'll cast our last spell and use 50 as the epoch number for now
        for epoch in range(50):
            epoch_loss = 0.0
            self.model.train()  # Still not super familiar with torch, but it cant hurt to have the model in train mode for training
            # We'll go through each batch and do our typical ML routine
            for batch in dataloader:
                X, Y = batch  # context is (batch_size, n) and target is (batch_size)

                # I think we may have to move these to GPU?
                # X = X.to(device)
                # Y = Y.to(device)

                # Zero the gradients from the last step
                optimizer.zero_grad()

                # Forward pass to get predictions
                output = self.model(X)  # output is (batch_size, vocab_size)

                # Compute loss
                loss = ce_loss(output, Y)
                # Backward pass to compute gradients
                loss.backward()

                # Let adam update weights
                optimizer.step()

                epoch_loss += loss.item()
            
            # We'll print the loss for this epoch to see how we're doing
            if verbose:
                epoch_loss /= len(dataloader)  # average loss per batch
                print(f"Loss for epoch {epoch + 1}: {epoch_loss:.4f}")


    def run_pred(self, data):
        preds = []
        self.model.eval()  # Same as train, but for... well, evaluation
        default_idx = self.char_to_idx.get('<unk>')  # for readability later

        # This loop structure should work fine
        for inp in data:
            if inp is None:
                inp = ""
            # First, we want to grab the last n characters of the input for our context
            context = inp[-self.n:]
            # Now let's make sure we have enough characters and pad with <unk> if we don't
            if len(context) < self.n:
                context = '<unk>' * (self.n - len(context)) + context  # Python string math is always so weird
            
            # Now we need to convert the context to indices
            context_indices = [self.char_to_idx.get(c, default_idx) for c in context]
            # And convert them to a tensor
            context_tensor = torch.tensor([context_indices], dtype=torch.long)  # (1, n)

            with torch.no_grad():  # we don't need gradients for prediction
                output = self.model(context_tensor)
                # Softmax will convert the output to actual probabilities
                probs = torch.softmax(output, dim=1)

                # Now we want to get the top 3 most likely next characters
                _, indices = torch.topk(probs, k=3)

                # Then we just convert the indices to the characters and concatenate them into a string
                indices = indices.squeeze(0).tolist()
                pred_chars = [self.idx_to_char[idx] for idx in indices]

                # We'll replace any unknown characters with a space for readability
                pred_chars = [c if c != '<unk>' else ' ' for c in pred_chars]
                
                preds.append(''.join(pred_chars))  # join the list of chars into a string

        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        torch.save(self.model, "model.pt")
        
        '''
        save_path = os.path.join(work_dir, "model.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "n": self.n,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim
        }, save_path)
        '''

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        model = torch.load("model.pt")


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
