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

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


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
