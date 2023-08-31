from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


class DoseChangePredictionDataset(Dataset):
    """Pytorch Dataset wrapper for our Indication prediction"""

    def __init__(self, examples, tokenizer: AutoTokenizer, transforms=None):
        # save the tokenizer
        self.tokenizer = tokenizer
        # save the transformer
        self.transforms = transforms
        # create a list flattening function
        self.flatten = lambda l: [item for sublist in l for item in sublist]
        # initialize a list to save the examples
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        text, label = self.examples[index]
        # map tokens to subword tokens
        token_ids_text = self.tokenizer.encode_plus(
            text=text, add_special_tokens=True
        )['input_ids']
        return (
            torch.tensor(token_ids_text, dtype=torch.long),
            label,
        )


def collate_fn(tokenizer: AutoTokenizer, batch):
    # unpack the x and y values from the batch (list of size batch_size, with each element containing the return of the dataset iterator)
    x_batch, y_batch = zip(*batch)
    x_batch_padded = pad_sequence(
        x_batch, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # pad the target values
    # return the normalized tensors
    return (
        x_batch_padded.to(device),
        torch.tensor(y_batch, dtype=torch.long).to(device),
    )
