## Predicting the Next Word in a Simple Sentence

This repository contains code for a simple language model that predicts the next word in a given sentence. The model is trained using a GRU-based neural network and utilizes the NLTK library for tokenization.

### Repository Specifications:
Repository Link: [Predicting-the-next-word-in-a-simple-sentence](https://github.com/armansouri9/Predicting-the-next-word-in-a-simple-sentence)

### Code:

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import torch
from torch import nn, optim

class Tokenizer:
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf8') as f:
            self.data = f.read().replace('\n', '')

    def tokenize(self):
        return list(self.data)

class Vocab:
    def __init__(self, tokens):
        self.token_to_index = {}
        self.index_to_token = {}
        self.token_frequency = {}

        self.add_token('')
        self.add_token('')

        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_index:
            index = len(self.token_to_index)
            self.token_to_index[token] = index
            self.index_to_token[index] = token
            self.token_frequency[token] = 1
        else:
            self.token_frequency[token] += 1

    def lookup_token(self, token):
        if token in self.token_to_index:
            return self.token_to_index[token]
        else:
            return self.token_to_index['']

    def lookup_index(self, index):
        if index in self.index_to_token:
            return self.index_to_token[index]
        else:
            return ''

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(SimpleLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

def train_model(model, train_data, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for input, target in train_data:
            model.zero_grad()

            output, hidden = model(input)
            loss = criterion(output.view(-1, model.vocab_size), target.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('Epoch:', epoch + 1, 'Loss:', total_loss)

    return model

def predict(model, input_text, vocab, tokenizer):
    input_tokens = word_tokenize(input_text)

    input_token_ids = [vocab.lookup_token(token) for token in input_tokens]
    input_tensor = torch.LongTensor([input_token_ids])

    hidden = None
    output_tokens = []
    used_tokens = {}
    with torch.no_grad():
        while True:
            output, hidden = model(input_tensor, hidden)
            last_output = output.squeeze(0)[-1]
            probabilities = nn.functional.softmax(last_output, dim=0)
            next_token_id = torch.argmax

(probabilities).item()
            output_tokens.append(next_token_id)
            if vocab.lookup_index(next_token_id) == '' or len(output_tokens) >= 1000:
                break
            input_tensor = torch.LongTensor([[next_token_id]])

    output_text = [vocab.lookup_index(token_id) for token_id in output_tokens]
    output_text = ' '.join(output_text)
    return output_text

# Reading the training text
tokenizer = Tokenizer('/content/data.txt')
tokens = tokenizer.tokenize()
vocab = Vocab(tokens)

# Converting data to numerical vectors
token_ids = [vocab.lookup_token(token) for token in tokens]
input_sequence = torch.LongTensor(token_ids[:-1]).unsqueeze(0)
target_sequence = torch.LongTensor(token_ids[1:]).unsqueeze(0)

# Defining and training the model
model = SimpleLanguageModel(len(vocab.token_to_index), 50, 100)
model = train_model(model, [(input_sequence, target_sequence)], 500, 0.001)

# Predicting the next word in a simple sentence
input_text = 'alireza Mansouri work as1'
predicted_text = predict(model, input_text, vocab, tokenizer)
a=predicted_text.replace('  ',',')
a=a.replace(' ','')
a=a.replace(',',' ')
from collections import Counter

sentences = a.split('.')

# Counting the occurrence of each sentence
sentence_counts = Counter(sentences)

# Repeated sentences (appearing more than once)
repeated_sentences = [sentence for sentence in sentences if sentence_counts[sentence] > 1]

# Unique sentences containing the input text
unique_sentences = list(set(sentences))
unique_sentences = [i for i in unique_sentences if input_text in i]
print(max(unique_sentences, key=len))
```

The code consists of the following components:

1. Tokenizer: A class that reads the training text and tokenizes it.
2. Vocab: A class that builds the vocabulary based on the tokens.
3. SimpleLanguageModel: A class defining the language model architecture using GRU.
4. train_model: A function to train the language model using the provided training data.
5. predict: A function that predicts the next word in a given input sentence.
6. Main code: It reads the training text, converts the data to numerical vectors, trains the model, and finally predicts the next word in a simple sentence based on the trained model.

The predicted output is printed, which is the longest unique sentence containing the input text.

Please note that this is just a sample code, and you may need to modify it according to your specific requirements.

## License

This project is licensed under a Free License.
