import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# data = pd.read_csv('./output_file_name.csv')
# text = data['review']
# label = data['sentiment']
data = pd.read_csv('./data.csv')
text = data['text']
label = data['label']
# print(text)
# print(label)
word_to_idx = {}
idx_to_word = {}

for sentence in text:
    for word in sentence.split():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
            idx_to_word[len(idx_to_word)] = word

vocab_size = len(word_to_idx)
embed_dim = 100
embedding_weight = np.random.uniform(-0.01, 0.01, (vocab_size, embed_dim))


class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(WordEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(embedding_weight), requires_grad=True)

    def forward(self, x):
        return self.embeddings(x)


embedding = WordEmbedding(vocab_size, embed_dim)
# print(embedding)


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


texts = []
for sentence in text:
    tmp = []
    for word in sentence.split():
        tmp.append(word_to_idx[word])
    texts.append(torch.LongTensor(tmp))

# normalize the embeddings matrix
normalized_embedding = normalize(embedding_weight)
embedding.embeddings.weight = nn.Parameter(torch.FloatTensor(normalized_embedding), requires_grad=True)

# pad sequences with 0
padded_text = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)


class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super(SentimentLSTM, self).__init__()
        self.embedding = WordEmbedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.output_layer(out[:, -1, :])
        return out


hidden_size = 128
output_size = 1
model = SentimentLSTM(vocab_size, embed_dim, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_text, label.values, test_size=0.2, random_state=42)
train_data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
test_data = [(X_test[i], y_test[i]) for i in range(len(X_test))]

# train the model
epochs = 10
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader), 1):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    test_loss = 0.0
    for i, data in enumerate(tqdm(test_loader), 1):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels.float())
        test_loss += loss.item()

    print('Epoch %d: train loss=%.3f, test loss=%.3f' % (
    epoch + 1, running_loss / len(train_loader), test_loss / len(test_loader)))


def predict_sentiment(model, sentence):
    model.eval()
    words = sentence.split()
    indexes = [word_to_idx.get(word, 0) for word in words] # 使用get方法获取词对应的索引，如果不存在则返回0
    tensor = torch.LongTensor(indexes).unsqueeze(0)
    prediction = torch.sigmoid(model(tensor)).item()
    return prediction


# example usage
print(predict_sentiment(model, "I really love this movie!"))
print(predict_sentiment(model, "I hate such boring movie."))

