import torch

class IMDBCNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        super(IMDBCNN, self).__init__()
        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)
        self.cnn = torch.nn.Conv1d(embedding_dim, hidden_dim, kernel_size=10, stride=5)
        self.cnn2 = torch.nn.Conv1d(hidden_dim, 50, kernel_size=10,stride=7)
        self.act = torch.nn.ReLU()
        self.fc = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embeds = self.emb(x)
        return self.forward_embed(embeds)

    def forward_embed(self, x):
        x = torch.transpose(x, 1, 2)
        cnn_out = self.cnn(x)
        cnn_out = self.act(cnn_out)
        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.act(cnn_out)
        out = self.fc(cnn_out.view(cnn_out.shape[0], -1))
        return out

    def get_embed(self, x):
        return self.emb(torch.tensor(x, dtype=torch.long).cuda())

class IMDBLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        super(IMDBLSTM, self).__init__()
        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim,hidden_size=hidden_dim,
                           num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 2)

    def forward(self,x):
        embeds = self.emb(x)
        return self.forward_embed(embeds)

    def forward_embed(self,x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out).view(-1, 100,2)[:,-1]
        return out

    def get_embed(self,x):
        return self.emb(torch.tensor(x, dtype=torch.long).cuda())


def train(x_train, y_train, x_test, y_test, n_epoch=30, batch_size=5000):
    model = IMDBLSTM(200000, 256, 100).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for e in range(n_epoch):
        print("Epoch {}".format(e+1))

        out = model(torch.tensor(x_test[:batch_size], dtype=torch.long).cuda())
        acc = (batch_size - torch.sum(torch.abs((torch.max(out, dim=-1)[1].cpu() - y_test[:batch_size])))) / batch_size
        print("Test acc: {}".format(acc))

        for b_idx in range(int(len(x_train)/batch_size)):
            idx = b_idx*batch_size
            x_batch = torch.tensor(x_train[idx:idx+batch_size], dtype=torch.long).cuda()
            y_batch = torch.tensor(y_train[idx:idx+batch_size], dtype=torch.long).cuda()
            model.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            print("Loss: {}".format(loss))
        # Validation
        out = model(torch.tensor(x_test[:batch_size], dtype=torch.long).cuda())
        acc = (batch_size - torch.sum(torch.abs((torch.max(out, dim=-1)[1].cpu() - y_test[:batch_size])))) / batch_size
        print("Test acc: {}".format(acc))
    return model


def gradxinput(model, emb):
    torch.autograd.set_detect_anomaly = True
    emb_tensor = torch.tensor(emb, dtype=torch.float, requires_grad=True).cuda()
    emb_tensor.retain_grad()
    output = model.forward_embed(emb_tensor)
    ind = output.data.max(1)[1]
    grad_out = output.data.clone()
    grad_out.fill_(0.0)
    grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
    output.backward(grad_out)
    return emb_tensor.grad.data.cpu().numpy() * emb



