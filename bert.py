# Load model and tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import torch
import numpy as np

class BertGradient:

    def __init__(self):

        model_name = "textattack/distilbert-base-uncased-imdb"

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='./data').cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

        self.embeddings = None
        self.embeddings_gradients = None

        handle = self._register_embedding_list_hook()
        hook = self._register_embedding_gradient_hooks()


    def _register_embedding_list_hook(self):
        def forward_hook(module, inputs, output):
            self.embeddings = output.squeeze(0).clone().cpu().detach().numpy()
        embedding_layer = self.model.get_input_embeddings()
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _register_embedding_gradient_hooks(self):
        def hook_layers(module, grad_in, grad_out):
            self.embeddings_gradients = grad_out[0].cpu().numpy()
        embedding_layer = self.model.get_input_embeddings()
        hook = embedding_layer.register_backward_hook(hook_layers)
        return hook

    def embedding(self, inp):
        x = self.tokenizer(inp, is_split_into_words=True, return_tensors='pt',
                           padding='max_length', truncation=True, max_length=45)
        x = self.assign_gpu(x)
        return self.model.get_input_embeddings()(x['input_ids']).detach().flatten(start_dim=1).cpu().numpy()

    def bbsds(self, inp, batch_size=128):
        x = self.tokenizer(inp, is_split_into_words=True, return_tensors='pt',
                           padding='max_length', truncation=True, max_length=45)

        dataset = torch.utils.data.TensorDataset(x['input_ids'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        all_out = np.zeros((0,2))

        for b in dataloader:

            batch = b[0].cuda()
            out = torch.softmax(self.model(batch).logits, dim=-1).detach().cpu().numpy()
            all_out = np.concatenate((all_out, out), axis=0)

        return all_out

    def assign_gpu(self, x):
        input_ids = x['input_ids'].to('cuda:0')
        attention_mask = x['attention_mask'].to('cuda:0')

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask}

        return output

    def grad_x_input(self, inp, batch_size=128):
        x = self.tokenizer(inp, is_split_into_words=True, return_tensors='pt',
                           padding='max_length', truncation=True, max_length=45)

        dataset = torch.utils.data.TensorDataset(x['input_ids'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        all_embeds = np.zeros((0,34560))

        for b in dataloader:

            batch = b[0].cuda()

            self.embeddings_list = []
            self.embeddings_gradients = []

            output = torch.softmax(self.model(batch).logits, dim=-1)
            ind = output.data.max(1)[1]

            probvalue = 1.0

            grad_out = output.data.clone()
            grad_out.fill_(0.0)

            grad_out.scatter_(1, ind.unsqueeze(0).t(), probvalue)
            self.model.zero_grad()
            output.backward(grad_out)

            gxi = (self.embeddings_gradients * self.embeddings).reshape(self.embeddings.shape[0],-1)
            all_embeds = np.concatenate((all_embeds, gxi),axis=0)

        return all_embeds

