import torch
import torch.nn.functional as F
from torch import optim
from model.embeddings import EmbeddingNet

class MetaLearner:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001):
        self.model = model
        self.lr_inner = lr_inner
        self.optimizer = optim.Adam(model.parameters(), lr=lr_outer)

    def adapt(self, support_x, support_y):
        cloned_model = EmbeddingNet(*self.model.args)
        cloned_model.load_state_dict(self.model.state_dict())
        cloned_model.train()

        for param in cloned_model.parameters():
            param.requires_grad = True

        pred = cloned_model.fc(support_x)
        loss = F.mse_loss(pred.view(-1), support_y.view(-1))

        grads = torch.autograd.grad(
            loss,
            cloned_model.parameters(),
            create_graph=True,
            allow_unused=True
        )

        adapted_state = {}
        for (name, param), grad in zip(cloned_model.named_parameters(), grads):
            if grad is not None:
                adapted_state[name] = param - self.lr_inner * grad
            else:
                adapted_state[name] = param  # no update

        for name, param in cloned_model.named_parameters():
            param.data.copy_(adapted_state[name].data)

        return cloned_model

    def train_task(self, support_x, support_y, query_x, query_y):
        self.model.train()
        adapted_model = self.adapt(support_x, support_y)

        pred = adapted_model.fc(query_x)
        loss = F.mse_loss(pred.view(-1), query_y.view(-1))

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)  # ✅ Fix to avoid double-backward error
        self.optimizer.step()

        return loss.item()
