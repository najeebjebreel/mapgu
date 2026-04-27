"""Neural network architectures: TabNet family and MLPModel for tabular classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GBN(nn.Module):
    def __init__(self, inp, vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        chunk = torch.chunk(x, max(1, x.size(0) // self.vbs), 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


class GLU(nn.Module):
    def __init__(self, inp_dim, out_dim, fc=None, vbs=128):
        super().__init__()
        self.fc = fc if fc else nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim

    def forward(self, x):
        x = self.bn(self.fc(x))
        return x[:, :self.od] * torch.sigmoid(x[:, self.od:])


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs=128):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))  # fixed: was `inp` (NameError)
        for _ in range(int(first), n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        # register_buffer so tensor moves with the model on .to(device)
        self.register_buffer('scale', torch.sqrt(torch.tensor([0.5])))

    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class AttentionTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, relax, vbs=128):
        super().__init__()
        self.fc = nn.Linear(inp_dim, out_dim)
        self.bn = GBN(out_dim, vbs=vbs)
        # register_buffer so tensor moves with the model on .to(device)
        self.register_buffer('r', torch.tensor([relax]))

    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = torch.sigmoid(a * priors)
        priors = priors * (self.r - mask)
        return mask


class DecisionStep(nn.Module):
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, loss


class TabNet(nn.Module):
    def __init__(self, inp_dim, final_out_dim, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=128):
        super().__init__()
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for _ in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))
        else:
            self.shared = None
        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind)
        self.steps = nn.ModuleList()
        for _ in range(n_steps - 1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs))
        self.fc = nn.Linear(n_d, final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d

    def forward(self, x):
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d:]
        loss = torch.zeros(1, device=x.device)
        out = torch.zeros(x.size(0), self.n_d, device=x.device)
        priors = torch.ones(x.shape, device=x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:, :self.n_d])
            x_a = x_te[:, self.n_d:]
            loss += l
        return self.fc(out), loss


class TabNetWithEmbed(nn.Module):
    def __init__(self, inp_dim, final_out_dim, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=128, cat_dims=None):
        super().__init__()
        self.tabnet = TabNet(inp_dim, final_out_dim, n_d, n_a, n_shared, n_ind, n_steps, relax, vbs)
        self.cat_embed = nn.ModuleList()
        if cat_dims is not None:
            for d in cat_dims:
                self.cat_embed.append(nn.Embedding(d, 2))

    def forward(self, contv, catv=None):
        x = contv
        if catv is not None and self.cat_embed:
            embeddings = [embed(catv[:, idx]) for idx, embed in enumerate(self.cat_embed)]
            catv_ = torch.cat(embeddings, 1)
            x = torch.cat((catv_, contv), 1).contiguous()
        x, _ = self.tabnet(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, include_bn=False):
        super().__init__()
        self.include_bn = include_bn
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        if include_bn:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        if self.include_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.include_bn:
            x = self.bn2(x)
        x = F.relu(x)
        return self.fc3(x)
