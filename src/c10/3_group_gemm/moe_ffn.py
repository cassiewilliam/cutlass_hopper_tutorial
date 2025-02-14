import torch
from torch import nn

class MLP(nn.Module):
    
    def __init__(self, config, index):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.ffn = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        ])
    
    def forward(self, x):
        return self.ffn(x)

class SwitchMoE(nn.Module):
    
    def __init__(self, config, index):
        super().__init__()

        self.hidden_dim = config["hidden_dim"]
        self.num_experts = config["num_experts"][index]
        self.top_experts_num = config["top_experts_num"]

        self.experts = nn.ModuleList([
            MLP(config, index=index)
            for index in range(self.num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):

        batch_size, seq_len, hidden_dim = x.shape

        x = x.reshape(batch_size * seq_len, hidden_dim)

        # input: [b * s, h] * weight: [h, num_experts] ==> output: [b * s, num_experts]
        gate_out = self.gate(x)

        print(gate_out.shape, gate_out)

        # input: [b * s, num_experts] ==> output: [b * s, top_experts_num]
        _, gate_indices = torch.topk(gate_out, self.top_experts_num, dim=-1)

        print(gate_indices.shape, gate_indices)

        # input: [b * s, top_experts_num] ==> output: [b * s, num_experts]
        one_hot = torch.nn.functional.one_hot(gate_indices, num_classes=self.num_experts).float()

        print(one_hot.shape, one_hot)
        # b0-token0: [0, 1], [[1, 0, 0, 0], [0, 1, 0, 0]]
        # b0-token1: [0, 1], [[1, 0, 0, 0], [0, 1, 0, 0]]

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            mask = one_hot[:, :, i].sum(dim=-1) > 0  # 将多个 top-k 映射到每个专家
            print(i, mask)
            if mask.any():  # 如果有输入分配到该专家
                print("x shape: ", x.shape)
                expert_input = x[mask]
                print("expert_input shape: ", expert_input.shape)
                expert_output = expert(expert_input)
                print("expert_output shape: ", expert_output.shape)
                expert_outputs.append((expert_output, mask))

        # 汇总专家输出
        output = torch.zeros_like(x)
        for expert_output, mask in expert_outputs:
            output[mask] += expert_output

        return output.reshape(batch_size, seq_len, hidden_dim)

batch_size = 1
seq_len = 2
hidden_dim = 128
x = torch.randn(batch_size, seq_len, hidden_dim).to("cuda")
moe_layer = SwitchMoE({"hidden_dim": hidden_dim, "num_experts": [4, 8, 16], "top_experts_num": 2}, 2).to("cuda")
y = moe_layer(x.to("cuda"))

# print(y)