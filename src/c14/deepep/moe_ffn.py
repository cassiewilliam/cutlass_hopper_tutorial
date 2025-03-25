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

class GroupGEMMMLP(nn.Module):
    def __init__(self, config, num_groups = 1):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_groups = num_groups
        
        self.w1 = nn.Parameter(torch.randn(num_groups, self.hidden_dim, self.hidden_dim * 4))
        self.w2 = nn.Parameter(torch.randn(num_groups, self.hidden_dim * 4, self.hidden_dim * 4))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, groups, hidden_dim)
        batch, groups, _ = x.shape
        
        # calculate first group gemm layer
        x = torch.bmm(x, self.w1)
        x = self.relu(x)

        # calculate second group gemm layer
        x = torch.bmm(x, self.w2)
        return x

class SwitchMoE(nn.Module):
    
    def __init__(self, config, index, use_group_gemm = True):
        super().__init__()

        self.hidden_dim = config["hidden_dim"]
        self.num_experts = config["num_experts"][index]
        self.top_experts_num = config["top_experts_num"]
        self.use_group_gemm = use_group_gemm

        self.experts = GroupGEMMMLP(config=config) if use_group_gemm else nn.ModuleList([
            MLP(config, index=index)
            for index in range(self.num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):

        print("input x shape(batch_size, seq_len, hidden_dim): ", x.shape)
        batch_size, seq_len, hidden_dim = x.shape

        x = x.reshape(batch_size * seq_len, hidden_dim)

        print("reshape x shape(batch_size * seq_len, hidden_dim): ", x.shape)
        # input: [b * s, h] * weight: [h, num_experts] ==> output: [b * s, num_experts]
        gate_out = self.gate(x)
        print("run gate layer: input: [b * s, h] * weight: [h, num_experts] ==> output: [b * s, num_experts]")

        print("gate_out shape(b * s, num_experts): ", gate_out.shape)

        # input: [b * s, num_experts] ==> output: [b * s, top_experts_num]
        _, gate_indices = torch.topk(gate_out, self.top_experts_num, dim=-1)
        print("run topk layer: input: [b * s, num_experts] ==> output: [b * s, top_experts_num]")

        print("gate_indices shape(b * s, top_experts_num): ", gate_indices.shape)
        
        print(gate_indices)

        # input: [b * s, top_experts_num] ==> output: [b * s, num_experts]
        one_hot = torch.nn.functional.one_hot(gate_indices, num_classes=self.num_experts).float()
        print("run one_hot layer: [b * s, top_experts_num] ==> output: [b * s, num_experts]")
        
        print("one_hot shape(b * s, num_experts): ", one_hot.shape)
        # b0-token0: [0, 1], [[1, 0, 0, 0], [0, 1, 0, 0]]
        # b0-token1: [0, 1], [[1, 0, 0, 0], [0, 1, 0, 0]]
        
        print(one_hot)

        # generate masks
        masks = []
        for i in range(self.num_experts):
            mask = one_hot[:, :, i].sum(dim=-1) > 0  # 将多个 top-k 映射到每个专家
            masks.append(mask)

        expert_inputs = [x[mask] for mask in masks]
        
        print(x)
        print(len(expert_inputs), expert_inputs)

        group_gemm = False
        if not group_gemm:
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                mask = one_hot[:, :, i].sum(dim=-1) > 0  # 将多个 top-k 映射到每个专家
                if mask.any():  # 如果有输入分配到该专家
                    print("x shape: ", x.shape)
                    expert_input = x[mask]
                    print("expert_input shape: ", expert_input.shape)
                    expert_output = expert(expert_input)
                    print("expert_output shape: ", expert_output.shape)
                    expert_outputs.append((expert_output, mask))

            # 汇总专家输出
            output = torch.zeros_like(x)
            print("output shape: ", output.shape)
            for expert_output, mask in expert_outputs:
                output[mask] += expert_output
        else:
            # **使用 GroupGEMMMLP 并行计算**
            # 1. 统计每个专家的输入数据
            expert_inputs = [x[mask] for mask in masks]
            

        return output.reshape(batch_size, seq_len, hidden_dim)

batch_size = 1
seq_len = 2
hidden_dim = 128
x = torch.randn(batch_size, seq_len, hidden_dim).to("cuda")
moe_layer = SwitchMoE({"hidden_dim": hidden_dim, "num_experts": [4, 8, 16], "top_experts_num": 2}, 0).to("cuda")
y = moe_layer(x.to("cuda"))

# print(y)