import os
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops
import matplotlib.pyplot as plt
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
print(ms.context.get_context("device_target"))

# ------------------ DeepONet 定义 ------------------
class MLP(nn.Cell):
    def __init__(self, layer_sizes, activation="tanh", last_activation=None):
        super().__init__()
        act_map = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Dense(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(act_map[activation])
        layers.append(nn.Dense(layer_sizes[-2], layer_sizes[-1]))
        if last_activation:
            layers.append(act_map[last_activation])
        self.net = nn.SequentialCell(layers)
    def construct(self, x):
        return self.net(x)

class DeepONet(nn.Cell):
    def __init__(self, layers_u, layers_y):
        super().__init__()
        self.net_u = MLP(layers_u, activation="tanh")
        self.net_y = MLP(layers_y, activation="tanh", last_activation="tanh")
        self.b0 = ms.Parameter(ms.Tensor(0.0, dtype=ms.float32))
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
    def construct(self, x_u, x_y):
        net_u = self.net_u(x_u)
        net_y = self.net_y(x_y)
        net_o = self.reduce_sum(net_u * net_y, 1) + self.b0
        return net_o

# ------------------ 配置参数 ------------------
class EvalConfig:
    def __init__(self):
        self.layers_u = [100, 64, 64, 64]
        self.layers_y = [1, 64, 64, 64]
        self.num_loc = 100  # 与训练输入长度一致
        self.num_y = 100    # 取样点数
        self.model_ckpt = "./ckpt/deeponet_best.ckpt"
        self.output_dir = "./eval_visual"
        os.makedirs(self.output_dir, exist_ok=True)

cfg = EvalConfig()

# ------------------ 载入模型 ------------------
net = DeepONet(cfg.layers_u, cfg.layers_y)
param_dict = ms.load_checkpoint(cfg.model_ckpt)
ms.load_param_into_net(net, param_dict)
net.set_train(False)

# ------------------ 测试函数族及可视化 ------------------
def generate_y_u_G_ref(u_func, G_func):
    x = np.linspace(0, 1, cfg.num_loc).reshape(1, cfg.num_loc)
    u = u_func(x)                          # shape: [1, num_loc]
    u = np.tile(u, [cfg.num_y, 1])         # shape: [num_y, num_loc]
    y = np.linspace(0, 1, cfg.num_y).reshape(cfg.num_y, 1)
    G_ref = G_func(y)                      # shape: [num_y, 1]
    return u.astype(np.float32), y.astype(np.float32), G_ref.astype(np.float32)

func_u_G_pair = [
    (r"$u=\cos(x),\,G(u)=\sin(x)$", lambda x: np.cos(x), lambda y: np.sin(y)),
    (r"$u=\sec^2(x),\,G(u)=\tan(x)$", lambda x: 1/np.cos(x)**2, lambda y: np.tan(y)),
    (r"$u=\sec(x)\tan(x),\,G(u)=\sec(x)-1$", lambda x: 1/np.cos(x)*np.tan(x), lambda y: 1/np.cos(y)-1),
    (r"$u=1.5^x\ln{1.5},\,G(u)=1.5^x-1$", lambda x: 1.5**x*np.log(1.5), lambda y: 1.5**y-1),
    (r"$u=3x^2,\,G(u)=x^3$", lambda x: 3*x**2, lambda y: y**3),
    (r"$u=4x^3,\,G(u)=x^4$", lambda x: 4*x**3, lambda y: y**4),
    (r"$u=5x^4,\,G(u)=x^5$", lambda x: 5*x**4, lambda y: y**5),
    (r"$u=6x^5,\,G(u)=x^6$", lambda x: 6*x**5, lambda y: y**6),
    (r"$u=e^x,\,G(u)=e^x-1$", lambda x: np.exp(x), lambda y: np.exp(y)-1),
]

for i, (title, u_func, G_func) in enumerate(func_u_G_pair):
    u, y, G_ref = generate_y_u_G_ref(u_func, G_func)
    # MindSpore模型预测
    u_tensor = Tensor(u, ms.float32)
    y_tensor = Tensor(y, ms.float32)
    G_pred = net(u_tensor, y_tensor).asnumpy()
    # 可视化
    plt.figure(figsize=(7, 4))
    plt.plot(y, G_ref, label="Analytical $G(u)$", lw=2)
    plt.plot(y, G_pred, '--', label="DeepONet $G(u)$ Prediction", lw=2)
    plt.xlabel(r"$y$")
    plt.ylabel(r"$G(u)(y)$")
    plt.title(title, fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    img_path = os.path.join(cfg.output_dir, f"func_{i+1}_result.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Saved: {img_path}")

print(f"All visual results saved in {cfg.output_dir}")

