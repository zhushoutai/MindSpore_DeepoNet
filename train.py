# import os
# import numpy as np
# import mindspore as ms
# from mindspore import nn, Tensor, context, ops
# import mindspore.dataset as ds
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
# print(ms.context.get_context("device_target")) 
# # ---------- 简单MLP模块 ----------
# class MLP(nn.Cell):
#     def __init__(self, layer_sizes, activation="tanh", last_activation=None):
#         super().__init__()
#         act_map = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}
#         layers = []
#         for i in range(len(layer_sizes) - 2):
#             layers.append(nn.Dense(layer_sizes[i], layer_sizes[i+1]))
#             layers.append(act_map[activation])
#         layers.append(nn.Dense(layer_sizes[-2], layer_sizes[-1]))
#         if last_activation:
#             layers.append(act_map[last_activation])
#         self.net = nn.SequentialCell(layers)
#     def construct(self, x):
#         return self.net(x)

# # ---------- DeepONet ----------
# class DeepONet(nn.Cell):
#     def __init__(self, layers_u, layers_y):
#         super().__init__()
#         self.net_u = MLP(layers_u, activation="tanh")
#         self.net_y = MLP(layers_y, activation="tanh", last_activation="tanh")
#         self.b0 = ms.Parameter(ms.Tensor(0.0, dtype=ms.float32))
#         self.reduce_sum = ops.ReduceSum(keep_dims=True)
#     def construct(self, x_u, x_y):
#         net_u = self.net_u(x_u)
#         net_y = self.net_y(x_y)
#         net_o = self.reduce_sum(net_u * net_y, 1) + self.b0
#         return net_o

# # ---------- 数据集工具 ----------
# def get_dataset(x_u, x_y, y, batch_size=128, shuffle=True):
#     data = {"x_u": x_u.astype(np.float32), "x_y": x_y.astype(np.float32), "y": y.astype(np.float32)}
#     ds_ = ds.NumpySlicesDataset(data, shuffle=shuffle)
#     ds_ = ds_.batch(batch_size)
#     return ds_

# # ---------- 训练和验证 ----------
# def train_and_validate(args):
#     print("dataing")
#     # 1. 加载数据
#     npz_train = np.load(args["TRAIN_FILE_PATH"])
#     npz_test = np.load(args["VALID_FILE_PATH"])
#     x_u_train = npz_train["X_train0"]
#     x_y_train = npz_train["X_train1"]
#     y_train = npz_train["y_train"]
#     x_u_test = npz_test["X_test0"]
#     x_y_test = npz_test["X_test1"]
#     y_test = npz_test["y_test"]

#     # 2. 网络与损失优化器
#     net = DeepONet(args["layers_u"], args["layers_y"])
#     criterion = nn.MSELoss()
#     optimizer = nn.Adam(net.trainable_params(), learning_rate=args["lr"])

#     # 3. 标准MindSpore Dataset
#     train_dataset = get_dataset(x_u_train, x_y_train, y_train, args["batch_size"], shuffle=True)
#     test_dataset = get_dataset(x_u_test, x_y_test, y_test, args["batch_size"], shuffle=False)

#     best_rel_err = 1e10

#     for epoch in tqdm(range(1, args["epochs"]+1)):
#         net.set_train(True)
#         train_loss_list = []
#         for batch in train_dataset.create_dict_iterator():
#             u = batch["x_u"]
#             y_ = batch["x_y"]
#             gt = batch["y"]
#             def forward_fn(u, y_, gt):
#                 pred = net(u, y_)
#                 loss = criterion(pred, gt)
#                 return loss
#             grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
#             loss, grads = grad_fn(u, y_, gt)
#             optimizer(grads)
#             train_loss_list.append(loss.asnumpy())
#         mean_train_loss = np.mean(train_loss_list)

#         # 验证
#         print("validating")
#         net.set_train(False)
#         val_loss_list = []
#         rel_err_list = []
#         for batch in test_dataset.create_dict_iterator():
#             u = batch["x_u"]
#             y_ = batch["x_y"]
#             gt = batch["y"]
#             pred = net(u, y_)
#             loss = criterion(pred, gt)
#             rel = np.linalg.norm(pred.asnumpy() - gt.asnumpy()) / (np.linalg.norm(gt.asnumpy()) + 1e-12)
#             val_loss_list.append(loss.asnumpy())
#             rel_err_list.append(rel)
#         mean_val_loss = np.mean(val_loss_list)
#         mean_rel_err = np.mean(rel_err_list)
#         # 输出日志
#         if epoch % args["print_interval"] == 0:
#             print(f"Epoch {epoch}: train_loss={mean_train_loss:.6e}, test_loss={mean_val_loss:.6e}, rel_err={mean_rel_err:.6e}")
#         # 自动保存最佳
#         if mean_rel_err < best_rel_err:
#             ms.save_checkpoint(net, os.path.join(args["save_ckpt_path"], "deeponet_best.ckpt"))
#             best_rel_err = mean_rel_err

#     # 最终保存
#     ms.save_checkpoint(net, os.path.join(args["save_ckpt_path"], "deeponet_last.ckpt"))
#     print("Training finished.")

#     # 可视化（随机取一批测试样本预测）
#     net.set_train(False)
#     test_batch = next(test_dataset.create_dict_iterator())
#     pred = net(test_batch["x_u"], test_batch["x_y"]).asnumpy()
#     gt = test_batch["y"].asnumpy()
#     plot_prediction(pred, gt, os.path.join(args["figures_path"], "prediction_vs_gt.png"))

# # ---------- 可视化工具 ----------
# def plot_prediction(y_pred, y_test, save_path):
#     plt.figure()
#     plt.plot(y_test, label="Ground Truth")
#     plt.plot(y_pred, label="Prediction")
#     plt.legend()
#     plt.title("Prediction vs Ground Truth")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Prediction plot saved: {save_path}")

# # ---------- 参数配置及主入口 ----------

# args = {
#     "TRAIN_FILE_PATH": "antiderivative_unaligned_train.npz",
#     "VALID_FILE_PATH": "antiderivative_unaligned_test.npz",
#     "layers_u": [100, 64, 64, 64],
#     "layers_y": [1, 64, 64, 64],
#     "batch_size": 2048,
#     "epochs": 1000,
#     "lr": 0.001,
#     "print_interval": 10,
#     "save_ckpt_path": "./ckpt",
#     "figures_path": "./figures",
# }
# os.makedirs(args["save_ckpt_path"], exist_ok=True)
# os.makedirs(args["figures_path"], exist_ok=True)
# train_and_validate(args)





import os
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, context, ops
import mindspore.dataset as ds
import matplotlib.pyplot as plt
from tqdm import tqdm

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
print(ms.context.get_context("device_target"))

# ---------- 简单MLP模块 ----------
class MLP(nn.Cell):
    def __init__(self, layer_sizes, activation="tanh", last_activation=None):
        super().__init__()
        act_map = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Dense(layer_sizes[i], layer_sizes[i+1]))
            layers.append(act_map[activation])
        layers.append(nn.Dense(layer_sizes[-2], layer_sizes[-1]))
        if last_activation:
            layers.append(act_map[last_activation])
        self.net = nn.SequentialCell(layers)

    def construct(self, x):
        return self.net(x)

# ---------- DeepONet ----------
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

# ---------- 数据集工具 ----------
def get_dataset(x_u, x_y, y, batch_size=128, shuffle=True):
    data = {"x_u": x_u.astype(np.float32), "x_y": x_y.astype(np.float32), "y": y.astype(np.float32)}
    ds_ = ds.NumpySlicesDataset(data, shuffle=shuffle)
    ds_ = ds_.batch(batch_size)
    return ds_

# ---------- 训练和验证 ----------
def train_and_validate(args):
    print("dataing")
    # 1. 加载数据
    npz_train = np.load(args["TRAIN_FILE_PATH"])
    npz_test = np.load(args["VALID_FILE_PATH"])
    x_u_train = npz_train["X_train0"]
    x_y_train = npz_train["X_train1"]
    y_train = npz_train["y_train"]
    x_u_test = npz_test["X_test0"]
    x_y_test = npz_test["X_test1"]
    y_test = npz_test["y_test"]

    # 2. 网络与损失优化器
    net = DeepONet(args["layers_u"], args["layers_y"])
    criterion = nn.MSELoss()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=args["lr"])

    # 3. 标准MindSpore Dataset
    train_dataset = get_dataset(x_u_train, x_y_train, y_train, args["batch_size"], shuffle=True)
    test_dataset = get_dataset(x_u_test, x_y_test, y_test, args["batch_size"], shuffle=False)

    # 用于记录每个epoch的损失和误差
    train_losses = []
    val_losses = []
    rel_errors = []

    best_rel_err = 1e10

    for epoch in tqdm(range(1, args["epochs"]+1)):
        net.set_train(True)
        train_loss_list = []
        for batch in train_dataset.create_dict_iterator():
            u = batch["x_u"]
            y_ = batch["x_y"]
            gt = batch["y"]
            def forward_fn(u, y_, gt):
                pred = net(u, y_)
                loss = criterion(pred, gt)
                return loss
            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
            loss, grads = grad_fn(u, y_, gt)
            optimizer(grads)
            train_loss_list.append(loss.asnumpy())
        mean_train_loss = np.mean(train_loss_list)

        # 验证
        print("validating")
        net.set_train(False)
        val_loss_list = []
        rel_err_list = []
        for batch in test_dataset.create_dict_iterator():
            u = batch["x_u"]
            y_ = batch["x_y"]
            gt = batch["y"]
            pred = net(u, y_)
            loss = criterion(pred, gt)
            rel = np.linalg.norm(pred.asnumpy() - gt.asnumpy()) / (np.linalg.norm(gt.asnumpy()) + 1e-12)
            val_loss_list.append(loss.asnumpy())
            rel_err_list.append(rel)
        mean_val_loss = np.mean(val_loss_list)
        mean_rel_err = np.mean(rel_err_list)

        # 输出日志
        if epoch % args["print_interval"] == 0:
            print(f"Epoch {epoch}: train_loss={mean_train_loss:.6e}, test_loss={mean_val_loss:.6e}, rel_err={mean_rel_err:.6e}")

        # 保存每个epoch的损失和误差
        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)
        rel_errors.append(mean_rel_err)

        # 自动保存最佳
        if mean_rel_err < best_rel_err:
            ms.save_checkpoint(net, os.path.join(args["save_ckpt_path"], "deeponet_best.ckpt"))
            best_rel_err = mean_rel_err

    # 最终保存
    ms.save_checkpoint(net, os.path.join(args["save_ckpt_path"], "deeponet_last.ckpt"))
    print("Training finished.")

    # 可视化训练过程中的loss和误差图
    plot_loss(train_losses, val_losses, rel_errors, args["figures_path"])

    # 可视化（随机取一批测试样本预测）
    net.set_train(False)
    test_batch = next(test_dataset.create_dict_iterator())
    pred = net(test_batch["x_u"], test_batch["x_y"]).asnumpy()
    gt = test_batch["y"].asnumpy()
    plot_prediction(pred, gt, os.path.join(args["figures_path"], "prediction_vs_gt.png"))

# ---------- 可视化工具 ----------

def plot_loss(train_losses, val_losses, rel_errors, save_path):
    plt.figure(figsize=(12, 4))

    # 训练损失
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()

    # 验证损失
    plt.subplot(1, 3, 2)
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    # 误差
    plt.subplot(1, 3, 3)
    plt.plot(rel_errors, label="Relative Error", color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    plt.title('Relative Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_and_error_plots.png"))
    plt.close()
    print(f"Loss and Error plots saved: {save_path}/loss_and_error_plots.png")

def plot_prediction(y_pred, y_test, save_path):
    plt.figure()
    plt.plot(y_test, label="Ground Truth")
    plt.plot(y_pred, label="Prediction")
    plt.legend()
    plt.title("Prediction vs Ground Truth")
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plot saved: {save_path}")

# ---------- 参数配置及主入口 ----------

args = {
    "TRAIN_FILE_PATH": "antiderivative_unaligned_train.npz",
    "VALID_FILE_PATH": "antiderivative_unaligned_test.npz",
    "layers_u": [100, 64, 64, 64],
    "layers_y": [1, 64, 64, 64],
    "batch_size": 4096,
    "epochs": 100,
    "lr": 0.002,
    "print_interval": 10,
    "save_ckpt_path": "./ckpts",
    "figures_path": "./figures",
}

os.makedirs(args["save_ckpt_path"], exist_ok=True)
os.makedirs(args["figures_path"], exist_ok=True)
train_and_validate(args)
