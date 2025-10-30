# copy dependencies from transformers/optimization.py (trimmed & adapted)
import math
import warnings
from typing import Callable, Generator, List, Iterable, Tuple, Optional, Union
from collections import defaultdict
from itertools import chain

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DeviceMesh, DTensor


# ---------------- Utility ----------------
def _to_scalar(x):
    """Tensor 标量 -> Python 标量；否则原样返回。"""
    return x.item() if isinstance(x, torch.Tensor) else x

def _get_capturable_supported_devices(supports_xla: bool = False):
    """返回支持 capturable 的设备集合（简化版）。"""
    return {"cpu", "cuda", "mps"}

def _get_value(x):
    """取 Python 标量值。"""
    return x.item() if isinstance(x, torch.Tensor) else x

def _stack_if_compiling(seq):
    """编译态可能 stack 标量张量；非编译直接返回。"""
    if torch.compiler.is_compiling():
        if isinstance(seq, (list, tuple)) and len(seq) > 0 and isinstance(seq[0], torch.Tensor):
            return torch.stack(list(seq))
    return seq

def _view_as_real(*args, **kwargs):
    """占位：未用到复数时空实现。"""
    return

DeviceDict = dict
DeviceDtypeDict = dict
ParamsT = list


# ---------------- DTensor helpers ----------------
def to_local(tensor: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
    """DTensor -> local Tensor（普通 Tensor 无操作）。"""
    if isinstance(tensor, Tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return [t.to_local() if isinstance(t, DTensor) else t for t in tensor]

def dtensor_from_local(
    tensor: Union[Tensor, List[Tensor]], ref: Tensor
) -> Union[DTensor, List[DTensor]]:
    """local Tensor -> DTensor，沿用参照张量的 mesh/placements；参照不是 DTensor 则直接返回输入。"""
    if not isinstance(ref, DTensor):
        assert isinstance(ref, Tensor)
        return tensor
    device_mesh = ref.device_mesh
    placements = ref.placements
    if isinstance(tensor, Tensor):
        assert not isinstance(tensor, DTensor)
        return DTensor.from_local(tensor, device_mesh=device_mesh, placements=placements)
    assert not isinstance(tensor[0], DTensor)
    return [DTensor.from_local(t, device_mesh=device_mesh, placements=placements) for t in tensor]


# ---------------- Parameter batching ----------------
def create_param_batches(params: List[Tensor], batch_size: int) -> Generator[List[Tensor], None, None]:
    """按 (shape, sharding, dtype) 分组并以 batch_size 切分。"""
    groups = defaultdict(list)
    for p in params:
        sharding = p.placements if isinstance(p, DTensor) else None
        groups[(p.shape, sharding, p.dtype)].append(p)
    for group in groups.values():
        for i in range(0, len(group), batch_size):
            yield group[i : i + batch_size]

def pad_batch(batch: List[Tensor], batch_size: int) -> List[Tensor]:
    """用空张量补齐到 batch_size。"""
    assert 0 < len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.empty_like(batch[0]))
    return batch


# ---------------- Lightweight async scheduler ----------------
class AsyncTask:
    """把生成器封装为任务：每次 run() 推进到下一个 yield。"""
    def __init__(self, generator: Generator[None, None, None]):
        self._generator = generator
        self.run()
    def run(self) -> bool:
        try:
            next(self._generator)
            return True
        except StopIteration:
            return False

class AsyncRuntime:
    """简单事件循环：并发执行多个 AsyncTask。"""
    def __init__(self, task_gen: Generator["AsyncTask", None, None], max_concurrent_tasks: int):
        if max_concurrent_tasks <= 0:
            raise ValueError(f"{max_concurrent_tasks=} cannot be <= 0")
        self._task_gen = task_gen
        self._max_concurrent_tasks = max_concurrent_tasks
    def _get_next_task(self) -> Optional["AsyncTask"]:
        try:
            return next(self._task_gen)
        except StopIteration:
            return None
    def run(self):
        have_new_tasks = True
        previous_tasks: List["AsyncTask"] = []
        while have_new_tasks or previous_tasks:
            running_tasks = []
            if have_new_tasks and len(previous_tasks) < self._max_concurrent_tasks:
                new_task = self._get_next_task()
                if new_task is not None:
                    running_tasks.append(new_task)
                else:
                    have_new_tasks = False
            for task in previous_tasks:
                if task.run():
                    running_tasks.append(task)
            previous_tasks = running_tasks


# ---------------- Multi-tensor C-AdamW (foreach path) ----------------
def _multi_tensor_c_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
    cautious: bool = False,
):
    """多张量 C-AdamW 更新（与 PyTorch foreach 风格一致，含 cautious mask）。"""
    if len(params) == 0:
        return

    if isinstance(lr, Tensor):
        if not capturable:
            raise RuntimeError("lr Tensor 需要 capturable=True")
        if lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
    if isinstance(beta1, Tensor):
        if not capturable:
            raise ValueError("beta1 Tensor 需要 capturable=True")
        if beta1.numel() != 1:
            raise ValueError("Tensor beta1 must be 1-element")
    if isinstance(beta2, Tensor):
        if not capturable:
            raise ValueError("beta2 Tensor 需要 capturable=True")
        if beta2.numel() != 1:
            raise ValueError("Tensor beta2 must be 1-element")

    if not torch.compiler.is_compiling() and capturable:
        capturable_supported = _get_capturable_supported_devices(False)
        assert all(
            p.device.type == step.device.type and p.device.type in capturable_supported
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on {capturable_supported}."

    assert grad_scale is None and found_inf is None
    assert not differentiable, "_foreach ops don't support autograd"

    lr = _to_scalar(lr)

    grouped = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )

    beta1_dict: Optional[DeviceDict] = (
        {beta1.device: beta1} if isinstance(beta1, Tensor) and str(beta1.device) != "cpu" else None
    )

    for (device_params_, device_grads_, device_exp_avgs_, device_exp_avg_sqs_, device_max_exp_avg_sqs_, device_state_steps_), _ in grouped.values():
        device_params = list(device_params_)  # type: ignore
        device_grads = list(device_grads_)    # type: ignore
        device_exp_avgs = list(device_exp_avgs_)  # type: ignore
        device_exp_avg_sqs = list(device_exp_avg_sqs_)  # type: ignore
        device_state_steps = list(device_state_steps_)  # type: ignore

        device = device_params[0].device
        if beta1_dict is not None and device not in beta1_dict:
            beta1_dict[device] = beta1.to(device=device, non_blocking=True)
        device_beta1 = beta1_dict[device] if beta1_dict else beta1

        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = list(device_max_exp_avg_sqs_)  # type: ignore
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs)
            else:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)

        # weight decay
        if weight_decay != 0:
            if decoupled_weight_decay:
                torch._foreach_mul_(device_params, 1 - lr * weight_decay)
            else:
                if maximize:
                    torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
                else:
                    device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        # EMA
        torch._foreach_lerp_(device_exp_avgs, device_grads, float(1 - device_beta1))
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        if isinstance(beta2, torch.Tensor):
            scaled_grads = torch._foreach_mul(device_grads, 1 - beta2)
            addval = 1.0
        else:
            scaled_grads = device_grads
            addval = 1 - beta2
        torch._foreach_addcmul_(device_exp_avg_sqs, scaled_grads, device_grads, addval)
        del scaled_grads

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            torch._foreach_neg_(bias_correction2)

            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)
            torch._foreach_sqrt_(bias_correction2)

            step_size = bias_correction1
            bc2_sqrt = bias_correction2

            exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bc2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            if cautious:
                mask = torch._foreach_mul(device_exp_avgs, device_grads)
                mask = [m.gt(0.0).to(e.dtype) for m, e in zip(mask, device_exp_avgs)]
                mean_mask = [m.mean().clamp(min=1e-3) for m in mask]
                mask = [m / mm for m, mm in zip(mask, mean_mask)]
                masked_exp_avg = torch._foreach_mul(device_exp_avgs, mask)
                torch._foreach_addcdiv_(device_params, masked_exp_avg, exp_avg_sq_sqrt)
            else:
                torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bc1 = [1 - beta1 ** _get_value(s) for s in device_state_steps]
            bc2 = [1 - beta2 ** _get_value(s) for s in device_state_steps]
            step_size = _stack_if_compiling([(lr / c1) * -1 for c1 in bc1])
            bc2_sqrt = [c**0.5 for c in bc2]

            exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bc2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)

            if cautious:
                mask = [m.gt(0.0).to(e.dtype) for m, e in zip(device_exp_avgs, device_exp_avgs)]
                mean_mask = [m.mean().clamp(min=1e-3) for m in mask]
                mask = [m / mm for m, mm in zip(mask, mean_mask)]
                masked_exp_avg = torch._foreach_mul(device_exp_avgs, mask)
                torch._foreach_addcdiv_(device_params, masked_exp_avg, exp_avg_sq_sqrt, step_size)
            else:
                torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)


# ---------------- Async foreach kernel (generator; cannot be torch.compile) ----------------
def c_adamw_update_foreach_async(
    X: List[Tensor],  # 权重（原地修改）
    G: List[Tensor],  # 梯度
    M: List[Tensor],  # 一阶动量（原地修改）
    V: List[Tensor],  # 二阶动量（原地修改）
    lr: Tensor,       # 学习率（标量张量）
    beta1: Tensor,    # β1
    beta2: Tensor,    # β2
    weight_decay: Tensor,  # 权重衰减
    step: int,
    epsilon: float,
):
    """
    C-AdamW 的 foreach 异步实现（生成器版，用于 AsyncRuntime）。
    """
    batch_size = len(X)
    assert batch_size == len(G) == len(M) == len(V)

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # M = β1*M + (1-β1)*G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = β2*V + (1-β2)*G*G
    G2 = torch._foreach_mul(G, G)
    G2 = [g.to(dtype=V_dtype) for g in G2]
    torch._foreach_lerp_(V, G2, [1 - beta2] * batch_size)

    # Bias correction
    bc1 = 1 - beta1**step
    bc2 = 1 - beta2**step
    bc2_sqrt = bc2.sqrt()

    # denom = sqrt(V)/sqrt(bc2) + eps
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bc2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # 调整学习率以吸收 bc1
    adj_lr = lr / bc1

    # cautious mask（逐元素，根据 exp_avg 与 grad 的同向性）
    mask = torch._foreach_mul(M, G)
    mask = [m.gt(0.0).to(e.dtype) for m, e in zip(mask, M)]
    mask_mean = [m.mean().clamp(min=1e-3) for m in mask]
    M = torch._foreach_mul(M, mask)

    # 归一化后更新量
    M_div = torch._foreach_div(M, denom)

    # decoupled weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # 权重更新
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_div_(M_div, mask_mean)
    torch._foreach_sub_(X, M_div)

    # 让调度器有机会切换到其他任务
    yield


# ---------------- Optimizer (AdamW with cautious mask & async foreach path) ----------------
class AdamW(Optimizer):
    """
    AdamW（带 cautious mask，可选 foreach 异步路径）
    - foreach=True 时：按参数批处理 + AsyncRuntime 驱动的生成器版 foreach 内核；
    - foreach=False 时：逐参数纯 PyTorch 更新路径。
    """
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        foreach: bool = True,
        fused: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This AdamW variant is for demo; prefer torch.optim.AdamW in production. "
                "Set no_deprecation_warning=True to silence.",
                FutureWarning,
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.init_lr = lr
        self._foreach = foreach
        try:
            self._world_size = dist.get_world_size()
        except Exception:
            self._world_size = 1

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients, use SparseAdam.")
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = (torch.tensor(0.0, ))
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
        return has_complex

    def _get_or_initialize_state(self, param):
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            state["variance"] = torch.zeros_like(param)
        return state

    def _create_tasks(self, param_groups):
        for group in param_groups:
            for params in create_param_batches(group["params"], batch_size=self._world_size):
                params = [p for p in params if p.grad is not None]
                if not params:
                    continue
                grads = [p.grad for p in params]
                states = [self._get_or_initialize_state(p) for p in params]
                m = [s["momentum"] for s in states]
                v = [s["variance"] for s in states]
                lr = torch.tensor(group["lr"])
                beta1 = torch.tensor(group["betas"][0])
                beta2 = torch.tensor(group["betas"][1])
                weight_decay = torch.tensor(group["weight_decay"])
                epsilon = torch.tensor(group["eps"])
                step = torch.tensor(group["step"])
                yield AsyncTask(
                    c_adamw_update_foreach_async(
                        X=pad_batch(params, self._world_size),
                        G=pad_batch(grads, self._world_size),
                        M=pad_batch(m, self._world_size),
                        V=pad_batch(v, self._world_size),
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=epsilon,
                    )
                )

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """执行一次优化步。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._foreach:
            groups = []
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                group["step"] += 1
                groups.append(group)
            tasks = self._create_tasks(groups)
            runtime = AsyncRuntime(chain(tasks), max_concurrent_tasks=3)
            runtime.run()
            return loss

        # 非 foreach 路径（逐参数）
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # decoupled weight decay
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # EMA
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bc1 = 1.0 - beta1 ** state["step"]
                    bc2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bc2) / bc1

                # cautious mask
                if isinstance(grad, DTensor):
                    mask = (exp_avg.full_tensor() * grad.full_tensor() > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    mask = distribute_tensor(mask, device_mesh=grad.device_mesh, placements=grad.placements)
                else:
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                norm_grad = (exp_avg * mask) / denom
                p.add_(norm_grad, alpha=-step_size)
        return loss


# ---------------- Example / quick test ----------------
def test_c_adamw():
    """最小可运行示例：单层线性回归跑 1 步，检查参数发生变化。"""
    import copy
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 配置
    B, Din, Dout = 16, 10, 1
    lr = 1e-2

    # 2) 构造模型与数据
    model = nn.Linear(Din, Dout).to(device)
    x = torch.randn(B, Din, device=device)
    y = torch.randn(B, Dout, device=device)
    loss_fn = nn.MSELoss()

    # 3) 实例化优化器（统一命名为 block）
    block = AdamW(model.parameters(), lr=lr, foreach=True, no_deprecation_warning=True)

    # 4) 前向 + 反传 + step
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    before = copy.deepcopy([p.detach().clone() for p in model.parameters()])
    block.step()
    block.zero_grad(set_to_none=True)
    after = [p.detach().clone() for p in model.parameters()]

    # 5) 打印结构与张量形状
    print(block)
    print("x.shape =", x.shape, "y.shape =", y.shape)

    # 简单检查参数有更新
    for i, (b, a) in enumerate(zip(before, after)):
        diff = (b - a).abs().sum().item()
        print(f"param[{i}] Δ={diff:.6f}")
        assert diff > 0, f"Parameter {i} did not update!"

    print("✅ C-AdamW step OK")

if __name__ == "__main__":
    test_c_adamw()
