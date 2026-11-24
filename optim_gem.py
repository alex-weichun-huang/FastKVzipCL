# ==============================================================================
# Gate training with Gradient Episodic Memory (GEM, Lopez-Paz & Ranzato, 2017).
#
# optim.py mixes all feature folders into one training set. Here each folder is a separate task
# (e.g. the same kind of text scored with qa_hint / retrieval_hint / summary_hint), trained one after
# another. After a task is done, a random subset of its training tokens is kept as episodic memory.
# While training later tasks, the gradient on each memory is computed every step; if the update
# would increase the loss on a memory (negative dot product), the gradient is projected to the
# closest gradient that does not, by solving GEM's small quadratic program.
#
# The gate has one independent module per layer (separate parameters and loss), so the projection
# is done per layer: each layer gets its own constraints.
#
# Output has the same format as optim.py (result_gate/<model>/q<G>_dim16_sink16_<tag>.pt),
# plus per-task test curves under "loss_task" / "acc_task".
#
# Example:
#   python -B optim_gem.py -m Qwen/Qwen3-4B-Instruct-2507 --tag fineweb_10k_gem \
#       --folders fineweb_10k_qa_hint fineweb_10k_retrieval_hint fineweb_10k_summary_hint
# ==============================================================================
import argparse
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig

from load_data import load_data, parse_folders
from optim import Weight, init_from_gate
from utils import acc_fn, loss_fn, lr_schedule, plot


def split_task(inputs, scores, device):
    """Same split as load_data.split_fn (every 100th token is test), test set on device."""
    n = inputs.shape[-2]
    test_mask = (torch.arange(n) + 1) % 100 == 0
    x_train, y_train = inputs[..., ~test_mask, :], scores[..., ~test_mask]
    x_test, y_test = inputs[..., test_mask, :].to(device), scores[..., test_mask].to(device)
    return x_train, x_test, y_train, y_test


def solve_gem_qp(gram, dots, margin=0.0, eps=1e-3):
    """GEM projection for one layer, in dual form as in the reference code (project2cone2):
        primal: min_z 1/2 ||z - g||^2  s.t.  G z >= 0          (G: k memory gradients, g: current)
        dual:   min_v 1/2 v^T (G G^T) v + (G g)^T v  s.t.  v >= margin,   then z = g + G^T v
    A positive margin pushes the update further toward the memories (GEM's "memory strength").

    gram: k x k (G G^T), dots: k (G g). k = number of memories (small), so it is solved exactly by
    enumerating active sets. Returns v (k,), float64 numpy.
    """
    k = len(dots)
    P = gram + eps * np.eye(k)
    # v = u + margin with u >= 0
    q = dots + P @ np.full(k, margin)
    best = None
    for r in range(1, k + 1):
        for S in itertools.combinations(range(k), r):
            S = list(S)
            try:
                vS = np.linalg.solve(P[np.ix_(S, S)], -q[S])
            except np.linalg.LinAlgError:
                continue
            if (vS < -1e-12).any():
                continue
            v = np.zeros(k)
            v[S] = vS
            # KKT: gradient of the dual objective must be >= 0 on inactive coordinates
            grad = P @ v + q
            inactive = [i for i in range(k) if i not in S]
            if all(grad[i] >= -1e-9 for i in inactive):
                obj = 0.5 * v @ P @ v + q @ v
                if best is None or obj < best[0]:
                    best = (obj, v)
    if best is None:  # should not happen for a PSD system; fall back to projected gradient
        v = np.zeros(k)
        lr = 1.0 / (np.linalg.norm(P, 2) + 1e-12)
        for _ in range(500):
            v = np.maximum(0.0, v - lr * (P @ v + q))
        best = (None, v)
    return best[1] + margin


def flat_layer_grads(module):
    return torch.cat([p.grad.reshape(-1) for p in module.parameters()])


def set_layer_grads(module, flat):
    i = 0
    for p in module.parameters():
        n = p.numel()
        p.grad.copy_(flat[i : i + n].view_as(p))
        i += n


def layer_losses(modules, x, y, loss_type):
    """List of per-layer losses on one batch (layer x seq x dim, layer x head x seq)."""
    return [loss_fn(modules[l](x[l]), y[l], type=loss_type) for l in range(len(modules))]


def sample_batch(x, y, n, device):
    idx = torch.randint(0, x.shape[-2], (min(n, x.shape[-2]),))
    return x[..., idx, :].to(device), y[..., idx].to(device)


def evaluate(modules, tasks, loss_type):
    """Per-task, per-layer test loss and accuracy: [n_task][n_layer]."""
    losses, accs = [], []
    with torch.no_grad():
        for t in tasks:
            lt, at = [], []
            for l in range(len(modules)):
                pred = modules[l](t["x_test"][l])
                lt.append(loss_fn(pred, t["y_test"][l], type=loss_type).item())
                at.append(acc_fn(pred, t["y_test"][l]).item())
            losses.append(lt)
            accs.append(at)
    return losses, accs


def optimize_gem(args, tasks, device, test_freq=50, batch_size=1000):
    input_dim = tasks[0]["x_train"].shape[-1]
    n_layer, nhead, _ = tasks[0]["y_train"].shape
    dtype = tasks[0]["x_train"].dtype

    modules = nn.ModuleList(
        [
            Weight(l, input_dim, args.dim, nhead, args.ngroup, dtype, sink=args.sink).to(device)
            for l in range(n_layer)
        ]
    )
    print(modules[0], "x", len(modules))
    if args.init_gate:
        init_from_gate(modules, args.model, args.init_gate)

    memories = []  # (x, y) on CPU, one per finished task
    hist = {"loss": [], "acc": [], "loss_test": [], "acc_test": [], "loss_task": [], "acc_task": []}
    s = time.time()
    for t, task in enumerate(tasks):
        # Fresh optimizer and lr schedule per task: momentum from the previous task's gradients
        # would otherwise carry updates that were never checked against the memories.
        optimizer = optim.SGD(modules.parameters(), lr=args.lr, momentum=args.momentum)
        x_train, y_train = task["x_train"], task["y_train"]
        n_samples = x_train.shape[-2]
        n_split = max(1, n_samples // batch_size)
        n_proj, n_checks = 0, 0
        print(
            f"\n=== Task {t}: {task['name']} ({n_samples} train tokens, {len(memories)} memories, "
            f"{args.steps_per_task} steps) ==="
        )

        for i in range(args.steps_per_task):
            split = i % n_split
            if split == 0:
                indices = torch.randperm(n_samples)
            idx = indices[split * batch_size : (split + 1) * batch_size]
            x = x_train[..., idx, :].to(device)
            y = y_train[..., idx].to(device)

            # Memory gradients, one per previous task: mem_grads[k][l] = flat grad of layer l
            mem_grads = []
            for mx, my in memories:
                bx, by = sample_batch(mx, my, args.mem_batch, device)
                optimizer.zero_grad(set_to_none=False)
                sum(layer_losses(modules, bx, by, args.loss)).backward()
                mem_grads.append([flat_layer_grads(m).clone() for m in modules])

            # Current task gradient
            optimizer.zero_grad(set_to_none=False)
            cur = layer_losses(modules, x, y, args.loss)
            sum(cur).backward()

            if mem_grads:
                k = len(mem_grads)
                g = torch.stack([flat_layer_grads(m) for m in modules])  # layer x p
                G = torch.stack([torch.stack(mg) for mg in mem_grads], dim=1)  # layer x k x p
                dots = torch.einsum("lkp,lp->lk", G.float(), g.float()).double().cpu().numpy()
                n_checks += n_layer
                bad = (dots < 0).any(1)
                if bad.any():
                    gram = torch.einsum("lkp,ljp->lkj", G.float(), G.float()).double().cpu().numpy()
                    for l in np.nonzero(bad)[0]:
                        v = solve_gem_qp(gram[l], dots[l], margin=args.margin)
                        v = torch.as_tensor(v, device=device, dtype=torch.float32)
                        new = g[l].float() + (v[:, None] * G[l].float()).sum(0)
                        set_layer_grads(modules[l], new.to(g.dtype))
                        n_proj += 1

            alpha = lr_schedule(i, args.steps_per_task)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * alpha
            optimizer.step()

            if i % test_freq == 0:
                with torch.no_grad():
                    acc_train = [acc_fn(modules[l](x[l]), y[l]).item() for l in range(n_layer)]
                lt, at = evaluate(modules, tasks, args.loss)
                hist["loss"].append([c.item() for c in cur])
                hist["acc"].append(acc_train)
                hist["loss_test"].append(np.mean(lt[: t + 1], 0))  # tasks seen so far
                hist["acc_test"].append(np.mean(at[: t + 1], 0))
                hist["loss_task"].append(np.mean(lt, 1))
                hist["acc_task"].append(np.mean(at, 1))
                per_task = " ".join(
                    f"{tk['name']}={np.mean(lt[j]):.3f}/{np.mean(at[j])*100:.1f}%"
                    for j, tk in enumerate(tasks)
                )
                proj = f" proj {n_proj}/{n_checks}" if memories else ""
                print(
                    f"[{t}:{i:5d}] loss {np.mean(hist['loss'][-1]):.3f} acc {np.mean(acc_train)*100:.1f}%"
                    f" | test {per_task}{proj} (time: {time.time()-s:.0f}s)"
                )
                n_proj, n_checks = 0, 0

        # Episodic memory for this task: random training tokens, kept on CPU
        if t < len(tasks) - 1:
            idx = torch.randperm(n_samples)[: args.mem_size]
            memories.append((x_train[..., idx, :].clone(), y_train[..., idx].clone()))
            print(f"memory {task['name']}: {len(idx)} tokens")

    hist = {k: np.array(v) for k, v in hist.items()}
    return modules, hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dim", "-d", type=int, default=16)
    parser.add_argument("--ngroup", "-q", type=int, default=-1, help="query group size")
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--loss", type=str, default="cnt")
    parser.add_argument("--lr", type=float, default=2e-1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="gem")
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=["fineweb_10k_qa_hint", "fineweb_10k_retrieval_hint", "fineweb_10k_summary_hint"],
        help="one task per entry, in training order (name or name:start-end); join folders with + "
        "to train them as one task, e.g. fineweb_10k:0-29+fineweb_10k_cat:0-5",
    )
    parser.add_argument(
        "--steps_per_task",
        type=int,
        default=0,
        help="SGD steps per task; default 5000 / #tasks (same total steps as optim.py)",
    )
    parser.add_argument("--mem_size", type=int, default=20000, help="episodic memory tokens per task")
    parser.add_argument("--mem_batch", type=int, default=1000, help="memory tokens per gradient check")
    parser.add_argument("--margin", type=float, default=0.0, help="GEM margin (memory strength)")
    parser.add_argument(
        "--max_tokens", type=int, default=0, help="if > 0, randomly keep at most this many tokens per task"
    )
    parser.add_argument(
        "--init_gate",
        type=str,
        default="",
        help="fine-tune from this gate instead of random init: 'fastkvzip' (released gate) "
        "or a tag in ./result_gate/<model>/ (e.g. q4_dim16_sink16_local)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.ngroup == -1:
        config = AutoConfig.from_pretrained(args.model)
        args.ngroup = config.num_attention_heads // config.num_key_value_heads
    if args.steps_per_task <= 0:
        args.steps_per_task = 5000 // len(args.folders)

    tag = f"q{args.ngroup}_dim{args.dim}"
    tag += f"_sink{args.sink}" if args.sink >= 0 else ""
    tag += f"_{args.tag}" if args.tag else ""
    tag += f"_{args.loss}" if args.loss != "cnt" else ""
    tag += f"_lr{args.lr}" if args.lr != 2e-1 else ""
    print(f"\nQuery group size: {args.ngroup} ({tag})\n{vars(args)}")
    torch.manual_seed(args.seed)

    name = args.model.split("/")[-1].lower()
    tasks = []
    for spec in args.folders:
        inputs, scores = load_data(path=f"./data/{name}", input_name="hidden", folders=parse_folders(spec.split("+")))
        if 0 < args.max_tokens < inputs.shape[-2]:
            keep = torch.randperm(inputs.shape[-2])[: args.max_tokens].sort().values
            inputs, scores = inputs[..., keep, :], scores[..., keep]
        x_train, x_test, y_train, y_test = split_task(inputs, scores, args.device)
        del inputs, scores
        print(f"task {spec}: train {tuple(x_train.shape)}, test {tuple(x_test.shape)}")
        tasks.append(
            {"name": "+".join(f.split(":")[0] for f in spec.split("+")), "x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
        )

    modules, hist = optimize_gem(args, tasks, args.device)

    path = f"./result_gate/{name}"
    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "module": [m.state_dict() for m in modules],
            "loss": hist["loss_test"],
            "acc": hist["acc_test"],
            "loss_train": hist["loss"],
            "acc_train": hist["acc"],
            "loss_task": hist["loss_task"],  # eval step x task (mean over layers)
            "acc_task": hist["acc_task"],
            "tasks": [t["name"] for t in tasks],
            "args": vars(args),
        },
        f"{path}/{tag}.pt",
    )
    plot(hist["loss"], hist["loss_test"], name=f"loss_{tag}", log=True, path=path)
    plot(hist["acc"], hist["acc_test"], name=f"acc_{tag}", path=path)
    print(f"Saved {path}/{tag}.pt\nFinished!")
