import torch
import tqdm
import re
import utils
from param import param
from pathlib import Path

class MergingMethod:

    @utils.args_inspector
    def __init__(
        self, 
        models_to_merge, 
        models_name,
    ):
        self.models_name = {n:i for i,n in enumerate(models_name)}
        # dict(zip(models_name, range(0, N)))
        self.models_to_merge = models_to_merge

    def get_model(self, model_name):
        return self.models_to_merge[self.models_name[model_name]]

    @utils.args_inspector
    def wudi_merge(
        self,
        base_model: param,
        models_to_merge: list,
        scaling: float = 1.0,

        # WUDI optimizer knobs
        iter_num: int = 200,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        device: str = "cuda",   # "cuda" strongly recommended; "cpu" will be very slow

        # which params to run WUDI on
        include: list = None,   # list of regex; if None, defaults to Llama proj matrices
        exclude: list = None,   # list of regex

        # fallback for keys not WUDI-optimized
        fallback: str = "task_arithmetic",  # "task_arithmetic" or "zero"
        eps: float = 1e-12,
        verbose: bool = True,
    ):
        """
        WUDI-style merge:
          - Compute task vectors tv_i = ft_i - base
          - For selected 2D keys, optimize a merged task vector per key via redundancy loss
          - For other keys, fallback to mean(tv_i) (Task Arithmetic)
          - Return base + scaling * merged_task_vector
        """

        if include is None:
            # Llama projection matrices only (safe & close to typical "merge only the big 2D weights")
            include = [
                r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
                r".*mlp\.(gate_proj|up_proj|down_proj)\.weight$",
            ]

        if exclude is None:
            exclude = []

        def _match_any(patterns, name: str) -> bool:
            return any(re.match(p, name) for p in patterns)

        def _use_wudi_for_key(name: str, t: torch.Tensor) -> bool:
            if t.ndim != 2:
                return False
            if include and not _match_any(include, name):
                return False
            if exclude and _match_any(exclude, name):
                return False
            return True

        # task vectors: tv_i = ft_i - base
        tvs = [m - base_model for m in models_to_merge]

        # choose device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dev = torch.device(device)

        merged_tv = {}

        # iterate over base keys (keeps structure stable)
        base_keys = list(base_model.keys())

        for k in tqdm.tqdm(base_keys, desc="WUDI merge (per-key)"):
            if k not in tvs[0]:
                continue

            # gather per-task tensors (keep dtype consistent)
            vecs_cpu = [tv[k] for tv in tvs]
            t0 = vecs_cpu[0]

            # keys not shared / shape mismatch -> fallback
            if any(v.shape != t0.shape for v in vecs_cpu):
                print(f'{k} is not optimized via WUDI due to Shape Mismatch')
                if fallback == "zero":
                    merged_tv[k] = torch.zeros_like(t0)
                else:
                    merged_tv[k] = torch.stack(vecs_cpu, dim=0).sum(dim=0)
                continue

            if _use_wudi_for_key(k, t0):
                # stack on device: (n_task, out, in)
                vectors = torch.stack([v.to(dev) for v in vecs_cpu], dim=0)

                # init like the ViT script: sum(vectors)
                merging_vector = torch.nn.Parameter(vectors.sum(dim=0))

                opt = torch.optim.Adam([merging_vector], lr=lr, weight_decay=weight_decay)

                # l2 norms per task vector
                l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)) + eps

                for _ in range(iter_num):
                    # disturbing_vectors: (n_task, out, in)
                    disturbing = merging_vector.unsqueeze(0) - vectors

                    # inner_product: (n_task, out, out)
                    inner = torch.matmul(disturbing, vectors.transpose(1, 2))

                    loss = torch.sum((inner * inner) / l2_norms.view(-1, 1, 1))

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                merged_tv[k] = merging_vector.detach().to("cpu")

                if verbose:
                    pass  # keep quiet unless you want per-key logging

            else:
                # fallback for embeddings / norms / lm_head / 1D etc.
                print(f'{k} is not optimized with WUDI')
                if fallback == "zero":
                    merged_tv[k] = torch.zeros_like(t0)
                else:
                    merged_tv[k] = torch.stack(vecs_cpu, dim=0).sum(dim=0)

        # scale the merged task vector and apply to base
        merged_task_param = param({k: (scaling * v) for k, v in merged_tv.items()})
        merged_param = base_model + merged_task_param
        return merged_param
