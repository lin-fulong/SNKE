from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .SNKE_hparams import SNKEHyperParams
from .gnn import GNN


def optimize_graph_delta(
    gnn_model: GNN,
    hparams: SNKEHyperParams,
    graph_input,
    node_indices: dict,
    init_rel_emb: torch.Tensor,
    request: dict,
    loss_fn: Callable[
        [torch.Tensor],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
        ],
    ],
) -> Tuple[torch.Tensor, torch.Tensor]:
    subject_id = node_indices[request["subject"]]
    print(
        f"[SNKE][gnn_training] start | case_id={request['case_id']} "
        f"| subject={request['subject']} | subject_id={subject_id} "
        f"| nodes={graph_input.num_nodes()} | edges={graph_input.num_edges()}"
    )
    scale_factor = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    target_init = None

    opt = torch.optim.AdamW(
        [
            {
                "params": gnn_model.parameters(),
                "lr": hparams.gnn_lr,
                "weight_decay": hparams.gnn_weight_decay,
            },
            {"params": [scale_factor], "lr": hparams.v_lr},
        ]
    )

    gnn_model.train()

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        outputs = gnn_model(
            graph_input, graph_input.ndata["feat"].float(), init_rel_emb.float()
        ).to("cuda")
        subject_feature = F.normalize(outputs[subject_id], p=2, dim=-1)
        target_delta_feature = scale_factor * subject_feature * hparams.factor_g

        if target_init is not None:
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if target_delta_feature.norm() > max_norm:
                with torch.no_grad():
                    target_delta_feature[...] = (
                        target_delta_feature * max_norm / target_delta_feature.norm()
                    )

        (
            loss,
            nll_loss_each,
            nll_loss,
            kl_loss,
            weight_decay,
            target_init,
        ) = loss_fn(target_delta_feature)

        print(
            f"[SNKE][gnn_training] iter={it + 1}/{hparams.v_num_grad_steps} "
            f"| loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + "
            f"{np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"| avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()} "
            f"| scale_factor={scale_factor.item():.4f}"
        )

        if loss < 5e-2 or it == hparams.v_num_grad_steps - 1:
            break

        loss.backward()
        opt.step()

    gnn_model.eval()
    with torch.no_grad():
        outputs = gnn_model(
            graph_input, graph_input.ndata["feat"].float(), init_rel_emb.float()
        ).to("cuda")
        subject_feature = F.normalize(outputs[subject_id], p=2, dim=-1)
        target_delta_feature = scale_factor * subject_feature * hparams.factor_g

        if target_init is not None:
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if target_delta_feature.norm() > max_norm:
                target_delta_feature = (
                    target_delta_feature * max_norm / target_delta_feature.norm()
                )

    print(
        f"[SNKE][gnn_training] finished | case_id={request['case_id']} "
        f"| target_init_norm={target_init.norm().item() if target_init is not None else 0.0:.4f} "
        f"| target_delta_norm={target_delta_feature.norm().item():.4f}"
    )
    return target_init, target_delta_feature
