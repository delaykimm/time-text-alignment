import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import argparse
import time
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from models.TimeCMA import Dual
from utils.metrics import MSE, MAE, metric
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

import faulthandler
faulthandler.enable()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

# -------------------------------
# Args
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--root_path", type=str, default="./dataset")
    p.add_argument("--data_path", type=str, default="ETTm1")
    p.add_argument("--channel", type=int, default=32)
    p.add_argument("--num_nodes", type=int, default=7)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--dropout_n", type=float, default=0.2)
    p.add_argument("--d_llm", type=int, default=768)
    p.add_argument("--e_layer", type=int, default=1)
    p.add_argument("--d_layer", type=int, default=1)
    p.add_argument("--head", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--es_patience", type=int, default=50)
    p.add_argument("--save", type=str, default="./logs/" + time.strftime("%Y-%m-%d-%H:%M:%S") + "-")
    # 정렬 관련
    p.add_argument("--align_weight", type=float, default=0.1, help="weight for cosine alignment loss")
    p.add_argument("--common_dim", type=int, default=64, help="shared space dimension D")
    p.add_argument("--print_every", type=int, default=50)
    
    return p.parse_args()

# -------------------------------
# Utils
# -------------------------------
def seed_it(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def to_tensor(x):
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)

def pad_prompt_embeddings(fps):
    """
    fps_list: 각 원소가 [T_i, d_llm, N] 또는 [K, T_i, d_llm, N] (K개 뷰)의 텐서/넘파이
    return: [B, T_max, d_llm, N]
    """
    # fps: list of [T_i, d_llm, N]  (혹은 [K, T_i, d_llm, N]이면 K 평균)
    _norm = []
    for t in fps:
        t = torch.as_tensor(t)
        if t.dim() == 4:
            # [K, T_i, d_llm, N] -> K 평균 → [T_i, d_llm, N]
            t = t.mean(dim=0)
        elif t.dim() != 3:
            raise ValueError(f"full_prompt_emb must be 3D [T_i, d_llm, N], got {tuple(t.shape)}")
        _norm.append(t)

    T_max = max(t.size(0) for t in _norm)
    d_llm = _norm[0].size(1)
    N     = _norm[0].size(2)

    padded = []
    for t in _norm:
        Ti = t.size(0)
        if Ti < T_max:
            last = t[-1:].expand(T_max - Ti, d_llm, N)  # 마지막 토큰 복제 패딩
            t = torch.cat([t, last], dim=0)
        padded.append(t)
    return torch.stack(padded, dim=0)  # [B, T_max, d_llm, N]

def _norm_last_emb(le):
    # le: last_emb sample → [d_llm, N]로 강제
    le = torch.as_tensor(le)
    if le.is_sparse:
        le = le.to_dense()
    # 과차원 있다면 앞축 평균
    while le.dim() > 2:
        le = le.mean(dim=0)
    if le.dim() == 1:
        # [d_llm]가 들어오면 N=1로 가정(권장하진 않음; 가능하면 Dataset에서 고쳐주세요)
        le = le.unsqueeze(-1)
    if le.dim() != 2:
        raise ValueError(f"last_emb must be 2D [d_llm, N], got {tuple(le.shape)}")
    return le  # [d_llm, N]

def _norm_anchor(anc):
    # anc: anchor_avg sample → [N, d_llm]로 강제
    anc = torch.as_tensor(anc)
    if anc.dim() == 4:
        # [K, N, d_llm] 앞에 뭔가 더 있으면 평균
        anc = anc.mean(dim=0)
    if anc.dim() == 3:
        # [K, N, d_llm] -> K 평균
        anc = anc.mean(dim=0)
    elif anc.dim() == 2:
        pass
    else:
        raise ValueError(f"anchor_avg bad shape: {tuple(anc.shape)} (need [N,d_llm] or [K,N,d_llm])")

    # 혹시 [d_llm, N] 순서면 전치
    if anc.size(0) != anc.size(-1) and anc.size(0) != anc.size(1):
        # 모양이 확실치 않으면 그대로 두되, 필요시 Dataset에서 고치세요.
        pass
    # [d_llm, N] → [N, d_llm]
    if anc.size(0) != anc.size(1) and anc.size(0) !=  anc.size(0):
        pass
    if anc.size(0) != anc.size(1) and anc.size(0) == anc.size(-1):
        anc = anc.permute(1, 0)  # [d_llm, N] → [N, d_llm]
    if anc.size(0) < anc.size(1):  # [d_llm, N] 형태일 때
        anc = anc.permute(1, 0)    # → [N, d_llm]
    return anc  # [N, d_llm]

def custom_collate(batch):
    """
    batch item: (x, y, x_mark, y_mark, last_emb, full_prompt, anchor_avg)
    x: [L, N], last_emb: [d_llm, N] (변형될 수 있으므로 정규화), full_prompt: [T_i, d_llm, N] or [K, T_i, d_llm, N]
    anchor_avg: [N, d_llm] or [K, N, d_llm]
    """
    xs, ys, xms, yms, lasts, fps, anchors = zip(*batch)

    x       = torch.stack([torch.as_tensor(t) for t in xs], dim=0)
    y       = torch.stack([torch.as_tensor(t) for t in ys], dim=0)
    x_mark  = torch.stack([torch.as_tensor(t) for t in xms], dim=0)
    y_mark  = torch.stack([torch.as_tensor(t) for t in yms], dim=0)

    # last_emb: 각 샘플을 [d_llm, N]로 통일 후 [B, d_llm, N]로 stack
    last_list = [_norm_last_emb(le) for le in lasts]
    # 여기서 모든 샘플이 같은 (d_llm, N)인지 확인
    d_llm, N = last_list[0].shape
    for t in last_list:
        if t.shape != (d_llm, N):
            raise ValueError(f"last_emb shapes differ inside batch: {tuple(t.shape)} vs {(d_llm, N)}")
    last = torch.stack(last_list, dim=0)  # [B, d_llm, N]

    # full_prompt: [B, T_max, d_llm, N]
    full_prompt = pad_prompt_embeddings(fps)

    # anchor_avg: 각 샘플을 [N, d_llm]로 통일 후 [B, N, d_llm]로 stack
    anc_list = [_norm_anchor(a) for a in anchors]
    N2, d2 = anc_list[0].shape
    for t in anc_list:
        if t.shape != (N2, d2):
            raise ValueError(f"anchor_avg shapes differ inside batch: {tuple(t.shape)} vs {(N2, d2)}")
    anchor = torch.stack(anc_list, dim=0)  # [B, N, d_llm]

    return x, y, x_mark, y_mark, last, full_prompt, anchor

def load_data(args):
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(flag='train', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path)
    val_set = data_class(flag='val', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path)
    test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path)

    scaler = train_set.scaler

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers, collate_fn = custom_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers, collate_fn = custom_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers, collate_fn = custom_collate)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler
 
def get_first_batch_embeddings(loader, model, device):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        x, _y, x_mark, _y_mark, last_emb, full_prompt_emb, _anchor = batch
        x = x.to(device); x_mark = x_mark.to(device)
        
        last_emb = last_emb.to(device)
        full_prompt_emb = full_prompt_emb.to(device)
        
        enc, pr, cr = model.get_embeddings(x, x_mark, last_emb, full_prompt_emb)
        return enc, pr, cr, full_prompt_emb 
    
# ----------------------------
# Similarity / Heatmap helpers
# ------------------------------- 
@torch.no_grad()
def visualize_common_space_all_per_channel(
    enc_out, cross_out, prompt_last, full_prompt_emb,
    ts_head, txt_head,
    *,
    method="tsne",
    token_sample_per_channel=50,
    color_map=None,            # {"ENC":"#..", "CROSS":"#..", "LAST":"#..", "TOK":"#.."}
    save_path="common_space_all.png"
):
    """
    - ENC / CROSS / LAST: [B,N,D] → (B*N,D)로 모두 찍음 (배치×채널 개별 포인트)
    - TOK: 채널별로 token_sample_per_channel 만큼 샘플링해서 같은 채널 마커로 찍음
    - 같은 채널이면 같은 마커, 종류(ENC/CROSS/LAST/TOK)는 색만 다르게
    """
    device = next(ts_head.parameters()).device

    # ---------- shape normalize ----------
    def _enc_bnC(x):  # [B,C,N] -> [B,N,C]
        if x.dim() == 2: x = x.unsqueeze(0)
        return x.permute(0, 2, 1).contiguous()

    def _last_bnE(x):  # [B,d_llm,N] -> [B,N,d_llm]
        if x.dim() == 2: x = x.unsqueeze(0)
        return x.permute(0, 2, 1).contiguous()

    enc_bnC   = _enc_bnC(enc_out).to(device)                       # [B,N,C]
    cross_bnC = (cross_out if cross_out.dim()==3 else cross_out.unsqueeze(0)).to(device)  # [B,N,C]
    last_bnE  = _last_bnE(prompt_last).to(device)                   # [B,N,E]

    B, N, C = enc_bnC.shape
    D = ts_head[-1].out_features if hasattr(ts_head[-1], "out_features") else ts_head(enc_bnC[:1]).shape[-1]

    # ---------- project to common space ----------
    enc_proj   = F.normalize(ts_head(enc_bnC),   dim=-1)   # [B,N,D]
    cross_proj = F.normalize(ts_head(cross_bnC), dim=-1)   # [B,N,D]
    last_proj  = F.normalize(txt_head(last_bnE), dim=-1)   # [B,N,D]

    # ---------- full prompt tokens ----------
    if full_prompt_emb.dim() == 3:   # [T,E,N] -> [1,T,E,N]
        full_prompt_emb = full_prompt_emb.unsqueeze(0)
    B_fp, T, E, N_fp = full_prompt_emb.shape
    assert N_fp == N, f"N mismatch: full_prompt N={N_fp} vs N={N}"

    # [B,N,T,E]
    fp_bnte = full_prompt_emb.permute(0, 3, 1, 2).contiguous()

    tok_list = []
    tok_ch_list = []
    for b in range(B_fp):
        for n in range(N):
            tok_e = fp_bnte[b, n]  # [T,E]
            k = min(T, token_sample_per_channel)
            if k <= 0: 
                continue
            idx = torch.randperm(T, device=device)[:k]
            sel = tok_e.index_select(0, idx)  # [k,E]
            # [k,E] -> [k,D] (txt_head는 [B,N,E] 기대라서 배치축 임시 추가)
            sel_d = F.normalize(txt_head(sel.unsqueeze(0)), dim=-1).squeeze(0)  # [k,D]
            tok_list.append(sel_d)
            tok_ch_list.append(torch.full((k,), n, device=device, dtype=torch.long))
    tok_proj = torch.cat(tok_list, dim=0) if len(tok_list) else torch.empty(0, D, device=device)
    tok_ch   = torch.cat(tok_ch_list, dim=0) if len(tok_ch_list) else torch.empty(0, dtype=torch.long, device=device)

    # ---------- flatten (B,N,D) -> (B*N,D) & channel index ----------
    def _flatten_with_channel(z):   # z: [B,N,D]
        zf = z.reshape(-1, z.size(-1))  # [B*N,D]
        ch = torch.arange(N, device=z.device).repeat(B)    # [B*N] : 0..N-1, 0..N-1, ...
        return zf, ch

    enc_flat,   enc_ch   = _flatten_with_channel(enc_proj)
    cross_flat, cross_ch = _flatten_with_channel(cross_proj)
    last_flat,  last_ch  = _flatten_with_channel(last_proj)
    # tok_proj, tok_ch already prepared

    arrays = [enc_flat.cpu().numpy(), cross_flat.cpu().numpy(), last_flat.cpu().numpy(), tok_proj.cpu().numpy()]
    names  = ["ENC", "CROSS", "LAST", "TOK"]
    chs    = [enc_ch.cpu().numpy(),   cross_ch.cpu().numpy(),   last_ch.cpu().numpy(),   tok_ch.cpu().numpy()]

    # concat for joint 2D projection
    X = np.concatenate(arrays, axis=0)  # [(sum K), D]
    if X.shape[0] < 3:
        print("Not enough points to visualize.")
        return

    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        X2 = reducer.fit_transform(X)
    else:
        # TSNE perplexity는 데이터 수에 맞게 안전하게
        perp = max(2, min(30, X.shape[0] // 3))
        reducer = TSNE(n_components=2, init="pca", perplexity=perp)
        X2 = reducer.fit_transform(X)

    sizes = [a.shape[0] for a in arrays]
    splits = np.cumsum(sizes)[:-1]
    Xparts = np.split(X2, splits, axis=0)
    CHparts = np.split(np.concatenate(chs, axis=0), splits, axis=0)

    # ---------- plot ----------
    plt.figure(figsize=(9, 7))

    # 색상: 종류별(ENC/CROSS/LAST/TOK), 마커: 채널별
    default_colors = {"ENC": "#1f77b4", "CROSS": "#2ca02c", "LAST": "#ff7f0e", "TOK": "#d62728"}
    if color_map is not None:
        default_colors.update(color_map)

    # 채널 마커 세트 (N=7 가정, 더 많으면 반복)
    channel_markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
    # 범례 표시를 위해 채널별 첫 호출 여부 관리
    channel_legend_done = [False] * N
    type_legend_done = {k: False for k in default_colors.keys()}

    for name, pts, ch_idx in zip(names, Xparts, CHparts):
        color = default_colors.get(name, None)
        for n in range(N):
            mask = (ch_idx == n)
            if not np.any(mask):
                continue
            mk = channel_markers[n % len(channel_markers)]
            # 첫 등장엔 채널 범례 + 타입 범례를 같이 표기
            label = None
            if not channel_legend_done[n] and not type_legend_done.get(name, False):
                label = f"{name} (ch{n})"
                channel_legend_done[n] = True
                type_legend_done[name] = True
            elif not channel_legend_done[n]:
                label = f"ch{n}"
                channel_legend_done[n] = True
            elif not type_legend_done.get(name, False):
                label = f"{name}"
                type_legend_done[name] = True

            plt.scatter(pts[mask, 0], pts[mask, 1],
                        s=24, alpha=0.85, marker=mk, color=color,
                        label=label)

    plt.title(f"Common-space (per-channel markers, {method.upper()})")
    plt.tight_layout()
    plt.legend(ncol=2, fontsize=9)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved common-space plot to {save_path}")  
 
def compute_cosine_similarity(prompt_last: torch.Tensor,
                              cross_out: torch.Tensor,
                              projector: nn.Module = None) -> torch.Tensor:
    """
    Compute cosine similarity between prompt_last and cross_out embeddings per channel.

    Args:
        prompt_last (Tensor): [d_llm, N] or [B, d_llm, N]
        cross_out (Tensor): [N, C] or [B, N, C]
        projector (nn.Module, optional): Optional linear layer for dimension alignment.

    Returns:
        Tensor: Cosine similarity per channel: [N] or [B, N]
    """
    if prompt_last.dim() == 2:
        prompt_last = prompt_last.unsqueeze(0)  # [1, d_llm, N]
    if cross_out.dim() == 2:
        cross_out = cross_out.unsqueeze(0)      # [1, N, C]

    B, d_llm, N = prompt_last.shape
    _, N2, C = cross_out.shape
    assert N == N2, f"Mismatch in channels: {N} vs {N2}"

    prompt_last = prompt_last.permute(0, 2, 1)  # [B, N, d_llm]

    # Optional projection if needed
    if d_llm != C:
        if projector is None:
            projector = nn.Linear(d_llm, C).to(prompt_last.device)
        prompt_last = projector(prompt_last)

    sim = F.cosine_similarity(prompt_last, cross_out, dim=-1)  # [B, N]
    return sim.squeeze(0) if sim.size(0) == 1 else sim

def plot_similarity_distributions(pre_sim, post_sim, save_path='sim_distribution.png'):
    pre_sim = pre_sim.detach().cpu().numpy().flatten()
    post_sim = post_sim.detach().cpu().numpy().flatten()

    plt.figure(figsize=(8,6))
    sns.histplot(pre_sim, color='blue', label='Pre-training', kde=True, stat="density", bins=20, alpha=0.6)
    sns.histplot(post_sim, color='red', label='Post-training', kde=True, stat="density", bins=20, alpha=0.6)
    plt.title("Cosine Similarity Distribution\n(prompt_last vs cross_out)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved similarity distribution to {save_path}")

def compute_pairwise_distance_mse(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute the MSE between pairwise distance matrices of X and Y.

    Args:
        X: [N, D1]
        Y: [N, D2]

    Returns:
        float: Mean squared error between pairwise distances
    """
    # Compute pairwise distances
    D1 = torch.cdist(X, X, p=2)  # [N, N]
    D2 = torch.cdist(Y, Y, p=2)  # [N, N]

    # Normalize (optional): min-max to [0,1] range for stability
    D1 = (D1 - D1.min()) / (D1.max() - D1.min() + 1e-8)
    D2 = (D2 - D2.min()) / (D2.max() - D2.min() + 1e-8)

    # MSE between distance matrices
    return F.mse_loss(D1, D2).item()

# def plot_pairwise_distance_heatmaps(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, save_path='pairwise_distance_heatmap.png'):
#     """
#     두 임베딩 X, Y의 pairwise distance matrix를 heatmap으로 시각화합니다.

#     Args:
#         X: [N, D1] torch.Tensor
#         Y: [N, D2] torch.Tensor
#     """
    
#     # 1. Pairwise distance 계산
#     D1 = torch.cdist(X, X, p=2).cpu().detach().numpy()  # [N, N] - 모든 임베딩 간의 PAIRWISE 거리 계산
#     D2 = torch.cdist(Y, Y, p=2).cpu().detach().numpy()  # [N, N]
#     D3 = torch.cdist(Z, Z, p=2).cpu().detach().numpy()  # [N, N]

#     D2 = D2[0]    # 첫번째 샘플
#     D3 = D3[0]    # 첫번째 샘플

#     # 2. 시각화
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5))

#     sns.heatmap(D1, ax=axs[0], cmap='viridis')
#     axs[0].set_title("Prompt Token Embedding Distance")

#     sns.heatmap(D2, ax=axs[1], cmap='viridis')
#     axs[1].set_title("Prompt_last Embedding Distance")

#     sns.heatmap(D3, ax=axs[2], cmap='viridis')
#     axs[2].set_title("Cross Embedding Distance")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     print(f"✅ Saved heatmap to {save_path}")

def plot_pairwise_distance_heatmaps(
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    save_path: str = 'pairwise_distance_heatmap.png',
    reduce_batch: str = 'first'  # 'first' or 'mean'
):
    """
    X, Y, Z가 [N, D] 또는 [B, N, D]일 때 pairwise distance heatmap을 그립니다.
    reduce_batch: 배치 차원이 있을 때 'first'는 첫 배치 사용, 'mean'은 배치 평균 사용.
    """

    def ensure_2d(t: torch.Tensor, how: str = 'first') -> torch.Tensor:
        # t: [N,D] 또는 [B,N,D]
        if t.dim() == 2:
            return t
        elif t.dim() == 3:
            if how == 'first':
                return t[0]           # [N, D]
            elif how == 'mean':
                return t.mean(dim=0)  # [N, D]
            else:
                raise ValueError(f"Unknown reduce_batch mode: {how}")
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(t.shape)}")

    # 0) 입력을 2D로 통일
    X2 = ensure_2d(X, reduce_batch)
    Y2 = ensure_2d(Y, reduce_batch)
    Z2 = ensure_2d(Z, reduce_batch)

    # 1) Pairwise distance 계산 (전부 [N, N]이 되게)
    D1 = torch.cdist(X2, X2, p=2).cpu().detach().numpy()  # [N, N]
    D2 = torch.cdist(Y2, Y2, p=2).cpu().detach().numpy()  # [N, N]
    D3 = torch.cdist(Z2, Z2, p=2).cpu().detach().numpy()  # [N, N]

    # 2) 시각화
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(D1, ax=axs[0], cmap='viridis')
    axs[0].set_title("Prompt Token Embedding Distance")

    sns.heatmap(D2, ax=axs[1], cmap='viridis')
    axs[1].set_title("Prompt_last Embedding Distance")

    sns.heatmap(D3, ax=axs[2], cmap='viridis')
    axs[2].set_title("Cross Embedding Distance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ Saved heatmap to {save_path}")

# -------------------------------
# Trainer with shared-space heads
# -------------------------------
class trainer:
    def __init__(self, scaler, channel, num_nodes, seq_len, pred_len, dropout_n,
                 d_llm, e_layer, d_layer, head, lrate, wdecay, device, epochs, common_dim):
        self.model = Dual(device=device, channel=channel, num_nodes=num_nodes,
                          seq_len=seq_len, pred_len=pred_len, dropout_n=dropout_n,
                          d_llm=d_llm, e_layer=e_layer, d_layer=d_layer, head=head, visualize=False)
        print("The number of trainable parameters:", self.model.count_trainable_params())
        print("The number of parameters:", self.model.param_num())

        # shared-space heads
        self.ts_head = nn.Sequential(
            nn.Linear(channel, 128), nn.ReLU(),
            nn.Linear(128, common_dim)
        ).to(device)

        self.txt_head = nn.Sequential(
            nn.Linear(d_llm, 128), nn.ReLU(),
            nn.Linear(128, common_dim)
        ).to(device)

        base_lr = lrate
        txt_lr  = lrate * 0.1   # 텍스트 헤드는 살짝만 업데이트 (완전 고정하려면 requires_grad=False + 옵티마에서 제외)
        self.optimizer = optim.AdamW(
            [
                {"params": self.model.parameters(),   "lr": base_lr, "weight_decay": wdecay},
                {"params": self.ts_head.parameters(), "lr": base_lr, "weight_decay": wdecay},
                {"params": self.txt_head.parameters(),"lr": txt_lr,  "weight_decay": wdecay},
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 50), eta_min=1e-6)
        self.loss = MSE
        self.MAE  = MAE
        self.clip = 5

    def train(self, input, mark, last_emb, prompt_emb, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input, mark, last_emb, prompt_emb)     ####
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()
    
    def eval(self, input, mark, last_emb, prompt_emb, real_val):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input,mark, last_emb, prompt_emb)     #### 
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()

# ---------------------
# Training loop
# ---------------------
def main():
    args = parse_args()
    seed_it(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    # vis용 로더(첫 배치만 씀)
    vis_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)

    path = os.path.join(args.save, args.data_path,
                        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}/")
    os.makedirs(path, exist_ok=True)
    
    engine = trainer(
        scaler=scaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs,
        common_dim = args.common_dim
    )
    
    best_val = float('inf'); bestid = -1; test_log = float('inf'); epochs_since_best = 0
    his_loss, val_time, train_time = [], [], []
    print(args)
    
    # ===== Pre-training: 첫 배치 임베딩/지표/시각화 =====
    enc_pre, pr_pre, cr_pre, full_prompt_pre = get_first_batch_embeddings(
        vis_loader, engine.model, device
    )
    
    # 공통 공간 투영: [N,C] → [N,D], [N,d_llm] → [N,D]
    with torch.no_grad():
        enc_vec_pre = enc_pre[0].permute(1, 0).contiguous()    # [N, C]
        cross_vec_pre = cr_pre.squeeze(0)                  # [N, C]
        pr_vec_pre    = pr_pre.permute(0,2,1).squeeze(0)  # [N, d_llm]
        # full_prompt_pre: [B, T_max, d_llm, N] -> 첫 배치 토큰 평균: [N, d_llm]
        fp_pre = full_prompt_pre[0]                         # [T_max, d_llm, N]
        fp_avg_pre = fp_pre.mean(dim=0).permute(1, 0)       # [N, d_llm]

        ts_z_pre = F.normalize(engine.ts_head(enc_vec_pre.to(device)), dim=-1)    # [N, D] (ENC)
        cross_z_pre  = F.normalize(engine.ts_head(cross_vec_pre), dim=-1)  # [N, D]
        txt_z_pre = F.normalize(engine.txt_head(pr_vec_pre),    dim=-1) # [N, D]
        txt_full_z_pre = F.normalize(engine.txt_head(fp_avg_pre.to(device)), dim=-1)  # [N, D]

    # 공통 공간에서 pairwise MSE
    pd_mse_pre_common = compute_pairwise_distance_mse(cross_z_pre.cpu(), txt_z_pre.cpu())
    print(f"[Common] Pre-training Pairwise Distance MSE: {pd_mse_pre_common:.6f}")
    pd_mse_pre_common_full = compute_pairwise_distance_mse(
        cross_z_pre.detach().cpu(), txt_full_z_pre.detach().cpu()
    )
    print(f"[Common] Post-training Pairwise Distance MSE (TS vs FULL-PROMPT): {pd_mse_pre_common_full:.6f}")

    # 공통 공간에서 히트맵
    plot_pairwise_distance_heatmaps(
        txt_full_z_pre,     # X : FULL-PROMPT 공통공간
        txt_z_pre,    # Y: Last prompt token 공통공간
        cross_z_pre,     # Z: Cross Embedding 공통공간
        save_path="heatmap_common_pre.png",
        reduce_batch = 'mean'
    )

    # 공통 공간에서 per-channel cosine (요약 통계)
    cos_pre_common = F.cosine_similarity(cross_z_pre, txt_z_pre, dim=-1)  # [N]
    print(f"[Common] Pre-training avg cosine: {cos_pre_common.mean().item():.4f}")
    
    visualize_common_space_all_per_channel(
        enc_out=enc_pre,               # [B,C,N]
        cross_out=cr_pre,              # [B,N,C]
        prompt_last=pr_pre,            # [B,d_llm,N]
        full_prompt_emb=fp_pre,        # [B,T,d_llm,N]
        ts_head=engine.ts_head,        # C->D
        txt_head=engine.txt_head,      # d_llm->D
        method="tsne",
        token_sample_per_channel=50,
        color_map={"ENC":"#1f77b4","CROSS":"#2ca02c","LAST":"#ff7f0e","TOK":"#d62728"},
        save_path="common_space_pre.png"
    )

    print("===== Start Training =====", flush=True)
    for epoch in range(1, args.epochs + 1):
        t1 = time.time()
        engine.model.train()
        tr_pred, tr_align, tr_total, tr_mae = [], [], [], []

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]", ncols=120)
        for iter, (x, y, x_mark, y_mark, last_emb, full_prompt, anchor_avg) in train_bar:
            x, y = x.to(device), y.to(device)
            x_mark = x_mark.to(device)
            last_emb = last_emb.to(device)
            full_prompt = full_prompt.to(device) if iter == 0 else None
            anchor_avg = anchor_avg.to(device)  # [B, N, d_llm]

            engine.optimizer.zero_grad(set_to_none=True)

            # 예측 손실
            pred = engine.model(x, x_mark, last_emb, full_prompt)
            loss_pred = engine.loss(pred, y)

            # 정렬 손실: cross vs anchor_avg
            # === get cross embeddings ===
            with torch.no_grad():
                _, _, cr = engine.model.get_embeddings(x, x_mark, last_emb, None)  # 기대: [B, N, C]

            # cr 정규화: [B, N, C] 보장
            if cr.dim() != 3:
                raise ValueError(f"cr must be 3D [B,N,C], got {tuple(cr.shape)}")
            # 만약 [B, C, N]로 온 경우 뒤집기
            if cr.shape[1] == args.channel and cr.shape[2] == args.num_nodes:
                cr = cr.permute(0, 2, 1).contiguous()  # [B, N, C]

            # === anchor 정규화: [B, N, d_llm] 보장 ===
            anc = anchor_avg  # 원본
            if anc.dim() == 4:
                # 보통 [B, K, N, d_llm] 형태인 경우 → K 평균 풀링
                anc = anc.mean(dim=1)  # [B, N, d_llm]
            elif anc.dim() == 3:
                # [B, d_llm, N] 로 오는 경우 → [B, N, d_llm] 로 변환
                if anc.shape[1] == args.d_llm and anc.shape[2] == args.num_nodes:
                    anc = anc.permute(0, 2, 1).contiguous()
                # 이미 [B, N, d_llm] 이면 그대로 사용
            else:
                raise ValueError(f"anchor_avg bad shape: {tuple(anc.shape)} (need [B,N,d_llm] or [B,K,N,d_llm] or [B,d_llm,N])")

            # 디버그 프린트(한 번만 보고 싶으면 if it==0 같은 조건 걸기)
            if iter == 0:
                print(f"[debug] cr shape={tuple(cr.shape)}, anchor_avg(norm) shape={tuple(anc.shape)}")

            # --- 공통공간 매핑 ---
            ts_z  = engine.ts_head(cr)     # [B, N, D]
            txt_z = engine.txt_head(anc)   # [B, N, D]

            # --- 코사인 정렬 손실 ---
            ts_z  = F.normalize(ts_z,  dim=-1)
            txt_z = F.normalize(txt_z, dim=-1)
            cos_sim = F.cosine_similarity(ts_z, txt_z, dim=-1)  # [B, N]
            loss_align = (1.0 - cos_sim).mean()
            cos_sim = F.cosine_similarity(ts_z, txt_z, dim=-1)    # [B,N]
            loss_align = (1.0 - cos_sim).mean()

            loss = loss_pred + args.align_weight * loss_align
            loss.backward()
            if engine.clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(engine.model.parameters()) + list(engine.ts_head.parameters()) + list(engine.txt_head.parameters()),
                    engine.clip
                )
            engine.optimizer.step()

            tr_pred.append(loss_pred.item())
            tr_align.append(loss_align.item())
            tr_total.append(loss.item())
            tr_mae.append(engine.MAE(pred, y).item())
            
            if iter % args.print_every == 0:
                print(f"[Train][Ep {epoch:03d}][It {iter:05d}] pred={tr_pred[-1]:.6f} align={tr_align[-1]:.6f} total={tr_total[-1]:.6f}")

        t2 = time.time(); train_time.append(t2 - t1)
        print(f"Epoch {epoch:03d} Training Time: {t2 - t1:.2f}s")

        # ===== Validation =====
        engine.model.eval()
        va_pred, va_align, va_total, va_mae = [], [], [], []
        s1 = time.time()
        with torch.no_grad():
            valid_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} [Val]", ncols=120)
            for it, (x, y, x_mark, y_mark, last_emb, full_prompt, anchor_avg) in valid_bar:
                x = x.to(device); y = y.to(device)
                x_mark = x_mark.to(device)
                last_emb = last_emb.to(device)
                full_prompt = full_prompt.to(device) if it == 0 else None
                anchor_avg = anchor_avg.to(device)

                pred = engine.model(x, x_mark, last_emb, full_prompt)
                loss_pred = engine.loss(pred, y)
                
                _, _, cr = engine.model.get_embeddings(x, x_mark, last_emb, None)  # 기대: [B, N, C]

                # cr 정규화: [B, N, C] 보장
                if cr.dim() != 3:
                    raise ValueError(f"cr must be 3D [B,N,C], got {tuple(cr.shape)}")
                # 만약 [B, C, N]로 온 경우 뒤집기
                if cr.shape[1] == args.channel and cr.shape[2] == args.num_nodes:
                    cr = cr.permute(0, 2, 1).contiguous()  # [B, N, C]

                # === anchor 정규화: [B, N, d_llm] 보장 ===
                anc = anchor_avg  # 원본
                if anc.dim() == 4:
                    # 보통 [B, K, N, d_llm] 형태인 경우 → K 평균 풀링
                    anc = anc.mean(dim=1)  # [B, N, d_llm]
                elif anc.dim() == 3:
                    # [B, d_llm, N] 로 오는 경우 → [B, N, d_llm] 로 변환
                    if anc.shape[1] == args.d_llm and anc.shape[2] == args.num_nodes:
                        anc = anc.permute(0, 2, 1).contiguous()
                    # 이미 [B, N, d_llm] 이면 그대로 사용
                else:
                    raise ValueError(f"anchor_avg bad shape: {tuple(anc.shape)} (need [B,N,d_llm] or [B,K,N,d_llm] or [B,d_llm,N])")

                # 디버그 프린트(한 번만 보고 싶으면 if it==0 같은 조건 걸기)
                if iter == 0:
                    print(f"[debug] cr shape={tuple(cr.shape)}, anchor_avg(norm) shape={tuple(anc.shape)}")
                # print("cr.shape : ", cr.shape)
                # print("anc.shape : ", anc.shape)
                
                ts_z  = engine.ts_head(cr)
                txt_z = engine.txt_head(anc)  # 고정값

                ts_z  = F.normalize(ts_z,  dim=-1)
                txt_z = F.normalize(txt_z, dim=-1)
                loss_align = (1.0 - F.cosine_similarity(ts_z, txt_z, dim=-1)).mean()

                loss = loss_pred + args.align_weight * loss_align

                va_pred.append(loss_pred.item())
                va_align.append(loss_align.item())
                va_total.append(loss.item())
                va_mae.append(engine.MAE(pred, y).item())

                if it % args.print_every == 0:
                    print(f"[Valid][Ep {epoch:03d}][It {it:05d}] pred={loss_pred.item():.6f} align={loss_align.item():.6f} total={loss.item():.6f}")

        s2 = time.time(); val_time.append(s2 - s1)
        mtrain_pred, mtrain_align, mtrain_total, mtrain_mae = map(float, (np.mean(tr_pred), np.mean(tr_align), np.mean(tr_total), np.mean(tr_mae)))
        mvalid_pred, mvalid_align, mvalid_total, mvalid_mae = map(float, (np.mean(va_pred), np.mean(va_align), np.mean(va_total), np.mean(va_mae)))

        print("-----------------------")
        print(f"[Epoch {epoch:03d}] Train: pred={mtrain_pred:.6f} align={mtrain_align:.6f} total={mtrain_total:.6f} MAE={mtrain_mae:.6f}")
        print(f"[Epoch {epoch:03d}] Valid: pred={mvalid_pred:.6f} align={mvalid_align:.6f} total={mvalid_total:.6f} MAE={mvalid_mae:.6f}")
        print("-----------------------")

        his_loss.append(mvalid_total)
        if mvalid_total < best_val:
            best_val = mvalid_total; bestid = epoch; epochs_since_best = 0
            torch.save({
                "model": engine.model.state_dict(),
                "ts_head": engine.ts_head.state_dict(),
                "txt_head": engine.txt_head.state_dict()
            }, os.path.join(path, "best_model.pth"))
            print(f"### Update! best valid(total)={best_val:.6f} @epoch {epoch}")
        else:
            epochs_since_best += 1
            print("No update")

        engine.scheduler.step()
        if epochs_since_best >= args.es_patience and epoch >= args.epochs // 2:
            print("Early stopping."); break

    print(f"Average Training Time: {np.mean(train_time):.2f}s/epoch")
    print(f"Average Validation Time: {np.mean(val_time):.2f}s")

   # ===== Post-training: 첫 배치 임베딩/지표/시각화 (예전 로직 유지) =====
    enc_post, pr_post, cr_post, full_prompt_post = get_first_batch_embeddings(vis_loader, engine.model, device)
    with torch.no_grad():
        enc_vec_post = enc_post[0].permute(1, 0).contiguous()    # [N, C]
        cross_vec_post = cr_post.squeeze(0)                  # [N, C]
        pr_vec_post   = pr_post.permute(0,2,1).squeeze(0)  # [N, d_llm]
        # full_prompt_post: [B, T_max, d_llm, N] -> 첫 배치 토큰 평균: [N, d_llm]
        fp_post = full_prompt_post[0]                         # [T_max, d_llm, N]
        fp_avg_post = fp_post.mean(dim=0).permute(1, 0)       # [N, d_llm]

        ts_z_post = F.normalize(engine.ts_head(enc_vec_post.to(device)), dim=-1)    # [N, D] (ENC)
        cross_z_post  = F.normalize(engine.ts_head(cross_vec_post), dim=-1)  # [N, D]
        txt_z_post = F.normalize(engine.txt_head(pr_vec_post),    dim=-1) # [N, D]
        txt_full_z_post = F.normalize(engine.txt_head(fp_avg_post.to(device)), dim=-1)  # [N, D]

    # 공통 공간에서 pairwise MSE
    pd_mse_post_common = compute_pairwise_distance_mse(cross_z_post.cpu(), txt_z_post.cpu())
    print(f"[Common] Post-training Pairwise Distance MSE: {pd_mse_post_common:.6f}")
    pd_mse_post_common_full = compute_pairwise_distance_mse(
        cross_z_post.detach().cpu(), txt_full_z_post.detach().cpu()
    )
    print(f"[Common] Post-training Pairwise Distance MSE (TS vs FULL-PROMPT): {pd_mse_post_common_full:.6f}")

    # 공통 공간에서 히트맵
    plot_pairwise_distance_heatmaps(
        txt_full_z_post,     # X : FULL-PROMPT 공통공간
        txt_z_post,    # Y: Last prompt token 공통공간
        cross_z_post,     # Z: Cross Embedding 공통공간
        save_path="heatmap_common_post.png",
        reduce_batch = 'mean'
    )

    # 공통 공간에서 per-channel cosine (요약 통계)
    cos_post_common = F.cosine_similarity(cross_z_post, txt_z_post, dim=-1)  # [N]
    print(f"[Common] Post-training avg cosine: {cos_post_common.mean().item():.4f}")
    
    visualize_common_space_all_per_channel(
        enc_out=enc_post, cross_out=cr_post, prompt_last=pr_post, full_prompt_emb=fp_post,
        ts_head=engine.ts_head, txt_head=engine.txt_head, 
        method="tsne", token_sample_per_channel=50,
        save_path="common_space_post.png"
    )
    print("==> visualized post-training embeddings (first batch)")

    # (선택) 분포 비교도 공통 공간으로
    plot_similarity_distributions(cos_pre_common, cos_post_common, save_path="sim_distribution_common.png")

    # ---------- Test ----------
    ckpt = torch.load(os.path.join(path, "best_model.pth"), map_location=device)
    engine.model.load_state_dict(ckpt["model"])
    engine.ts_head.load_state_dict(ckpt["ts_head"])
    engine.txt_head.load_state_dict(ckpt["txt_head"])

    test_outs, test_ys = [], []
    engine.model.eval()
    with torch.no_grad():
        for it, (x, y, x_mark, y_mark, last_emb, full_prompt, anchor_avg) in enumerate(test_loader):
            x = x.to(device); y = y.to(device)
            x_mark = x_mark.to(device)
            last_emb = last_emb.to(device)
            full_prompt = full_prompt.to(device)  # 리소스 부족하면 None도 가능
            preds = engine.model(x, x_mark, last_emb, full_prompt)
            test_outs.append(preds); test_ys.append(y)

    test_pre = torch.cat(test_outs, 0)
    test_real = torch.cat(test_ys, 0)

    amse, amae = [], []
    for j in range(args.pred_len):
        pred = test_pre[:, j,].to(device)
        real = test_real[:, j, ].to(device)
        m = metric(pred, real)
        amse.append(m[0]); amae.append(m[1])

    print(f"On average horizons, Test MSE: {np.mean(amse):.4f}, Test MAE: {np.mean(amae):.4f}")
    print(f"Best epoch: {bestid}, Best valid(total loss): {best_val:.6f}")
    
if __name__ == "__main__":
    t1 = time.time()
    main()  # 위에서 정의한 main() 호출
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.4f}")