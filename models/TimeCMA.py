import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
# import umap.umap_ as umap

from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from transformers import GPT2Tokenizer, GPT2Model

class Dual(nn.Module):
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,
        common_dim=128,
        e_layer = 1,
        d_layer = 1,
        d_ff=32,
        head =8, 
        visualize = False
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        #self.common_dim =128
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        # Time Series Encoder
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                           norm_first = True,dropout = self.dropout_n).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_llm, nhead = self.head, batch_first=True, 
                                                               norm_first = True,dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Cross-modality alignment
        # self.cross_layer = nn.TransformerDecoderLayer(d_model = self.num_nodes, nhead = 1, batch_first=True, norm_first = True,dropout = self.dropout_n).to(self.device)
        # self.cross = nn.TransformerDecoder(self.cross_layer, num_layers = 1).to(self.device)
        self.cross = CrossModal(d_model= self.num_nodes, n_heads= 1, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False).to(self.device)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        # Projection
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        self.value_proj = nn.Sequential(
            nn.Linear(1, self.d_llm),
            nn.ReLU(),
            nn.Linear(self.d_llm, self.d_llm),
            nn.LayerNorm(self.d_llm)
        ).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_data,
        input_data_mark,
        last_emb,         # [B, d_llm, N] — 기존 마지막 토큰 임베딩
        full_prompt_emb=None,  # [B, T_max, d_llm, N] — optional, 필요하면 사용
        *,
        visualize=False,
        tsne_perplexity=30.0
    ):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()

        # RevIN 적용
        input_data = self.normalize_layers(input_data, 'norm')

        # --- 입력 축 보정 ---
        if input_data.dim() != 3:
            raise ValueError(f"input_data must be 3D [B, L, N], got {tuple(input_data.shape)}")

        B, L, N = input_data.shape
        # [B, N, L]로 잘못 들어온 경우 복구
        if L == self.num_nodes and N == self.seq_len:
            input_data = input_data.permute(0, 2, 1)  # [B, L, N]
            B, L, N = input_data.shape

        # print(f"[DEBUG] after input reshape: {input_data.shape}")  # [B, L, N] 기대

        # [B, L, N] → [B, N, L]
        x_tokens = input_data.permute(0, 2, 1).contiguous()

        # print(f"[DEBUG] before length_to_feature: {x_tokens.shape}")  # [B, N, L] 기대

        # Linear(L -> C)
        x_tokens = self.length_to_feature(x_tokens)  # [B, N, C]
        # print(f"[DEBUG] after length_to_feature: {x_tokens.shape}")  # [B, N, C], 여기서 C=32 기대

        # Transformer Encoder
        enc_out = self.ts_encoder(x_tokens)          # [B, N, C]
        enc_out = enc_out.permute(0, 2, 1).contiguous()  # [B, C, N]

        # --- 이하 last_emb, prompt_encoded, cross_out 처리 ---
        print("last_emb.shape: ", last_emb.shape)
        if last_emb.is_sparse:
            last_emb = last_emb.to_dense()
        while last_emb.dim() > 3:
            last_emb = last_emb.mean(dim=1)
        if last_emb.dim() == 2:
            last_emb = last_emb.unsqueeze(0)
        if last_emb.dim() != 3:
            raise ValueError(f"last_emb must be 3D [B, d_llm, N], got {tuple(last_emb.shape)}")

        embeddings = last_emb.float().permute(0, 2, 1).contiguous()  # [B, N, d_llm]
        prompt_encoded = self.prompt_encoder(embeddings)             # [B, N, d_llm]
        prompt_encoded = prompt_encoded.permute(0, 2, 1)              # [B, d_llm, N]

        cross_out = self.cross(enc_out, prompt_encoded, prompt_encoded)  # [B, C, N]
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]

        # (선택) 시각화용 full_prompt_emb 정규화
        if visualize and (full_prompt_emb is not None):
            if full_prompt_emb.is_sparse:
                full_prompt_emb = full_prompt_emb.to_dense()
            while full_prompt_emb.dim() > 4:
                full_prompt_emb = full_prompt_emb.mean(dim=1)            # → [B, T, d_llm, N]
            if full_prompt_emb.dim() == 3:
                full_prompt_emb = full_prompt_emb.unsqueeze(0)           # [T, d_llm, N] → [1, T, d_llm, N]
            if full_prompt_emb.dim() != 4:
                raise ValueError(f"full_prompt_emb must be 4D [B, T, d_llm, N], got {tuple(full_prompt_emb.shape)}")

            self.visualize_embeddings(
                enc_out,              # [B, C, N]
                prompt_encoded,       # [B, d_llm, N]
                cross_out,            # [B, N, C]
                full_prompt_emb,
                perplexity=tsne_perplexity
            )

        # Decoder
        dec_out = self.decoder(cross_out, cross_out) # [B, N, C]

        # Projection
        dec_out = self.c_to_length(dec_out) # [B, N, L]
        dec_out = dec_out.permute(0,2,1) # [B, L, N]

        # denorm
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out
    
    def get_embeddings(
        self,
        input_data: torch.Tensor,
        input_data_mark: torch.Tensor,
        last_emb: torch.Tensor,               # 기대: [B, d_llm, N]  (실제: [B, K, d_llm, N] 가능)
        full_prompt_emb: torch.Tensor = None  # 기대: [B, T, d_llm, N] (실제: [B, K, T, d_llm, N] 가능)
    ):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()

        # RevIN 적용
        input_data = self.normalize_layers(input_data, 'norm')

        # --- 입력 축 보정 ---
        if input_data.dim() != 3:
            raise ValueError(f"input_data must be 3D [B, L, N], got {tuple(input_data.shape)}")

        B, L, N = input_data.shape
        # [B, N, L]로 잘못 들어온 경우 복구
        if L == self.num_nodes and N == self.seq_len:
            input_data = input_data.permute(0, 2, 1)  # [B, L, N]
            B, L, N = input_data.shape

        # print(f"[DEBUG] after input reshape: {input_data.shape}")  # [B, L, N] 기대

        # [B, L, N] → [B, N, L]
        x_tokens = input_data.permute(0, 2, 1).contiguous()

        # print(f"[DEBUG] before length_to_feature: {x_tokens.shape}")  # [B, N, L] 기대

        # Linear(L -> C)
        x_tokens = self.length_to_feature(x_tokens)  # [B, N, C]
        # print(f"[DEBUG] after length_to_feature: {x_tokens.shape}")  # [B, N, C], 여기서 C=32 기대

        # Transformer Encoder
        enc_out = self.ts_encoder(x_tokens)          # [B, N, C]
        enc_out = enc_out.permute(0, 2, 1).contiguous()  # [B, C, N]

        # --- 이하 last_emb, prompt_encoded, cross_out 처리 ---
        if last_emb.is_sparse:
            last_emb = last_emb.to_dense()
        while last_emb.dim() > 3:
            last_emb = last_emb.mean(dim=1)
        if last_emb.dim() == 2:
            last_emb = last_emb.unsqueeze(0)
        if last_emb.dim() != 3:
            raise ValueError(f"last_emb must be 3D [B, d_llm, N], got {tuple(last_emb.shape)}")

        embeddings = last_emb.float().permute(0, 2, 1).contiguous()  # [B, N, d_llm]
        prompt_encoded = self.prompt_encoder(embeddings)             # [B, N, d_llm]
        prompt_encoded = prompt_encoded.permute(0, 2, 1)              # [B, d_llm, N]

        cross_out = self.cross(enc_out, prompt_encoded, prompt_encoded)  # [B, C, N]
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]                 # [B, N, C]

        return enc_out.detach(), prompt_encoded.detach(), cross_out.detach()
    
    def compute_alignment_loss(
        self,
        full_prompt_emb: torch.Tensor,   # [B, T_max, d_llm, N]
        num_values_list,                 # length B list; each is length N list of length-L tensors (값 시퀀스)
        num_token_idx_list,              # length B list; each is length N list of length-L list[int] (토큰 idx들)
        *,
        reduce: str = "mean"
    ):
        """
        숫자값과 그 숫자에 대응하는 프롬프트 토큰 임베딩들의 평균이
        같은 방향을 보도록(=높은 코사인 유사도) 학습시키는 정렬 손실.
        """
        device = full_prompt_emb.device
        total = full_prompt_emb.new_tensor(0.0)
        count = 0

        B = full_prompt_emb.shape[0]
        N = full_prompt_emb.shape[-1]   # 채널 개수

        for b in range(B):
            for n in range(N):
                vals_t = num_values_list[b][n]     # torch.Tensor, shape [L]
                idxs_t = num_token_idx_list[b][n]  # list of list[int], len L
                # 보호: 길이 불일치/빈 리스트 skip
                L = min(len(vals_t), len(idxs_t))
                if L == 0: 
                    continue

                vals_t = vals_t.to(device)

                for t in range(L):
                    token_ids = idxs_t[t]
                    if not token_ids:   # 비어 있으면 skip
                        continue

                    # 1) 프롬프트 토큰 임베딩 평균: [d_llm]
                    tok_vec = full_prompt_emb[b, token_ids, :, n].mean(dim=0)     # [d_llm]

                    # 2) 숫자값 -> 벡터: [1] -> [1,1] -> value_proj -> [1, d_llm] -> [d_llm]
                    v = vals_t[t].view(1, 1)
                    val_vec = self.value_proj(v).view(-1)                         # [d_llm]

                    # 3) 코사인 손실: 1 - cos
                    loss_t = 1.0 - F.cosine_similarity(
                        F.normalize(val_vec, dim=0),
                        F.normalize(tok_vec.detach(), dim=0),  # 토큰 임베딩은 저장된 고정값이면 detach 권장
                        dim=0
                    )
                    total = total + loss_t
                    count += 1

        if count == 0:
            return total  # 0

        if reduce == "mean":
            return total / count
        else:
            return total
        
    def visualize_embeddings(
        self,
        enc_out: torch.Tensor,          # [B, C, N] or [C, N]
        prompt_emb: torch.Tensor,       # [B, d_llm, N] or [d_llm, N]
        cross_out: torch.Tensor,        # [B, N, C] or [N, C]
        full_prompt_emb: torch.Tensor | None = None,  # [B, T_max, d_llm, N]
        model_name='gpt2',
        llm_token_sample_size=50,
        perplexity: float = 30.0,
        save_path: str = 'combined_embedding_viz.png'
    ):
        # ---- 0. Normalize shapes to have batch dim ----
        if enc_out.dim() == 2:  # [C, N] -> [1, C, N]
            enc_out = enc_out.unsqueeze(0)
        if prompt_emb.dim() == 2:  # [d_llm, N] -> [1, d_llm, N]
            prompt_emb = prompt_emb.unsqueeze(0)
        if cross_out.dim() == 2:  # [N, C] -> [1, N, C]
            cross_out = cross_out.unsqueeze(0)

        B, C, N = enc_out.shape
        _, E, _ = prompt_emb.shape  # d_llm

        # ---- 1. Time-series embedding flatten ----
        ts_emb = enc_out.permute(0, 2, 1).reshape(-1, C).cpu().detach().numpy()  # [B*N, C]

        # ---- 2. Prompt last-token embedding flatten + PCA to match C ----
        pr = prompt_emb.permute(0, 2, 1).reshape(-1, E)      # [B*N, E]
        try:
            projector = nn.Linear(E, C).to(pr.device)       # [B*N, C]
            pr_proj = projector(pr)
            #print(pr_proj.shape)
            #pr = prompt_emb.permute(0, 2, 1).reshape(-1, E).cpu().detach().numpy()  # [B*N, E]
            pr_proj = pr_proj.cpu().detach().numpy()
        except Exception:
            print("fallback")
            pr_proj = pr  # fallback

        # ---- 3. Cross embedding ----
        cross = cross_out.reshape(-1, C).cpu().detach().numpy()  # [B*N, C]
        cross_proj = cross  # no need for PCA if same dim; could apply if desired

        # ---- 4. Full prompt token embeddings ----
        if full_prompt_emb is None:
            raise ValueError("full_prompt_emb is required for token-level prompt visualization")
        # Ensure full_prompt_emb has batch dim
        if full_prompt_emb.dim() == 3:  # maybe passed as [T_max, d_llm, N]
            full_prompt_emb = full_prompt_emb.unsqueeze(0)  # [1, T_max, d_llm, N]
        fp = full_prompt_emb[0]  # [T_max, d_llm, N]
        T_max, d_llm, nch = fp.shape

        if B > 16 :
            token_embeddings = fp.permute(0, 2, 1).reshape(-1, d_llm).cpu().detach().numpy()  # [T_max * N, d_llm]
            try:
                token_proj = PCA(n_components=C).fit_transform(token_embeddings)
            except Exception:
                token_proj = token_embeddings
        else:
            token_proj = None
            print(f"Skipped prompt token embedding visualization due to small batch size (B={B})")

        # ---- 5. Stack for visualization ----
        X_list = [ts_emb, pr_proj, cross_proj]
        labels = (
            ['ts'] * ts_emb.shape[0]
            + ['prompt_last'] * pr_proj.shape[0]
            + ['cross'] * cross_proj.shape[0]
        )
        if token_proj is not None:
            X_list.append(token_proj)
            labels += ['prompt_tokens'] * token_proj.shape[0]

        X = np.vstack(X_list)

        # ---- 6. t-SNE ----
        proj = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42).fit_transform(X)
        plt.figure(figsize=(8, 8))

        color_map = {
            'ts': '#1f77b4',          # 파란
            'cross': '#2ca02c',       # 초록
            'prompt_last': '#ff7f0e', # 주황
            'prompt_tokens': '#d62728' # 빨강
        }

        def scatter_with_label(name, marker, indices):
            plt.scatter(
                proj[indices, 0],
                proj[indices, 1],
                label=name,
                marker=marker,
                alpha=0.7,
                s=30,
                color=color_map.get(name)
            )

        idx_ts = [i for i, l in enumerate(labels) if l == 'ts']
        idx_cross = [i for i, l in enumerate(labels) if l == 'cross']
        idx_last = [i for i, l in enumerate(labels) if l == 'prompt_last']
        idx_tokens = [i for i, l in enumerate(labels) if l == 'prompt_tokens'] if 'prompt_tokens' in labels else []

        scatter_with_label('ts', 'x', idx_ts)
        scatter_with_label('cross', 'x', idx_cross)
        scatter_with_label('prompt_last', 'x', idx_last)
        if idx_tokens:  # token_proj가 있을 때만 실행
            scatter_with_label('prompt_tokens', 'x', idx_tokens)

        plt.legend()
        plt.title("Time-series vs Prompt Last vs Prompt Token Embeddings")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved plot to {save_path}")
