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
        #common_dim=128,
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

        # Data Normalization Layer
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # === Time-series Encoder (Node-wise) ===
        # L(length) -> C(channel)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        # N(nodes) as sequence axis, C(channel) as feature dimension
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True, norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers=self.e_layer).to(self.device)

        # === Time-series Encoder (Temporal) ===
        # Scalar value -> C(channel) vector embedding
        self.value_embed = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, self.channel),
            nn.LayerNorm(self.channel)
        ).to(self.device)
        # L(length) as sequence axis, C(channel) as feature dimension
        self.temp_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True, norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.temp_encoder = nn.TransformerEncoder(self.temp_layer, num_layers=self.e_layer).to(self.device)
        self._enc_time_cache = None

        # === Prompt Encoder ===
        # Encodes LLM embeddings
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True, norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers=self.e_layer).to(self.device)

        # === LLM Embedding Projection ===
        # Projects LLM embedding dim (d_llm) to TS channel dim (channel) for alignment
        self.proj_llm = nn.Linear(self.d_llm, self.channel).to(self.device)

        # === Cross-Modal Alignment ===
        # Query: [B, N, C], Key/Value: [B, N, C] -> Output: [B, N, C]
        self.cross = CrossModal(
            d_model=self.channel, n_heads=head, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n,
            dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # === Decoder ===
        # Predicts based on aligned features
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True, norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.d_layer).to(self.device)

        # === Final Projection ===
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

        # === Alignment Projection (NEW) ===
        # Projects enc_time (C dim) to LLM embedding dim (d_llm) for alignment loss calculation
        self.proj_enc_time = nn.Linear(self.channel, self.d_llm).to(self.device)
        
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
        # --- channel embedding path(기존) ---
        enc_out = self.ts_encoder(x_tokens)          # [B, N, C]
        #enc_out = enc_out_channel.permute(0, 2, 1).contiguous()  # [B, C, N]

        # === (2) Time-series Encoder (Temporal) ===
        # input_data: [B, L, N] -> unsqueeze -> [B, L, N, 1]
        v = input_data.unsqueeze(-1)
        # value_embed -> [B, L, N, C]
        v = self.value_embed(v)
        # reshape -> [B*N, L, C]
        B, L, N, C = v.shape
        v_flat = v.permute(0, 2, 1, 3).contiguous().view(B * N, L, C)
        # 메모리 정리
        del v
        # TransformerEncoder -> [B*N, L, C]
        enc_time_flat = self.temp_encoder(v_flat)
        # reshape -> [B, L, N, C]
        enc_time = enc_time_flat.view(B, N, L, C).permute(0, 2, 1, 3).contiguous()
        self._enc_time_cache = enc_time
        # 메모리 정리
        del v_flat, enc_time_flat
        
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

        # === (3) Prompt Encoder ===
        # last_emb: [B, d_llm, N] -> permute -> [B, N, d_llm]
        embeddings = last_emb.float().permute(0, 2, 1).contiguous()
        # TransformerEncoder -> [B, N, d_llm]
        prompt_encoded = self.prompt_encoder(embeddings)

        # === (4) LLM Embedding Projection ===
        # proj_llm(d_llm -> C) -> [B, N, C]
        prompt_proj = self.proj_llm(prompt_encoded)

        # === (5) Cross-Modal Alignment (Node-wise) ===
        # Query: enc_out [B, N, C], Key/Value: prompt_proj [B, N, C]
        cross_out = self.cross(enc_out, prompt_proj, prompt_proj)

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

        # === (6) Decoder ===
        # Decoder(cross_out, cross_out) -> [B, N, C]
        dec_out = self.decoder(cross_out, cross_out)
        # 메모리 정리
        del cross_out

        # === (7) Final Projection ===
        # c_to_length(C -> pred_len) -> [B, N, pred_len]
        dec_out = self.c_to_length(dec_out)
        # permute -> [B, pred_len, N]
        dec_out = dec_out.permute(0, 2, 1)
        # RevIN Denormalization
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out
    
    def get_embeddings(self, input_data, input_data_mark, last_emb, full_prompt_emb=None):
        input_data = input_data.float()
        input_data = self.normalize_layers(input_data, 'norm')

        # 안전한 permute 처리
        try:
            if input_data.dim() == 4:
                # 4차원 텐서인 경우 처리: [B, L1, L2, N]
                B, L1, L2, N = input_data.shape
                print(f"🔍 4차원 텐서 감지: B={B}, L1={L1}, L2={L2}, N={N}")
                
                if L1 == L2:
                    # 정사각형인 경우 평균 풀링으로 차원 축소
                    x_tokens = input_data.mean(dim=2).permute(0, 2, 1).contiguous()  # [B, N, L1]
                    print(f"✅ 정사각형 처리: {x_tokens.shape}")
                else:
                    # 정사각형이 아닌 경우 첫 번째 L 차원 사용
                    x_tokens = input_data[:, :, 0, :].permute(0, 2, 1).contiguous()  # [B, N, L1]
                    print(f"✅ 비정사각형 처리: {x_tokens.shape}")
            elif input_data.dim() == 3:
                # 3차원 텐서인 경우 기존 처리
                x_tokens = input_data.permute(0, 2, 1).contiguous()
                print(f"✅ 3차원 처리: {x_tokens.shape}")
            else:
                raise ValueError(f"예상치 못한 차원: {input_data.dim()}, shape: {input_data.shape}")
                
        except Exception as e:
            print(f"⚠️ Error in input_data permute: {e}, shape: {input_data.shape}")
            # [B, L, N] 형태가 아닌 경우 처리
            if input_data.dim() == 3 and input_data.shape[1] == self.num_nodes:
                x_tokens = input_data  # 이미 [B, N, L] 형태
            else:
                raise e
        
        x_tokens = self.length_to_feature(x_tokens)
        enc_out = self.ts_encoder(x_tokens)
        
        # last_emb 안전한 permute 처리
        try:
            if last_emb.dim() == 4:
                # 4차원인 경우 squeeze(0)으로 3차원으로 변환
                B, L, d_llm, N = last_emb.shape
                print(f"🔍 last_emb 4차원 감지: B={B}, L={L}, d_llm={d_llm}, N={N}")
                
                # squeeze(0)으로 배치 차원 제거
                squeezed_emb = last_emb.squeeze(0)  # [L, d_llm, N]
                print(f"✅ last_emb squeeze(0) 후: {squeezed_emb.shape}")
                
                # [L, d_llm, N] -> [L, N, d_llm]로 permute
                embeddings = squeezed_emb.permute(0, 2, 1).contiguous()  # [L, N, d_llm]
                print(f"✅ last_emb 4차원 처리 완료: {embeddings.shape}")
            elif last_emb.dim() == 3:
                # 3차원인 경우 기존 처리
                embeddings = last_emb.float().permute(0, 2, 1).contiguous()
                print(f"✅ last_emb 3차원 처리: {embeddings.shape}")
            else:
                raise ValueError(f"예상치 못한 last_emb 차원: {last_emb.dim()}, shape: {last_emb.shape}")
                
        except Exception as e:
            print(f"⚠️ Error in last_emb permute: {e}, shape: {last_emb.shape}")
            # [B, d_llm, N] 형태가 아닌 경우 처리
            if last_emb.dim() == 3 and last_emb.shape[2] == self.num_nodes:
                embeddings = last_emb.float()  # 이미 [B, N, d_llm] 형태
            else:
                raise e
        
        prompt_encoded = self.prompt_encoder(embeddings)
        prompt_proj = self.proj_llm(prompt_encoded)
        
        cross_out = self.cross(enc_out, prompt_proj, prompt_proj)

        return enc_out.detach(), prompt_encoded.detach(), cross_out.detach()
    
    # === (B) Dual 클래스 안에 메서드 추가 ===
    @torch.no_grad()  # 시점 정렬만 뽑을 때는 고정도 가능(원하면 제거해서 joint로 학습 가능)
    def get_time_embeddings(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        입력:  input_data [B, L, N]
        반환:  ts_time [B, L, N, C]  (시점별 임베딩)
        """
        x = input_data.float()
        # RevIN normalize (학습과 동일)
        x = self.normalize_layers(x, 'norm')              # [B, L, N]

        B, L, N = x.shape
        # 값(스칼라) -> 채널차원 C : [B,L,N,1] -> [B,L,N,C]
        xv = x.unsqueeze(-1)
        xv = self.value_embed(xv)                         # [B,L,N,C]

        # temporal encoder: L 차원 문맥 반영
        # [B,L,N,C] -> [B,N,L,C] -> (B*N, L, C)
        tmp = xv.permute(0, 2, 1, 3).reshape(B * N, L, self.channel)
        tmp = self.temp_encoder(tmp)                      # [B*N, L, C]
        ts_time = tmp.reshape(B, N, L, self.channel).permute(0, 2, 1, 3).contiguous()  # [B,L,N,C]
        return ts_time
    
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
        
        # enc_time은 forward 과정에서 캐시된 텐서를 사용합니다.
        enc_time = self._enc_time_cache

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

                    # 텍스트 토큰 임베딩 (d_llm 차원)
                    tok_vec = full_prompt_emb[b, token_ids, :, n].mean(dim=0)

                    # enc_time 임베딩 (L, N, C) -> (L, N, C) -> (t, n, C)
                    enc_time_emb = enc_time[t, n, :]                        # [d_llm]

                    # C 차원 -> d_llm 차원으로 투영
                    proj_enc_time = self.proj_enc_time(enc_time_emb)
                    
                    # 코사인 유사도 계산
                    loss_t = 1.0 - F.cosine_similarity(
                        F.normalize(proj_enc_time, dim=0),
                        F.normalize(tok_vec.detach(), dim=0),
                        dim=0
                    )
                    
                    total = total + loss_t
                    count += 1

        if count == 0:
            return total 

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
        # model_name='gpt2',
        # llm_token_sample_size=50,
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
    
    @staticmethod
    def visualize_triple_embeddings(
        enc_out: torch.Tensor,
        prompt_proj: torch.Tensor,
        cross_out: torch.Tensor,
        save_path: str = "triple_embedding_viz.png",
        reduce_mode: str = "flatten_bn", # 'first_b' | 'mean_b' | 'flatten_bn' | 'mean_bn'
        method: str = "tsne",
        perplexity: float = 30.0,
        random_state: int = 42,
        figsize=(8, 8),
    ):
        """
        시계열, 투영된 프롬프트, 교차 모달 임베딩을 시각화하는 함수.
        reduce_mode를 통해 다양한 데이터 변환 방법을 지원함.
        """
        
        def _to_2d_and_flatten(t: torch.Tensor, how: str) -> np.ndarray:
            if t.dim() != 3:
                return t.cpu().detach().numpy()
            
            B, N, C = t.shape
            if how == "first_b":
                t_2d = t[0]
            elif how == "mean_b":
                t_2d = t.mean(dim=0)
            elif how == "flatten_bn":
                t_2d = t.reshape(B * N, C)
            elif how == "mean_bn":
                t_2d = t.mean(dim=(0, 1)).unsqueeze(0)
            else:
                raise ValueError(f"Unknown reduce_mode: {how}")
            
            return t_2d.cpu().detach().numpy()
            
        # === 1. 입력 텐서 변환 및 CPU로 이동 ===
        ts_emb = _to_2d_and_flatten(enc_out, reduce_mode)
        pr_emb = _to_2d_and_flatten(prompt_proj, reduce_mode)
        cross_emb = _to_2d_and_flatten(cross_out, reduce_mode)

        # === 2. 모든 임베딩을 하나의 배열로 합치기 ===
        X = np.vstack([ts_emb, pr_emb, cross_emb])
        
        # 각 임베딩 타입에 대한 레이블 생성
        labels = (
            ['Time-series'] * ts_emb.shape[0] +
            ['Projected Prompt'] * pr_emb.shape[0] +
            ['Cross-modal'] * cross_emb.shape[0]
        )

        # === 3. 차원 축소 (PCA 또는 t-SNE) ===
        if method.lower() == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            X_reduced = reducer.fit_transform(X)
            title = f"PCA Visualization (var: {reducer.explained_variance_ratio_.sum():.2f})"
        elif method.lower() == "tsne":
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                init='pca',
                random_state=random_state,
                learning_rate='auto'
            )
            X_reduced = reducer.fit_transform(X)
            title = f"t-SNE Visualization (perplexity={perplexity})"
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        # === 4. 시각화 ===
        plt.figure(figsize=figsize)
        
        color_map = {
            'Time-series': '#1f77b4',
            'Projected Prompt': '#ff7f0e',
            'Cross-modal': '#2ca02c',
        }
        marker_map = {
            'Time-series': 'o',
            'Projected Prompt': 's',
            'Cross-modal': '^',
        }
        
        for label_name, color in color_map.items():
            indices = [i for i, l in enumerate(labels) if l == label_name]
            plt.scatter(
                X_reduced[indices, 0],
                X_reduced[indices, 1],
                label=label_name,
                marker=marker_map[label_name],
                alpha=0.7,
                s=30,
                color=color
            )

        plt.title(title + f" | Mode: {reduce_mode}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Plot saved to {save_path}")
        
    # def visualize_common_embeddings(
    #     ts_z: torch.Tensor,
    #     txt_z: torch.Tensor,
    #     save_path: str = "common_embed.png",
    #     method: str = "pca",
    #     reduce_mode: str = "mean_b",  # 'first_b' | 'mean_b' | 'flatten_bn' | 'mean_bn'
    #     standardize: bool = True,       # 시각화 전 표준화(z-score) 여부
    #     max_points: int = 5000,         # 너무 많으면 성능 위해 샘플링
    #     random_state: int = 42,         # 재현성
    #     title: str | None = None,
    #     figsize=(7, 6),
    #     alpha: float = 0.7,
    #     s: float = 24.0,
    #     channel_labels: list[str] | None = None, # 옵션 : 채널 이름
    # ):
    #     """
    #     ts_z, txt_z: [N, D] 또는 [B, N, D]
    #     reduce_mode:
    #     - 'first_b'   : 첫 배치만 사용 → [N, D]
    #     - 'mean_b'    : 배치 평균     → [N, D]  ← 추천
    #     - 'flatten_bn': (B*N, D)로 펼쳐서 시각화
    #     - 'mean_bn'   : 배치/노드 모두 평균 → [D]
    #     """
    #     def to_2d(t: torch.Tensor, how: str) -> torch.Tensor:
    #         if t.dim() == 2:
    #             return t
    #         if t.dim() == 3:
    #             B, N, D = t.shape
    #             if how == "first_b":
    #                 return t[0]                # [N, D]
    #             elif how == "mean_b":
    #                 return t.mean(dim=0)       # [N, D]
    #             elif how == "flatten_bn":
    #                 return t.reshape(B * N, D) # [B*N, D]
    #             elif how == "mean_bn":
    #                 return t.mean(dim=(0, 1))  # [D]
    #             else:
    #                 raise ValueError(f"Unknown reduce_mode: {how}")
    #         raise ValueError(f"Expected 2D or 3D tensor, got {tuple(t.shape)}")

    #     ts_z_2d  = to_2d(ts_z,  reduce_mode)
    #     txt_z_2d = to_2d(txt_z, reduce_mode)

    #     # 'mean_bn'이면 [D]가 되므로 2D로 확장
    #     if ts_z_2d.dim() == 1:
    #         ts_z_2d = ts_z_2d.unsqueeze(0)  # [1, D]
    #     if txt_z_2d.dim() == 1:
    #         txt_z_2d = txt_z_2d.unsqueeze(0)  # [1, D]

    #     assert ts_z_2d.shape[1] == txt_z_2d.shape[1], \
    #         f"Dim mismatch: {ts_z_2d.shape} vs {txt_z_2d.shape} (D must match)"

    #     # 2) 텐서를 CPU numpy로 변환
    #     Ts = ts_z_2d.detach().cpu().float().numpy()  # [M1, D]
    #     Tx = txt_z_2d.detach().cpu().float().numpy() # [M2, D]

    #     # 3) 너무 많은 점이면 샘플링 (성능/가독성)
    #     rng = np.random.default_rng(random_state)
    #     def maybe_subsample(A: np.ndarray, max_n: int) -> np.ndarray:
    #         if A.shape[0] <= max_n:
    #             return A
    #         idx = rng.choice(A.shape[0], size=max_n, replace=False)
    #         return A[idx]
    #     # 양쪽을 독립적으로 샘플링(라벨 균형 유지)
    #     Ts = maybe_subsample(Ts, max_points // 2 if max_points else Ts.shape[0])
    #     Tx = maybe_subsample(Tx, max_points // 2 if max_points else Tx.shape[0])

    #     # 4) 표준화(옵션) — 두 분포를 같은 스케일로
    #     if standardize:
    #         from sklearn.preprocessing import StandardScaler
    #         scaler = StandardScaler()
    #         Z = np.vstack([Ts, Tx])          # [M1+M2, D]
    #         Z = scaler.fit_transform(Z)
    #     else:
    #         Z = np.vstack([Ts, Tx])

    #     # 5) 차원축소 (PCA 또는 t-SNE)
    #     if method.lower() == "pca":
    #         from sklearn.decomposition import PCA
    #         reducer = PCA(n_components=2, random_state=random_state)
    #         Z2 = reducer.fit_transform(Z)    # [M, 2]
    #         sub_title = f"PCA (var: {reducer.explained_variance_ratio_.sum():.2f})"
    #     elif method.lower() == "tsne":
    #         from sklearn.manifold import TSNE
    #         # perplexity는 표본 수보다 작아야 안정적
    #         perplexity = min(30, max(5, (Z.shape[0] // 10)))
    #         reducer = TSNE(
    #             n_components=2,
    #             perplexity=perplexity,
    #             learning_rate="auto",
    #             init="pca",
    #             random_state=random_state,
    #             verbose=False,
    #         )
    #         Z2 = reducer.fit_transform(Z)
    #         sub_title = f"t-SNE (perp={perplexity})"
    #     else:
    #         raise ValueError("method must be 'pca' or 'tsne'")

    #     # 6) 분리해서 색상/마커로 표시
    #     m1 = Ts.shape[0]
    #     xy_ts = Z2[:m1]
    #     xy_tx = Z2[m1:]

    #     plt.figure(figsize=figsize)
    #     plt.scatter(xy_ts[:, 0], xy_ts[:, 1], s=s, alpha=alpha, label="TS embeddings", marker='o')
    #     plt.scatter(xy_tx[:, 0], xy_tx[:, 1], s=s, alpha=alpha, label="TXT embeddings", marker='^')
        
    #     # 채널 라벨 표시(옵션) — reduce_mode가 [N,D]일 때만 의미 있음
    #     if channel_labels and ts_z_2d.shape[0] == len(channel_labels):
    #         for i, name in enumerate(channel_labels):
    #             plt.annotate(name, (xy_ts[i, 0], xy_ts[i, 1]), fontsize=9, alpha=0.8)
    #             if i < xy_tx.shape[0]:
    #                 plt.annotate(name, (xy_tx[i, 0], xy_tx[i, 1]), fontsize=9, alpha=0.8)

    #     ttl = title if title is not None else f"Common space visualization - {sub_title} | mode={reduce_mode}"
    #     plt.title(ttl)
    #     plt.xlabel("Component 1")
    #     plt.ylabel("Component 2")
    #     plt.legend(loc="best", frameon=True)
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=150)
    #     plt.close()
    #     print(f"✅ Saved common-embedding plot to {save_path}")