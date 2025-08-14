import sys
import os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import h5py
import argparse
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))
    return parser.parse_args()

def get_dataset(data_path, flag, input_len, output_len):
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path)

def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = get_dataset(args.data_path, 'train', args.input_len, args.output_len)
    test_set = get_dataset(args.data_path, 'test', args.input_len, args.output_len)
    val_set = get_dataset(args.data_path, 'val', args.input_len, args.output_len)

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    }[args.divide]

    gen_prompt_emb = GenPromptEmb(
        device=device, # type: ignore
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide
    ).to(device)

    save_path = f"./MY/Embeddings/{args.data_path}/{args.divide}/"
    # save_path2 = f"./TimeCMA/Prompt_emb/{args.data_path}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(save_path2, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        ### debugging included code ###
        print(f"[{args.divide}] Processing batch {i}")
        x = x.to(device)              # [B, L, N]
        x_mark = x_mark.to(device)    # [B, L, F]
        print("Shape of input x:", x.shape)
        print("Shape of input x_mark:", x_mark.shape)

        # --- ① 프롬프트 임베딩 전체/마지막 토큰 임베딩 산출 ---
        # 아래 generate 함수가 (last_token_emb, in_prompt_emb, meta) 를 반환하도록
        # gen_prompt_emb.py를 업데이트했다는 가정 (아래에 예시 제공)
        last_token_emb, prompt_token_emb, align_meta = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))
        ### in_prompt_emb : prompt에 이용된 단어들의 임베딩 집합
        # last_emb: [B, d_llm, N]   (보통 B=1)
        # full_prompt_emb: [B, T_max, d_llm, N]
        # align_meta: list of length B; align_meta[b][n] = [(v_t, [idx...]), ...] (길이 L)

        print("Shape of last_token_emb:", last_token_emb.shape)     # [B, d_model, N]
        print("Shape of prompt_token_emb:", prompt_token_emb.shape) # [B, T_max, d_model, N]

        # --- ② 채널별 anchor_avg 계산 ---
        # anchor_avg[b, n, d_llm] = 그 채널 prompt 안의 모든 "숫자 토큰 인덱스" 임베딩 평균
        B, T_max, d_llm, N = prompt_token_emb.shape
        anchor_avg = torch.zeros((B, N, d_llm), device=device)
        num_token_idx_all = []   # JSON 저장용 (배치마다 채널별 리스트)
        num_values_all   = []    # JSON 저장용

        for b in range(B):
            num_token_idx_per_b = []
            num_values_per_b    = []
            for n in range(N):
                # meta[b][n]가 없다면 skip
                entry = align_meta[b].get(n, None) if isinstance(align_meta[b], dict) else None
                if entry is None:
                    # fallback: 마지막 토큰 하나만 anchor로
                    anchor_avg[b, n] = prompt_token_emb[b, -1, :, n]
                    num_token_idx_per_b.append([])  # 빈 리스트
                    num_values_per_b.append([])
                    continue

                value_index_pairs = entry.get("value_index_pairs", [])  # [(value, [idx, ...]), ...]
                values            = entry.get("values", [])             # [float, float, ...]
                # 숫자 토큰 인덱스 전부 합치기
                all_token_idx = []
                for _, idx_list in value_index_pairs:
                    all_token_idx.extend(idx_list)
                # valid 인덱스만 남기기
                all_token_idx = [idx for idx in all_token_idx if 0 <= idx < T_max]

                if len(all_token_idx) == 0:
                    # 없으면 마지막 토큰 임베딩 사용
                    anchor_avg[b, n] = prompt_token_emb[b, -1, :, n]
                else:
                    # [lenK, d_llm] 평균
                    token_emb = prompt_token_emb[b, all_token_idx, :, n]  # [K, d_llm]
                    anchor_avg[b, n] = token_emb.mean(dim=0)

                num_token_idx_per_b.append(all_token_idx)
                num_values_per_b.append(values)

            num_token_idx_all.append(num_token_idx_per_b)
            num_values_all.append(num_values_per_b)

        # --- ③ 저장 ---
        file_path = f"{save_path}{i}.h5"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print("Saving to:", file_path)

        with h5py.File(file_path, 'w') as hf:
            # 마지막 토큰 임베딩
            hf.create_dataset(
                'last_token_embeddings',
                data = last_token_emb.cpu().numpy(),
                compression = 'gzip'
            )   # [B, d_llm, N] 
            # 프롬프트에 포함된 모든 토큰 임베딩
            hf.create_dataset(
                'prompt_token_embeddings', 
                data = prompt_token_emb.cpu().numpy(),
                compression = 'gzip'
            )   # [B, T_max, d_llm, N]
            # 새로 추가
            hf.create_dataset('anchor_avg', data=anchor_avg.detach().cpu().numpy())                  # [B, N, d_llm]
            # ragged: JSON 문자열로
            hf.create_dataset('num_token_idx_json', data=np.bytes_(json.dumps(num_token_idx_all)))
            hf.create_dataset('num_values_json',    data=np.bytes_(json.dumps(num_values_all)))

        print(f"✅ Saved {file_path}\n")
        
        ### original code
        # embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))

        # file_path = f"{save_path}{i}.h5"
        # with h5py.File(file_path, 'w') as hf:
        #     hf.create_dataset('embeddings', data = embeddings.cpu().numpy())

        # Save and visualize the first sample
        # if i >= 0:
        #     break
    
if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")