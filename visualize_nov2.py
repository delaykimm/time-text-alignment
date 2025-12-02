#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py로 저장된 모델을 로드하고 채널 간 similarity distribution을 시각화하는 스크립트

사용법:
    python visualize_nov.py --model_path <모델_경로> [옵션들]
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.TimeCMA import Dual
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from train import custom_collate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, GPT2Model

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_trained_model(model_path, args, sample_x=None):
    """
    train.py를 통해 저장된 모델을 로드
    
    Args:
        model_path: 저장된 모델 체크포인트 경로
        args: 모델 설정을 담은 argparse.Namespace 객체
        sample_x: 샘플 입력 데이터 [B, L, N] (선택사항, 제공되면 shape에서 num_nodes 추출)
    
    Returns:
        model: 로드된 모델 (eval 모드)
    """
    print(f"📂 모델 불러오는 중: {model_path}")
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 모델 가중치 불러오기 (channel 확인용)
    checkpoint = torch.load(model_path, map_location=device)
    
    # num_nodes와 channel을 동적으로 설정
    # 1. num_nodes: 입력 shape의 마지막 차원 (N)
    if sample_x is not None:
        # sample_x shape: [B, L, N]
        print(f"🔍 sample_x shape: {sample_x.shape}")
        args.num_nodes = sample_x.shape[-1]
        print(f"✅ 입력 데이터에서 num_nodes={args.num_nodes} 추출 (shape: {sample_x.shape})")
    else:
        # sample_x가 없으면 기본값 사용
        print(f"⚠️  샘플 데이터가 없어 기본값 num_nodes={args.num_nodes} 사용")
    
    # 2. channel: 체크포인트의 length_to_feature 출력 차원 (x_tokens의 마지막 차원)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        model_state = checkpoint
    
    # length_to_feature의 출력 차원 확인
    if "length_to_feature.weight" in model_state or "_orig_mod.length_to_feature.weight" in model_state:
        key = "length_to_feature.weight" if "length_to_feature.weight" in model_state else "_orig_mod.length_to_feature.weight"
        # length_to_feature: [channel, seq_len]
        args.channel = model_state[key].shape[0]
        print(f"✅ 체크포인트에서 channel={args.channel} 추출 (length_to_feature 출력 차원)")
    else:
        print(f"⚠️  length_to_feature를 찾을 수 없어 기본값 channel={args.channel} 사용")
    
    # 모델 초기화
    model = Dual(
        device=device,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout,
        d_llm=args.d_llm,
        e_layer=args.e_layers,
        d_layer=args.d_layers,
        d_ff=args.d_ff,
        head=args.n_heads,
        visualize=False
    )
    
    # alignment_combiner 입력 차원 확인 및 재생성 (방법 2)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_state = checkpoint["model"]
        # alignment_combiner.0.weight 확인
        alignment_key = None
        if "alignment_combiner.0.weight" in model_state:
            alignment_key = "alignment_combiner.0.weight"
        elif "_orig_mod.alignment_combiner.0.weight" in model_state:
            alignment_key = "_orig_mod.alignment_combiner.0.weight"
        
        if alignment_key:
            ckpt_input_dim = model_state[alignment_key].shape[1]  # 입력 차원
            ckpt_output_dim = model_state[alignment_key].shape[0]  # 출력 차원 (channel)
            
            if ckpt_input_dim != args.channel:
                print(f"⚠️  alignment_combiner 입력 차원 불일치 감지:")
                print(f"   - 체크포인트: 입력={ckpt_input_dim}, 출력={ckpt_output_dim}")
                print(f"   - 현재 모델: 입력={args.channel}, 출력={args.channel}")
                print(f"   - alignment_combiner를 재생성합니다...")
                
                # alignment_combiner 재생성
                model.alignment_combiner = nn.Sequential(
                    nn.Linear(ckpt_input_dim, ckpt_output_dim),
                    nn.LayerNorm(ckpt_output_dim),
                    nn.ReLU()
                ).to(device)
                
                print(f"   ✅ alignment_combiner 재생성 완료 (입력={ckpt_input_dim}, 출력={ckpt_output_dim})")
    
    # 체크포인트에서 ts_head/txt_head가 있는지 먼저 확인
    has_ts_head = False
    has_txt_head = False
    ts_head_dict = None
    txt_head_dict = None
    common_dim = getattr(args, 'common_dim', 64)  # 기본값
    
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            # train.py에서 저장한 형태: {"model": ..., "ts_head": ..., "txt_head": ...}
            if "ts_head" in checkpoint:
                has_ts_head = True
                ts_head_dict = checkpoint["ts_head"]
                # common_dim 추론: ts_head의 마지막 레이어 출력 차원 확인
                if "2.weight" in ts_head_dict:
                    common_dim = ts_head_dict["2.weight"].shape[0]
                    print(f"✅ 체크포인트에서 common_dim={common_dim} 추론 (ts_head 마지막 레이어)")
            if "txt_head" in checkpoint:
                has_txt_head = True
                txt_head_dict = checkpoint["txt_head"]
                # common_dim 추론: txt_head의 마지막 레이어 출력 차원 확인
                if "2.weight" in txt_head_dict:
                    inferred_common_dim = txt_head_dict["2.weight"].shape[0]
                    if not has_ts_head:  # ts_head가 없었으면 txt_head에서 추론
                        common_dim = inferred_common_dim
                        print(f"✅ 체크포인트에서 common_dim={common_dim} 추론 (txt_head 마지막 레이어)")
                    elif common_dim != inferred_common_dim:
                        print(f"⚠️  경고: ts_head와 txt_head의 common_dim이 다릅니다. ts_head: {common_dim}, txt_head: {inferred_common_dim}")
        elif "state_dict" in checkpoint:
            # 일반적인 체크포인트 형태
            if "ts_head_state_dict" in checkpoint or "ts_head" in checkpoint:
                ts_head_dict = checkpoint.get("ts_head_state_dict", checkpoint.get("ts_head", {}))
                if ts_head_dict:
                    has_ts_head = True
                    if "2.weight" in ts_head_dict:
                        common_dim = ts_head_dict["2.weight"].shape[0]
            if "txt_head_state_dict" in checkpoint or "txt_head" in checkpoint:
                txt_head_dict = checkpoint.get("txt_head_state_dict", checkpoint.get("txt_head", {}))
                if txt_head_dict:
                    has_txt_head = True
                    if "2.weight" in txt_head_dict:
                        inferred_common_dim = txt_head_dict["2.weight"].shape[0]
                        if not has_ts_head:
                            common_dim = inferred_common_dim
    
    # ts_head와 txt_head 생성 (체크포인트에 있으면 로드, 없으면 새로 생성)
    if has_ts_head:
        print(f"✅ 체크포인트에서 ts_head 발견 - 로드합니다")
        model.ts_head = nn.Sequential(
            nn.Linear(args.channel, 128), 
            nn.ReLU(),
            nn.Linear(128, common_dim)
        ).to(device)
    else:
        print(f"⚠️  체크포인트에 ts_head가 없습니다 - 새로 초기화합니다 (common_dim={common_dim})")
        model.ts_head = nn.Sequential(
            nn.Linear(args.channel, 128), 
            nn.ReLU(),
            nn.Linear(128, common_dim)
        ).to(device)
    
    if has_txt_head:
        print(f"✅ 체크포인트에서 txt_head 발견 - 로드합니다")
        model.txt_head = nn.Sequential(
            nn.Linear(args.d_llm, 128), 
            nn.ReLU(),
            nn.Linear(128, common_dim)
        ).to(device)
    else:
        print(f"⚠️  체크포인트에 txt_head가 없습니다 - 새로 초기화합니다 (common_dim={common_dim})")
        model.txt_head = nn.Sequential(
            nn.Linear(args.d_llm, 128), 
            nn.ReLU(),
            nn.Linear(128, common_dim)
        ).to(device)
    
    # 체크포인트에서 실제 사용된 파라미터 확인 (이미 위에서 로드한 checkpoint 사용)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_state = checkpoint["model"]
        
        # cross 모듈의 실제 shape 확인 (CrossModal의 d_model은 channel 사용)
        if "cross.layers.0.self_attn.W_Q.weight" in model_state or "_orig_mod.cross.layers.0.self_attn.W_Q.weight" in model_state:
            key = "cross.layers.0.self_attn.W_Q.weight" if "cross.layers.0.self_attn.W_Q.weight" in model_state else "_orig_mod.cross.layers.0.self_attn.W_Q.weight"
            actual_cross_dim = model_state[key].shape[0]
            if actual_cross_dim != args.channel:
                print(f"⚠️  경고: 체크포인트의 cross 모듈 d_model={actual_cross_dim}과 설정된 channel={args.channel}이 다릅니다.")
                print(f"   설정된 값(channel={args.channel})을 유지합니다.")
    
    # 저장된 모델 구조에 따라 다르게 처리
    model_missing_keys = []
    model_unexpected_keys = []
    ts_head_missing_keys = []
    ts_head_unexpected_keys = []
    txt_head_missing_keys = []
    txt_head_unexpected_keys = []
    
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            # train.py에서 저장한 형태: {"model": ..., "ts_head": ..., "txt_head": ...}
            # _orig_mod prefix 처리
            model_state_to_load = checkpoint["model"]
            if any("_orig_mod" in k for k in model_state_to_load.keys()):
                # _orig_mod prefix 제거
                model_state_to_load = {k.replace("_orig_mod.", ""): v for k, v in model_state_to_load.items()}
                print("   🔄 '_orig_mod' prefix 제거하여 로드합니다")
            
            result = model.load_state_dict(model_state_to_load, strict=False)
            model_missing_keys = result.missing_keys
            model_unexpected_keys = result.unexpected_keys
            
            if has_ts_head:
                result = model.ts_head.load_state_dict(ts_head_dict, strict=False)
                ts_head_missing_keys = result.missing_keys
                ts_head_unexpected_keys = result.unexpected_keys
            if has_txt_head:
                result = model.txt_head.load_state_dict(txt_head_dict, strict=False)
                txt_head_missing_keys = result.missing_keys
                txt_head_unexpected_keys = result.unexpected_keys
        elif "state_dict" in checkpoint:
            # 일반적인 체크포인트 형태: {"state_dict": ..., "epoch": ..., ...}
            # _orig_mod prefix 처리
            model_state_to_load = checkpoint["state_dict"]
            if any("_orig_mod" in k for k in model_state_to_load.keys()):
                # _orig_mod prefix 제거
                model_state_to_load = {k.replace("_orig_mod.", ""): v for k, v in model_state_to_load.items()}
                print("   🔄 '_orig_mod' prefix 제거하여 로드합니다")
            
            result = model.load_state_dict(model_state_to_load, strict=False)
            model_missing_keys = result.missing_keys
            model_unexpected_keys = result.unexpected_keys
            
            if has_ts_head:
                result = model.ts_head.load_state_dict(ts_head_dict, strict=False)
                ts_head_missing_keys = result.missing_keys
                ts_head_unexpected_keys = result.unexpected_keys
            if has_txt_head:
                result = model.txt_head.load_state_dict(txt_head_dict, strict=False)
                txt_head_missing_keys = result.missing_keys
                txt_head_unexpected_keys = result.unexpected_keys
        else:
            # 직접 state_dict만 저장된 형태
            # _orig_mod prefix 처리
            model_state_to_load = checkpoint
            if any("_orig_mod" in k for k in model_state_to_load.keys()):
                # _orig_mod prefix 제거
                model_state_to_load = {k.replace("_orig_mod.", ""): v for k, v in model_state_to_load.items()}
                print("   🔄 '_orig_mod' prefix 제거하여 로드합니다")
            
            result = model.load_state_dict(model_state_to_load, strict=False)
            model_missing_keys = result.missing_keys
            model_unexpected_keys = result.unexpected_keys
    else:
        # 직접 state_dict만 저장된 형태
        # _orig_mod prefix 처리
        model_state_to_load = checkpoint
        if isinstance(checkpoint, dict) and any("_orig_mod" in k for k in checkpoint.keys()):
            # _orig_mod prefix 제거
            model_state_to_load = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
            print("   🔄 '_orig_mod' prefix 제거하여 로드합니다")
        
        result = model.load_state_dict(model_state_to_load, strict=False)
        model_missing_keys = result.missing_keys
        model_unexpected_keys = result.unexpected_keys
    
    # 모델 로드 검증
    print("\n📋 모델 로드 검증:")
    print(f"   - 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    if has_ts_head:
        print(f"   - ts_head 파라미터 수: {sum(p.numel() for p in model.ts_head.parameters()):,}")
    if has_txt_head:
        print(f"   - txt_head 파라미터 수: {sum(p.numel() for p in model.txt_head.parameters()):,}")
    
    # 누락된 키 확인 (중요하지 않은 것들 제외)
    # ts_head와 txt_head는 별도로 저장되므로 missing_keys에서 제외
    ignore_patterns = ['_orig_mod', 'visualize', 'cache', 'ts_head', 'txt_head']
    important_missing = [k for k in model_missing_keys if not any(ignore in k for ignore in ignore_patterns)]
    if important_missing:
        print(f"⚠️  모델에서 누락된 중요 키 ({len(important_missing)}개):")
        for key in important_missing[:10]:  # 최대 10개만 출력
            print(f"      - {key}")
        if len(important_missing) > 10:
            print(f"      ... 외 {len(important_missing) - 10}개")
    else:
        # ts_head/txt_head만 누락된 경우 (정상)
        ts_txt_missing = [k for k in model_missing_keys if 'ts_head' in k or 'txt_head' in k]
        if ts_txt_missing and has_ts_head and has_txt_head:
            print(f"✅ ts_head/txt_head는 별도로 로드되었습니다 (누락 키 {len(ts_txt_missing)}개는 정상)")
    
    if model_unexpected_keys:
        print(f"⚠️  모델에서 예상치 못한 키 ({len(model_unexpected_keys)}개):")
        for key in model_unexpected_keys[:10]:  # 최대 10개만 출력
            print(f"      - {key}")
        if len(model_unexpected_keys) > 10:
            print(f"      ... 외 {len(model_unexpected_keys) - 10}개")
    
    if has_ts_head and ts_head_missing_keys:
        print(f"⚠️  ts_head에서 누락된 키 ({len(ts_head_missing_keys)}개): {ts_head_missing_keys}")
    if has_ts_head and ts_head_unexpected_keys:
        print(f"⚠️  ts_head에서 예상치 못한 키 ({len(ts_head_unexpected_keys)}개): {ts_head_unexpected_keys}")
    
    if has_txt_head and txt_head_missing_keys:
        print(f"⚠️  txt_head에서 누락된 키 ({len(txt_head_missing_keys)}개): {txt_head_missing_keys}")
    if has_txt_head and txt_head_unexpected_keys:
        print(f"⚠️  txt_head에서 예상치 못한 키 ({len(txt_head_unexpected_keys)}개): {txt_head_unexpected_keys}")
    
    # 체크포인트의 키 확인 및 파라미터 수 비교
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint_keys = set(checkpoint["model"].keys())
        model_keys = set(model.state_dict().keys())
        
        # ts_head/txt_head 제외하고 비교
        checkpoint_keys_no_heads = {k for k in checkpoint_keys if 'ts_head' not in k and 'txt_head' not in k}
        model_keys_no_heads = {k for k in model_keys if 'ts_head' not in k and 'txt_head' not in k}
        matched_keys = checkpoint_keys_no_heads & model_keys_no_heads
        
        print(f"\n📊 키 매칭 (ts_head/txt_head 제외):")
        print(f"   - 체크포인트 키 수: {len(checkpoint_keys_no_heads)}")
        print(f"   - 모델 키 수: {len(model_keys_no_heads)}")
        print(f"   - 매칭된 키 수: {len(matched_keys)}")
        if len(checkpoint_keys_no_heads) > 0:
            print(f"   - 매칭률: {len(matched_keys) / len(checkpoint_keys_no_heads) * 100:.1f}%")
        
        # 파라미터 수 비교
        checkpoint_param_count = sum(p.numel() for p in checkpoint["model"].values())
        model_param_count = sum(p.numel() for p in model.state_dict().values())
        # ts_head/txt_head 제외
        model_param_count_no_heads = sum(p.numel() for k, p in model.state_dict().items() 
                                         if 'ts_head' not in k and 'txt_head' not in k)
        
        print(f"\n📊 파라미터 수 비교:")
        print(f"   - 체크포인트 모델 파라미터 수: {checkpoint_param_count:,}")
        print(f"   - 로드된 모델 파라미터 수 (ts/txt_head 제외): {model_param_count_no_heads:,}")
        if checkpoint_param_count > 0:
            param_match_ratio = model_param_count_no_heads / checkpoint_param_count * 100
            print(f"   - 파라미터 매칭률: {param_match_ratio:.2f}%")
            if abs(param_match_ratio - 100.0) < 0.01:
                print(f"   ✅ 파라미터 수가 완벽하게 일치합니다!")
            elif param_match_ratio > 99.9:
                print(f"   ✅ 파라미터 수가 거의 일치합니다 (오차 < 0.1%)")
            else:
                print(f"   ⚠️  파라미터 수가 일치하지 않습니다")
        
        # ts_head/txt_head 파라미터 비교
        if has_ts_head and "ts_head" in checkpoint:
            ckpt_ts_head_params = sum(p.numel() for p in checkpoint["ts_head"].values())
            loaded_ts_head_params = sum(p.numel() for p in model.ts_head.state_dict().values())
            print(f"   - 체크포인트 ts_head 파라미터 수: {ckpt_ts_head_params:,}")
            print(f"   - 로드된 ts_head 파라미터 수: {loaded_ts_head_params:,}")
            if ckpt_ts_head_params == loaded_ts_head_params:
                print(f"   ✅ ts_head 파라미터 수 일치")
            else:
                print(f"   ⚠️  ts_head 파라미터 수 불일치")
        
        if has_txt_head and "txt_head" in checkpoint:
            ckpt_txt_head_params = sum(p.numel() for p in checkpoint["txt_head"].values())
            loaded_txt_head_params = sum(p.numel() for p in model.txt_head.state_dict().values())
            print(f"   - 체크포인트 txt_head 파라미터 수: {ckpt_txt_head_params:,}")
            print(f"   - 로드된 txt_head 파라미터 수: {loaded_txt_head_params:,}")
            if ckpt_txt_head_params == loaded_txt_head_params:
                print(f"   ✅ txt_head 파라미터 수 일치")
            else:
                print(f"   ⚠️  txt_head 파라미터 수 불일치")
        
        # _orig_mod prefix 처리 확인
        has_orig_mod = any("_orig_mod" in k for k in checkpoint_keys)
        if has_orig_mod:
            print(f"\n   ℹ️  체크포인트에 '_orig_mod' prefix가 있습니다 (torch.compile 사용됨)")
            print(f"   ✅ 자동으로 처리되었습니다")
    
    model.eval()
    
    print("\n✅ 모델 불러오기 완료")
    print(f"   - 디바이스: {device}")
    print(f"   - 공통 공간 차원: {common_dim}")
    print(f"   - ts_head 로드: {'✅' if has_ts_head else '❌ (새로 초기화)'}")
    print(f"   - txt_head 로드: {'✅' if has_txt_head else '❌ (새로 초기화)'}")
    
    return model

def load_data(args, flag='test'):
    """
    데이터셋 로드
    
    Args:
        args: argparse.Namespace 객체
        flag: 'train', 'val', 'test'
    
    Returns:
        dataset, data_loader: 데이터셋과 데이터로더
    """
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    
    dataset = data_class(
        flag=flag, 
        scale=True, 
        size=[args.seq_len, 0, args.pred_len], 
        data_path=args.data_path,
        num_nodes=args.num_nodes,
        stride=getattr(args, 'stride', 1)  # stride 파라미터 추가 (기본값: 1)
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=custom_collate
    )
    
    return dataset, data_loader

def extract_common_space_embeddings(model, data_loader, args, max_batches=None):
    """
    모델에 데이터를 통과시켜 공통 공간(64차원)으로 투영한 임베딩 추출
    
    Args:
        model: 로드된 모델
        data_loader: 데이터 로더
        args: 설정 인자
        max_batches: 최대 처리할 배치 수 (None이면 전체)
    
    Returns:
        dict: 공통 공간 임베딩 데이터 (last_emb_common, enc_emb_common, cr_emb_common, channel_labels)
    """
    device = next(model.parameters()).device
    model.eval()
    
    print("\n📊 공통 공간 임베딩 추출 중...")
    if max_batches is not None:
        print(f"   (앞에서부터 순서대로 최대 {max_batches}개 배치만 처리)")
    
    # 공통 공간 임베딩 수집 리스트
    last_emb_common_list = []
    enc_emb_common_list = []
    cr_emb_common_list = []
    # 채널 라벨 리스트 (각 타입별로 독립적으로)
    last_emb_channel_list = []
    enc_emb_channel_list = []
    cr_emb_channel_list = []
    # 순서 정보 추적 (각 타입별로 독립적으로)
    last_emb_order_list = []
    enc_emb_order_list = []
    cr_emb_order_list = []
    
    processed_batches = 0
    debug_mode = getattr(args, 'debug', False)
    common_dim = getattr(args, 'common_dim', 64)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="데이터 처리")):
            if max_batches is not None and processed_batches >= max_batches:
                break
            
            try:
                # 배치 데이터 언패킹
                if len(batch) == 8:
                    x, y, x_mark, y_mark, last_emb, full_prompt, anchor_avg, num_token_idx = batch
                elif len(batch) == 5:
                    x, y, x_mark, y_mark, embeddings = batch
                    last_emb = None
                    full_prompt = None
                    anchor_avg = None
                else:
                    x, y, x_mark, y_mark = batch
                    last_emb = None
                    full_prompt = None
                    anchor_avg = None
                
                if last_emb is None:
                    continue
                
                # 디바이스로 이동
                x = x.to(device, non_blocking=True).float()
                x_mark = x_mark.to(device, non_blocking=True).float()
                
                # last_emb shape 정규화: [B, d_llm, N] 형태로 변환
                if len(last_emb.shape) > 3:
                    last_emb = last_emb.squeeze(0)
                last_emb = last_emb.to(device, non_blocking=True).float()
                if len(last_emb.shape) == 2:
                    last_emb = last_emb.unsqueeze(0)
                
                if full_prompt is not None:
                    full_prompt = full_prompt.to(device, non_blocking=True).float()
                
                # get_embeddings 사용
                enc, pr, cr = model.get_embeddings(x, x_mark, last_emb, full_prompt)
                
                # cr shape 정규화: [B, N, C] 형태로 변환
                if len(cr.shape) == 2:
                    if cr.shape[0] == args.channel and cr.shape[1] == args.num_nodes:
                        cr = cr.T.unsqueeze(0)  # [C, N] -> [1, N, C]
                    else:
                        cr = cr.unsqueeze(0)  # [N, C] -> [1, N, C]
                elif len(cr.shape) == 3:
                    if cr.shape[1] == args.channel and cr.shape[2] == args.num_nodes:
                        cr = cr.permute(0, 2, 1)  # [B, C, N] -> [B, N, C]
                
                # enc shape 정규화: [B, N, C] 형태로 변환
                enc_orig_shape = enc.shape
                if len(enc.shape) == 3:
                    if enc.shape[1] == args.channel and enc.shape[2] == args.num_nodes:
                        enc = enc.permute(0, 2, 1)  # [B, C, N] -> [B, N, C]
                elif len(enc.shape) == 2:
                    # [N, C] 또는 [C, N] 형태
                    if enc.shape[0] == args.channel and enc.shape[1] == args.num_nodes:
                        enc = enc.T.unsqueeze(0)  # [C, N] -> [1, N, C]
                    elif enc.shape[0] == args.num_nodes and enc.shape[1] == args.channel:
                        enc = enc.unsqueeze(0)  # [N, C] -> [1, N, C]
                
                if debug_mode and batch_idx < 3:
                    print(f"   배치 {batch_idx} - enc shape: {enc_orig_shape} -> {enc.shape}")
                
                B = cr.shape[0] if len(cr.shape) == 3 else 1
                B_enc = enc.shape[0] if len(enc.shape) == 3 else 1
                B_last = last_emb.shape[0] if len(last_emb.shape) == 3 else 1
                B_common = max(B, B_enc, B_last)  # 모든 임베딩이 같은 배치 수를 가지도록
                
                if debug_mode and batch_idx < 3:
                    print(f"   배치 {batch_idx} - B: {B}, B_enc: {B_enc}, B_last: {B_last}, B_common: {B_common}")
                
                # 공통 공간으로 투영
                for b_idx in range(B_common):
                    # cr (cross embedding) → ts_head → 공통 공간
                    if b_idx < B:
                        cr_batch = cr[b_idx] if len(cr.shape) == 3 else cr.squeeze(0)  # [N, C]
                        if cr_batch.shape != (args.num_nodes, args.channel):
                            if cr_batch.shape == (args.channel, args.num_nodes):
                                cr_batch = cr_batch.T
                        
                        # ts_head로 공통 공간 투영
                        cr_common = model.ts_head(cr_batch)  # [N, common_dim]
                        cr_common = F.normalize(cr_common, dim=-1)  # [N, common_dim]
                        # 순서 정보 저장 (추가하기 전의 길이를 기준으로)
                        current_cr_order = len(cr_emb_common_list)
                        cr_emb_common_list.extend(cr_common.cpu().numpy())
                        cr_emb_order_list.extend([current_cr_order + i for i in range(args.num_nodes)])
                        # 채널 라벨 저장
                        cr_emb_channel_list.extend(list(range(args.num_nodes)))
                    
                    # enc (time-series embedding) → ts_head → 공통 공간
                    if b_idx < B_enc:
                        if len(enc.shape) == 3:
                            enc_batch = enc[b_idx]  # [N, C]
                        elif len(enc.shape) == 2:
                            enc_batch = enc  # [N, C]
                        else:
                            enc_batch = None
                        
                        if enc_batch is not None:
                            # shape 확인 및 정규화
                            if enc_batch.shape == (args.channel, args.num_nodes):
                                enc_batch = enc_batch.T  # [C, N] -> [N, C]
                            
                            if enc_batch.shape == (args.num_nodes, args.channel):
                                enc_common = model.ts_head(enc_batch)  # [N, common_dim]
                                enc_common = F.normalize(enc_common, dim=-1)  # [N, common_dim]
                                # 순서 정보 저장 (추가하기 전의 길이를 기준으로)
                                current_enc_order = len(enc_emb_common_list)
                                enc_emb_common_list.extend(enc_common.cpu().numpy())
                                enc_emb_order_list.extend([current_enc_order + i for i in range(args.num_nodes)])
                                # 채널 라벨 저장
                                enc_emb_channel_list.extend(list(range(args.num_nodes)))
                            elif debug_mode and batch_idx < 3:
                                print(f"   ⚠️  배치 {batch_idx}, b_idx {b_idx}: enc_batch shape 불일치: {enc_batch.shape}, 예상: ({args.num_nodes}, {args.channel})")
                        elif debug_mode and batch_idx < 3:
                            print(f"   ⚠️  배치 {batch_idx}, b_idx {b_idx}: enc_batch가 None입니다")
                    
                    # last_emb → txt_head → 공통 공간
                    if b_idx < B_last and len(last_emb.shape) == 3:
                        last_emb_batch = last_emb[b_idx]  # [d_llm, N]
                        last_emb_batch = last_emb_batch.permute(1, 0)  # [N, d_llm]
                        last_emb_common = model.txt_head(last_emb_batch)  # [N, common_dim]
                        last_emb_common = F.normalize(last_emb_common, dim=-1)  # [N, common_dim]
                        # 순서 정보 저장 (추가하기 전의 길이를 기준으로)
                        current_last_order = len(last_emb_common_list)
                        last_emb_common_list.extend(last_emb_common.cpu().numpy())
                        last_emb_order_list.extend([current_last_order + i for i in range(args.num_nodes)])
                        # 채널 라벨 저장
                        last_emb_channel_list.extend(list(range(args.num_nodes)))
                
                processed_batches += 1
                    
            except Exception as e:
                if debug_mode:
                    print(f"⚠️  배치 {batch_idx} 처리 중 오류: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
    
    # 처리된 배치 수 출력
    if processed_batches > 0:
        print(f"\n✅ 총 {processed_batches}개 배치 처리 완료")
        print(f"   - last_emb_common: {len(last_emb_common_list)}개")
        print(f"   - enc_emb_common: {len(enc_emb_common_list)}개")
        print(f"   - cr_emb_common: {len(cr_emb_common_list)}개")
    else:
        raise ValueError("처리된 배치가 없습니다. 데이터를 확인해주세요.")
    
    # 임베딩 리스트를 numpy 배열로 변환
    last_emb_common_array = np.array(last_emb_common_list) if len(last_emb_common_list) > 0 else None
    enc_emb_common_array = np.array(enc_emb_common_list) if len(enc_emb_common_list) > 0 else None
    cr_emb_common_array = np.array(cr_emb_common_list) if len(cr_emb_common_list) > 0 else None
    
    # 채널 라벨 배열 변환 (각 타입별로 독립적)
    last_emb_channel_array = np.array(last_emb_channel_list) if len(last_emb_channel_list) > 0 else None
    enc_emb_channel_array = np.array(enc_emb_channel_list) if len(enc_emb_channel_list) > 0 else None
    cr_emb_channel_array = np.array(cr_emb_channel_list) if len(cr_emb_channel_list) > 0 else None
    
    # 순서 정보를 numpy 배열로 변환
    last_emb_order_array = np.array(last_emb_order_list) if len(last_emb_order_list) > 0 else None
    enc_emb_order_array = np.array(enc_emb_order_list) if len(enc_emb_order_list) > 0 else None
    cr_emb_order_array = np.array(cr_emb_order_list) if len(cr_emb_order_list) > 0 else None
    
    return {
        'last_emb_common': last_emb_common_array,  # [N_samples, common_dim]
        'enc_emb_common': enc_emb_common_array,  # [N_samples, common_dim]
        'cr_emb_common': cr_emb_common_array,  # [N_samples, common_dim]
        'last_emb_channel': last_emb_channel_array,  # [N_samples] - last_emb 채널 라벨
        'enc_emb_channel': enc_emb_channel_array,  # [N_samples] - enc_emb 채널 라벨
        'cr_emb_channel': cr_emb_channel_array,  # [N_samples] - cr_emb 채널 라벨
        'last_emb_order': last_emb_order_array,  # [N_samples] - 순서 정보
        'enc_emb_order': enc_emb_order_array,  # [N_samples] - 순서 정보
        'cr_emb_order': cr_emb_order_array  # [N_samples] - 순서 정보
    }

def visualize_embeddings(embedding_data, args, save_path="embeddings_visualization.png", method='pca', max_samples=5000):
    """
    공통 공간 임베딩 시각화: last_emb_common, enc_emb_common, cr_emb_common를 2D로 축소하여 시각화
    
    Args:
        embedding_data: extract_common_space_embeddings의 반환값
        args: 설정 인자
        save_path: 저장 경로
        method: 차원 축소 방법 ('pca' 또는 'tsne')
        max_samples: 최대 샘플 수 (너무 많으면 앞에서부터 순서대로 자름)
    """
    print(f"\n🎨 임베딩 시각화 생성 중... (방법: {method})")
    print(f"   📁 저장 경로: {save_path}")
    
    last_emb_common = embedding_data.get('last_emb_common', None)
    enc_emb_common = embedding_data.get('enc_emb_common', None)
    cr_emb_common = embedding_data.get('cr_emb_common', None)
    # 채널 라벨 (각 타입별로 독립적)
    last_emb_channel = embedding_data.get('last_emb_channel', None)
    enc_emb_channel = embedding_data.get('enc_emb_channel', None)
    cr_emb_channel = embedding_data.get('cr_emb_channel', None)
    # 순서 정보
    last_emb_order = embedding_data.get('last_emb_order', None)
    enc_emb_order = embedding_data.get('enc_emb_order', None)
    cr_emb_order = embedding_data.get('cr_emb_order', None)
    
    print(f"   - last_emb_common: {last_emb_common.shape if last_emb_common is not None else None}")
    print(f"   - enc_emb_common: {enc_emb_common.shape if enc_emb_common is not None else None}")
    print(f"   - cr_emb_common: {cr_emb_common.shape if cr_emb_common is not None else None}")
    
    # 데이터 확인
    if last_emb_common is None and enc_emb_common is None and cr_emb_common is None:
        print("⚠️  시각화할 임베딩 데이터가 없습니다.")
        return
    
    # 모든 임베딩을 하나의 배열로 결합 (모두 common_dim 차원이므로 가능)
    all_embeddings = []
    embedding_types = []  # 각 임베딩의 타입 (0: last_emb, 1: enc_emb, 2: cr_emb)
    embedding_channels = []  # 각 임베딩의 채널 인덱스
    embedding_orders = []  # 각 임베딩의 순서 정보 (타입별로 독립적)
    
    if last_emb_common is not None:
        all_embeddings.append(last_emb_common)
        embedding_types.extend([0] * len(last_emb_common))
        if last_emb_channel is not None and len(last_emb_channel) == len(last_emb_common):
            embedding_channels.extend(last_emb_channel)
        else:
            embedding_channels.extend([-1] * len(last_emb_common))
        # 순서 정보 추가
        if last_emb_order is not None and len(last_emb_order) == len(last_emb_common):
            embedding_orders.extend(last_emb_order)
        else:
            # 순서 정보가 없으면 인덱스로 대체
            embedding_orders.extend(range(len(last_emb_common)))
    
    if enc_emb_common is not None:
        all_embeddings.append(enc_emb_common)
        embedding_types.extend([1] * len(enc_emb_common))
        if enc_emb_channel is not None and len(enc_emb_channel) == len(enc_emb_common):
            embedding_channels.extend(enc_emb_channel)
        else:
            embedding_channels.extend([-1] * len(enc_emb_common))
        # 순서 정보 추가
        if enc_emb_order is not None and len(enc_emb_order) == len(enc_emb_common):
            embedding_orders.extend(enc_emb_order)
        else:
            # 순서 정보가 없으면 인덱스로 대체
            embedding_orders.extend(range(len(enc_emb_common)))
    
    if cr_emb_common is not None:
        all_embeddings.append(cr_emb_common)
        embedding_types.extend([2] * len(cr_emb_common))
        if cr_emb_channel is not None and len(cr_emb_channel) == len(cr_emb_common):
            embedding_channels.extend(cr_emb_channel)
        else:
            embedding_channels.extend([-1] * len(cr_emb_common))
        # 순서 정보 추가
        if cr_emb_order is not None and len(cr_emb_order) == len(cr_emb_common):
            embedding_orders.extend(cr_emb_order)
        else:
            # 순서 정보가 없으면 인덱스로 대체
            embedding_orders.extend(range(len(cr_emb_common)))
    
    # 각 타입별 개수 확인
    n_last = len(last_emb_common) if last_emb_common is not None else 0
    n_enc = len(enc_emb_common) if enc_emb_common is not None else 0
    n_cr = len(cr_emb_common) if cr_emb_common is not None else 0
    
    # 사용 가능한 타입 수 계산
    num_types = sum([n_last > 0, n_enc > 0, n_cr > 0])
    samples_per_channel = 100  # 각 타입-채널 조합당 샘플 수
    
    print(f"   - 전체 임베딩 수: {n_last + n_enc + n_cr}")
    print(f"   - last_emb: {n_last}개, enc_emb: {n_enc}개, cr_emb: {n_cr}개")
    print(f"   - 사용 가능한 타입 수: {num_types}")
    print(f"   - 각 타입-채널 조합당 샘플 수: {samples_per_channel}개")
    print(f"   - 예상 총 샘플 수: {num_types * args.num_nodes * samples_per_channel}개")
    
    # 각 타입-채널 조합별로 샘플링
    sampled_embeddings = []
    sampled_types = []
    sampled_channels = []
    sampled_orders = []  # 각 타입-채널 조합 내에서의 순서 (0~99)
    
    # last_emb 샘플링 (각 채널별로)
    if n_last > 0 and last_emb_channel is not None:
        for ch_idx in range(args.num_nodes):
            ch_mask = last_emb_channel == ch_idx
            ch_indices = np.where(ch_mask)[0]
            
            if len(ch_indices) > 0:
                n_sample = min(len(ch_indices), samples_per_channel)
                selected_indices = ch_indices[:n_sample]
                
                sampled_embeddings.append(last_emb_common[selected_indices])
                sampled_types.extend([0] * n_sample)
                sampled_channels.extend([ch_idx] * n_sample)
                # 각 타입-채널 조합 내에서 순서를 0부터 시작하도록 재설정
                sampled_orders.extend(range(n_sample))
                
                if len(ch_indices) > samples_per_channel:
                    print(f"   - last_emb 채널{ch_idx}: {len(ch_indices)} -> {n_sample} (앞에서부터)")
    
    # enc_emb 샘플링 (각 채널별로)
    if n_enc > 0 and enc_emb_channel is not None:
        for ch_idx in range(args.num_nodes):
            ch_mask = enc_emb_channel == ch_idx
            ch_indices = np.where(ch_mask)[0]
            
            if len(ch_indices) > 0:
                n_sample = min(len(ch_indices), samples_per_channel)
                selected_indices = ch_indices[:n_sample]
                
                sampled_embeddings.append(enc_emb_common[selected_indices])
                sampled_types.extend([1] * n_sample)
                sampled_channels.extend([ch_idx] * n_sample)
                # 각 타입-채널 조합 내에서 순서를 0부터 시작하도록 재설정
                sampled_orders.extend(range(n_sample))
                
                if len(ch_indices) > samples_per_channel:
                    print(f"   - enc_emb 채널{ch_idx}: {len(ch_indices)} -> {n_sample} (앞에서부터)")
    
    # cr_emb 샘플링 (각 채널별로)
    if n_cr > 0 and cr_emb_channel is not None:
        for ch_idx in range(args.num_nodes):
            ch_mask = cr_emb_channel == ch_idx
            ch_indices = np.where(ch_mask)[0]
            
            if len(ch_indices) > 0:
                n_sample = min(len(ch_indices), samples_per_channel)
                selected_indices = ch_indices[:n_sample]
                
                sampled_embeddings.append(cr_emb_common[selected_indices])
                sampled_types.extend([2] * n_sample)
                sampled_channels.extend([ch_idx] * n_sample)
                # 각 타입-채널 조합 내에서 순서를 0부터 시작하도록 재설정
                sampled_orders.extend(range(n_sample))
                
                if len(ch_indices) > samples_per_channel:
                    print(f"   - cr_emb 채널{ch_idx}: {len(ch_indices)} -> {n_sample} (앞에서부터)")
    
    # 샘플링된 임베딩 결합
    if len(sampled_embeddings) > 0:
        all_embeddings = np.vstack(sampled_embeddings)
        embedding_types = np.array(sampled_types)
        embedding_channels = np.array(sampled_channels)
        embedding_orders = np.array(sampled_orders)
    else:
        # 샘플링 실패 시 기존 방식 유지 (전체 사용)
        if len(all_embeddings) > 0:
            all_embeddings = np.vstack(all_embeddings)
            embedding_types = np.array(embedding_types)
            embedding_channels = np.array(embedding_channels)
            embedding_orders = np.array(embedding_orders)
        else:
            raise ValueError("시각화할 임베딩 데이터가 없습니다.")
    
    print(f"   - 샘플링 후 전체 임베딩 수: {len(all_embeddings)}")
    print(f"   - 임베딩 차원: {all_embeddings.shape[1]} (공통 공간)")
    
    # 차원 축소
    print(f"   - {method.upper()}로 2D 축소 중...")
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(all_embeddings)
        explained_var = reducer.explained_variance_ratio_
        print(f"   - 설명된 분산: {explained_var[0]:.2%}, {explained_var[1]:.2%} (총 {sum(explained_var):.2%})")
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = reducer.fit_transform(all_embeddings)
    else:
        raise ValueError(f"지원하지 않는 방법: {method}")
    
    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 임베딩 타입별 색상
    colors = ['red', 'blue', 'green']  # last_emb: red, enc_emb: blue, cr_emb: green
    type_labels = ['last_emb (text)', 'enc_emb (time-series)', 'cr_emb (cross)']
    
    # 채널별 마커 (0-6번 채널에 대해 다른 마커 사용)
    markers = ['o', 's', '^', 'v', 'D', 'p', '*']  # 원, 사각형, 삼각형, 역삼각형, 다이아몬드, 오각형, 별
    
    # 각 임베딩 타입별로 플롯
    for emb_type in [0, 1, 2]:
        mask = embedding_types == emb_type
        if not np.any(mask):
            continue
        
        type_embeddings = embeddings_2d[mask]
        type_channels = embedding_channels[mask]
        type_orders = embedding_orders[mask]  # 해당 타입의 순서 정보 (각 타입-채널 조합 내에서 0~99)
        
        # 각 채널별로 플롯
        for ch_idx in range(args.num_nodes):
            ch_mask = type_channels == ch_idx
            if not np.any(ch_mask):
                continue
            
            ch_embeddings = type_embeddings[ch_mask]
            ch_orders = type_orders[ch_mask]  # 해당 타입-채널 조합의 순서 정보 (0~99)
            
            # 각 타입-채널 조합 내에서 순서에 따른 alpha 값 계산
            if method.lower() == 'tsne':
                # 해당 타입-채널 조합 내에서의 최대 순서
                max_order = ch_orders.max() if len(ch_orders) > 0 else 1
                min_order = ch_orders.min() if len(ch_orders) > 0 else 0
                
                # alpha 범위: 앞쪽(진함) 0.9 ~ 뒤쪽(흐림) 0.1
                alpha_min = 0.1
                alpha_max = 0.9
                
                if max_order > min_order:
                    # 선형적으로 감소: 앞쪽이 진하고 뒤로 갈수록 흐림
                    ch_alphas = alpha_max - (ch_orders - min_order) / (max_order - min_order) * (alpha_max - alpha_min)
                else:
                    # 순서가 모두 같으면 모두 진하게
                    ch_alphas = np.full(len(ch_orders), alpha_max)
            else:
                # PCA일 때는 고정 alpha
                ch_alphas = np.full(len(ch_orders), 0.6)
            
            marker = markers[ch_idx % len(markers)]
            color = colors[emb_type]
            # 레이블: 타입별로 첫 번째 채널만 표시하거나, 모든 채널을 표시
            if ch_idx == 0:
                label = type_labels[emb_type]
            else:
                label = f"{type_labels[emb_type]} (ch{ch_idx})"
            
            # scatter plot에 alpha 배열 직접 적용 (효율적)
            ax.scatter(ch_embeddings[:, 0], ch_embeddings[:, 1], 
                      c=color, marker=marker, s=30, alpha=ch_alphas, 
                      label=label,
                      edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'Common Space Embedding Visualization ({method.upper()})', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 임베딩 시각화 저장 완료: {save_path}")
    plt.close()

def compute_text_prototype_embeddings(text_prototypes, model, args):
    """
    텍스트 프로토타입을 공통 공간 임베딩으로 변환
    
    Args:
        text_prototypes: 텍스트 프로토타입 리스트 (예: ['trend', 'increase', ...])
        model: 로드된 모델
        args: 설정 인자
    
    Returns:
        torch.Tensor: 공통 공간 텍스트 임베딩 [num_prototypes, common_dim]
    """
    device = next(model.parameters()).device
    
    # GPT2 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_model = GPT2Model.from_pretrained("gpt2").to(device)
    gpt2_model.eval()
    
    print(f"\n📝 텍스트 프로토타입 임베딩 생성 중... ({len(text_prototypes)}개)")
    
    # 텍스트를 토크나이즈하고 임베딩 생성
    text_embeddings = []
    with torch.no_grad():
        for text in text_prototypes:
            # 토크나이즈
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # GPT2로 임베딩 생성 (마지막 토큰의 hidden state 사용)
            outputs = gpt2_model(**tokens)
            last_hidden = outputs.last_hidden_state  # [1, seq_len, d_llm]
            # 마지막 토큰의 임베딩 사용
            text_emb = last_hidden[0, -1, :]  # [d_llm]
            text_embeddings.append(text_emb)
    
    # 텍스트 임베딩을 텐서로 변환
    text_embeddings = torch.stack(text_embeddings, dim=0)  # [num_prototypes, d_llm]
    
    # txt_head로 공통 공간 투영
    text_common = model.txt_head(text_embeddings)  # [num_prototypes, common_dim]
    text_common = F.normalize(text_common, dim=-1)  # [num_prototypes, common_dim]
    
    print(f"   ✅ 텍스트 프로토타입 임베딩 생성 완료: {text_common.shape}")
    
    # 메모리 정리
    del gpt2_model, tokenizer
    torch.cuda.empty_cache()
    
    return text_common

def compute_attention_map(text_prototypes, model, data_loader, args, random_seed=42):
    """
    텍스트 프로토타입과 시계열 임베딩 간 attention score 계산 및 시각화
    
    Args:
        text_prototypes: 텍스트 프로토타입 리스트
        model: 로드된 모델
        data_loader: 데이터 로더
        args: 설정 인자
        random_seed: 랜덤 시드
    
    Returns:
        attention_scores: attention score 행렬 [num_prototypes, num_channels]
        text_prototypes: 텍스트 프로토타입 리스트
        batch_idx: 선택된 배치 인덱스
    """
    device = next(model.parameters()).device
    model.eval()
    
    print(f"\n🎯 Attention Map 계산 중...")
    print(f"   - 텍스트 프로토타입: {text_prototypes}")
    
    # 텍스트 프로토타입을 공통 공간 임베딩으로 변환
    text_common = compute_text_prototype_embeddings(text_prototypes, model, args)  # [num_prototypes, common_dim]
    
    # 랜덤으로 시계열 데이터 선택
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 데이터 로더에서 랜덤 배치 선택
    all_batches = list(data_loader)
    if len(all_batches) == 0:
        raise ValueError("데이터 로더에 배치가 없습니다.")
    
    random_batch_idx = np.random.randint(0, len(all_batches))
    batch = all_batches[random_batch_idx]
    
    print(f"   - 선택된 배치 인덱스: {random_batch_idx}")
    
    # 배치 데이터 언패킹
    if len(batch) == 8:
        x, y, x_mark, y_mark, last_emb, full_prompt, anchor_avg, num_token_idx = batch
    else:
        x, y, x_mark, y_mark = batch
        last_emb = None
        full_prompt = None
    
    if last_emb is None:
        raise ValueError("선택된 배치에 last_emb가 없습니다.")
    
    # 디바이스로 이동
    x = x.to(device).float()
    x_mark = x_mark.to(device).float()
    last_emb = last_emb.squeeze(0) if len(last_emb.shape) > 3 else last_emb
    last_emb = last_emb.to(device).float()
    if len(last_emb.shape) == 2:
        last_emb = last_emb.unsqueeze(0)
    
    # 모든 채널 선택
    print(f"   - 모든 채널 사용: 0~{args.num_nodes-1}")
    
    # get_embeddings로 cross embedding 추출
    with torch.no_grad():
        enc, pr, cr = model.get_embeddings(x, x_mark, last_emb, full_prompt)
    
    # cr shape 정규화: [B, N, C] 형태로 변환
    if len(cr.shape) == 2:
        if cr.shape[0] == args.channel and cr.shape[1] == args.num_nodes:
            cr = cr.T.unsqueeze(0)  # [C, N] -> [1, N, C]
        else:
            cr = cr.unsqueeze(0)  # [N, C] -> [1, N, C]
    elif len(cr.shape) == 3:
        if cr.shape[1] == args.channel and cr.shape[2] == args.num_nodes:
            cr = cr.permute(0, 2, 1)  # [B, C, N] -> [B, N, C]
    
    # 첫 번째 배치의 모든 채널 임베딩 사용
    B = cr.shape[0] if len(cr.shape) == 3 else 1
    cr_batch = cr[0] if len(cr.shape) == 3 else cr.squeeze(0)  # [N, C]
    if cr_batch.shape != (args.num_nodes, args.channel):
        if cr_batch.shape == (args.channel, args.num_nodes):
            cr_batch = cr_batch.T
    
    # 모든 채널의 임베딩을 공통 공간으로 투영
    # cr_batch: [N, C] -> 각 채널별로 ts_head 적용
    ts_common_list = []
    for ch_idx in range(args.num_nodes):
        cr_channel = cr_batch[ch_idx, :]  # [C]
        cr_channel = cr_channel.unsqueeze(0)  # [1, C]
        ts_common_ch = model.ts_head(cr_channel)  # [1, common_dim]
        ts_common_ch = F.normalize(ts_common_ch, dim=-1)  # [1, common_dim]
        ts_common_list.append(ts_common_ch)
    
    # 모든 채널의 임베딩을 스택
    ts_common = torch.cat(ts_common_list, dim=0)  # [N, common_dim]
    
    print(f"   - 시계열 임베딩 shape: {ts_common.shape}")
    
    # Attention score 계산 (cosine similarity)
    # text_common: [num_prototypes, common_dim]
    # ts_common: [N, common_dim]
    # attention_scores: [num_prototypes, N]
    attention_scores = F.cosine_similarity(
        text_common.unsqueeze(1),  # [num_prototypes, 1, common_dim]
        ts_common.unsqueeze(0),    # [1, N, common_dim]
        dim=-1
    )  # [num_prototypes, N]
    
    print(f"   - Attention scores shape: {attention_scores.shape}")
    
    return attention_scores.detach().cpu().numpy(), text_prototypes, random_batch_idx

def visualize_attention_map(attention_scores, text_prototypes, batch_idx, save_path="attention_map.png"):
    """
    Attention map 시각화 (10개 텍스트 프로토타입 × 7개 채널)
    
    Args:
        attention_scores: attention score 행렬 [num_prototypes, num_channels]
        text_prototypes: 텍스트 프로토타입 리스트
        batch_idx: 선택된 배치 인덱스
        save_path: 저장 경로
    """
    print(f"\n🎨 Attention Map 시각화 생성 중...")
    
    num_prototypes, num_channels = attention_scores.shape
    print(f"   - Attention scores shape: [{num_prototypes}, {num_channels}]")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # 1. Heatmap (10 × 7)
    im = ax1.imshow(attention_scores, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
    ax1.set_xticks(range(num_channels))
    ax1.set_xticklabels([f'Ch{i}' for i in range(num_channels)], fontsize=11)
    ax1.set_yticks(range(num_prototypes))
    ax1.set_yticklabels(text_prototypes, fontsize=11)
    ax1.set_xlabel('Channels', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Text Prototypes', fontsize=12, fontweight='bold')
    ax1.set_title(f'Attention Map: Text Prototypes vs Channels\n(Batch {batch_idx})', 
                 fontsize=14, fontweight='bold')
    
    # 각 셀에 값 표시
    for i in range(num_prototypes):
        for j in range(num_channels):
            score = attention_scores[i, j]
            text = ax1.text(j, i, f'{score:.2f}',
                         ha="center", va="center", 
                         color="white" if score < 0 else "black",
                         fontsize=9, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Cosine Similarity (Attention Score)', fontsize=11)
    
    # 2. 각 채널별 평균 attention score (Bar plot)
    channel_means = attention_scores.mean(axis=0)  # [num_channels]
    colors = plt.cm.viridis((channel_means - channel_means.min()) / (channel_means.max() - channel_means.min() + 1e-8))
    bars = ax2.bar(range(num_channels), channel_means, color=colors)
    ax2.set_xticks(range(num_channels))
    ax2.set_xticklabels([f'Ch{i}' for i in range(num_channels)], fontsize=11)
    ax2.set_ylabel('Mean Attention Score', fontsize=12)
    ax2.set_xlabel('Channels', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Attention Score per Channel', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 각 막대에 값 표시
    for i, (bar, score) in enumerate(zip(bars, channel_means)):
        ax2.text(i, score, f' {score:.3f}', va='bottom' if score >= 0 else 'top', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Attention Map 저장 완료: {save_path}")
    plt.close()
    
    # 통계 정보 출력
    print(f"\n📊 Attention Score 통계:")
    print(f"   - 전체 평균: {attention_scores.mean():.4f}")
    print(f"   - 전체 최대: {attention_scores.max():.4f}")
    print(f"   - 전체 최소: {attention_scores.min():.4f}")
    print(f"\n   채널별 평균:")
    for ch_idx in range(num_channels):
        print(f"   - Channel {ch_idx}: {channel_means[ch_idx]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='학습된 모델 로드 및 채널 간 similarity distribution 시각화')
    
    # 모델 경로
    parser.add_argument('--model_path', type=str, required=True, 
                       help='저장된 모델 체크포인트 경로 (예: logs/.../best_model.pth)')
    
    # 데이터 설정
    parser.add_argument('--data_path', type=str, default='ETTh1', help='데이터셋 이름')
    parser.add_argument('--data_flag', type=str, default='test', choices=['train', 'val', 'test'],
                       help='사용할 데이터셋 분할')
    
    # 모델 설정 (train.py와 동일한 설정 사용)
    parser.add_argument('--seq_len', type=int, default=96, help='입력 시퀀스 길이')
    parser.add_argument('--pred_len', type=int, default=96, help='예측 시퀀스 길이')
    parser.add_argument('--channel', type=int, default=32, help='채널 수')
    parser.add_argument('--num_nodes', type=int, default=7, help='노드 수')
    parser.add_argument('--stride', type=int, default=1, help='시계열 슬라이딩 윈도우 stride (기본값: 1)')
    parser.add_argument('--dropout', type=float, default=0.05, help='드롭아웃')
    parser.add_argument('--d_llm', type=int, default=768, help='LLM 임베딩 차원')
    parser.add_argument('--e_layers', type=int, default=1, help='인코더 레이어 수')
    parser.add_argument('--d_layers', type=int, default=1, help='디코더 레이어 수')
    parser.add_argument('--d_ff', type=int, default=32, help='피드포워드 차원')
    parser.add_argument('--n_heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--common_dim', type=int, default=64, help='공통 공간 차원')
    
    # 기타 설정
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='디바이스 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='배치 크기')
    parser.add_argument('--max_batches', type=int, default=None, 
                       help='최대 처리할 배치 수 (기본값: None=전체, 디버깅: 10 추천)')
    parser.add_argument('--save_path', type=str, default='channel_similarity_distribution.png',
                       help='시각화 저장 경로')
    parser.add_argument('--debug', action='store_true', 
                       help='디버깅 정보 출력 (설정 시 max_batches=10으로 자동 설정)')
    
    # 시각화 플래그
    parser.add_argument('--visualize_similarity', action='store_true',
                       help='Similarity distribution 시각화 수행 (기본값: False, 임베딩만 수집)')
    parser.add_argument('--visualize_embeddings', action='store_true',
                       help='임베딩 시각화 수행 (PCA/t-SNE)')
    parser.add_argument('--embedding_method', type=str, default='pca', choices=['pca', 'tsne', 'both'],
                       help='임베딩 시각화 방법 (pca/tsne/both, 기본값: pca)')
    parser.add_argument('--max_embedding_samples', type=int, default=1000,
                       help='임베딩 시각화 최대 샘플 수 (앞에서부터 순서대로, 기본값: 1000)')
    
    # Attention map 옵션
    parser.add_argument('--visualize_attention', action='store_true',
                       help='텍스트 프로토타입과 시계열 임베딩 간 attention map 시각화')
    parser.add_argument('--text_prototypes', type=str, nargs='+', 
                       default=['trend', 'increase', 'decay', 'variance', 'high', 'low', 'period', 'pattern', 'correlation', 'stationary'],
                       help='텍스트 프로토타입 리스트 (기본값: trend increase decay variance high low period pattern correlation stationary)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    
    args = parser.parse_args()
    
    # 디버그 모드일 때 max_batches 자동 설정
    if args.debug:
        if args.max_batches is None:
            args.max_batches = 10
            print("🔧 디버그 모드: max_batches=10으로 자동 설정")
        else:
            print(f"🔧 디버그 모드 활성화 (max_batches={args.max_batches})")
    
    # max_batches 정보 출력
    if args.max_batches is not None:
        print(f"🔧 디버깅 모드: 최대 {args.max_batches}개 배치만 처리합니다")
    else:
        print(f"📊 전체 데이터 처리 모드")
    
    print("🚀 모델 로드 및 채널 간 Similarity Distribution 계산 시작")
    print(f"📁 모델 경로: {args.model_path}")
    print(f"📊 데이터셋: {args.data_path} ({args.data_flag})")
    print(f"💻 디바이스: {args.device}")
    print(f"🔧 Stride: {getattr(args, 'stride', 1)}")
    print("-" * 50)
    
    # 모델이 존재하는지 확인
    if not os.path.exists(args.model_path):
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {args.model_path}")
        return
    
    # 모델 불러오기
    try:
        # 먼저 데이터를 로드하여 shape 확인
        print(f"\n📊 {args.data_flag} 데이터셋 로드 중...")
        stride_value = getattr(args, 'stride', 1)
        print(f"   - Stride: {stride_value}")
        dataset, data_loader = load_data(args, flag=args.data_flag)
        print(f"   - 데이터셋 크기: {len(dataset)} 샘플 (stride={stride_value} 적용 후)")
        
        # 🔍 실제 .h5 파일 구조 확인 (임시 디버깅)
        import h5py
        import json
        test_h5_path = f"./MY/Embeddings/{args.data_flag}/test/0.h5"
        if os.path.exists(test_h5_path):
            print(f"\n🔍 실제 .h5 파일 구조 확인: {test_h5_path}")
            print("=" * 70)
            with h5py.File(test_h5_path, 'r') as hf:
                print("🔑 저장된 키 목록:")
                for key in sorted(hf.keys()):
                    print(f"   - {key}")
                print()
                
                for key in sorted(hf.keys()):
                    dataset = hf[key]
                    print(f"📊 {key}:")
                    print(f"   Shape: {dataset.shape}")
                    print(f"   Dtype: {dataset.dtype}")
                    print(f"   Size: {dataset.size:,} elements")
                    
                    # 메모리 크기 계산
                    if dataset.dtype == 'float32':
                        size_mb = dataset.size * 4 / (1024**2)
                    elif dataset.dtype == 'float64':
                        size_mb = dataset.size * 8 / (1024**2)
                    else:
                        size_mb = dataset.size * dataset.dtype.itemsize / (1024**2)
                    print(f"   Memory: ~{size_mb:.2f} MB")
                    
                    # JSON 데이터인 경우
                    if 'json' in key.lower():
                        try:
                            content = json.loads(dataset[()].decode())
                            if isinstance(content, list):
                                print(f"   Type: JSON list, Length: {len(content)}")
                                if len(content) > 0 and isinstance(content[0], list):
                                    print(f"   Structure: [{len(content)}][{len(content[0]) if content[0] else 0}]...")
                                    if len(content[0]) > 0 and isinstance(content[0][0], list):
                                        print(f"   Deep: [{len(content)}][{len(content[0])}][{len(content[0][0]) if content[0][0] else 0}]...")
                        except:
                            print(f"   Type: JSON bytes")
                    else:
                        # 샘플 값
                        if len(dataset.shape) <= 2:
                            print(f"   Sample (first few): {dataset[:min(3, dataset.shape[0])]}")
                        elif len(dataset.shape) == 3:
                            print(f"   Sample shape breakdown: [{dataset.shape[0]}, {dataset.shape[1]}, {dataset.shape[2]}]")
                        elif len(dataset.shape) == 4:
                            print(f"   Sample shape breakdown: [{dataset.shape[0]}, {dataset.shape[1]}, {dataset.shape[2]}, {dataset.shape[3]}]")
                    print()
            print("=" * 70 + "\n")
        
        # 첫 번째 배치를 가져와서 shape 확인
        sample_batch = next(iter(data_loader))
        print(f"sample_batch's shape:{len(sample_batch)}")
        if len(sample_batch) >= 4:
            sample_x = sample_batch[0]  # [B, L, N]
            print(f"   - 샘플 입력 shape: {sample_x.shape}")
        else:
            sample_x = None
            print(f"   ⚠️  샘플 데이터를 가져올 수 없습니다.")
        
        # 데이터 shape를 확인한 후 모델 로드
        model = load_trained_model(args.model_path, args, sample_x=sample_x)
        
        # 공통 공간 임베딩 추출
        print("\n" + "="*50)
        print("📊 공통 공간 임베딩 추출 시작")
        print("="*50)
        embedding_data = extract_common_space_embeddings(
            model, data_loader, args, max_batches=args.max_batches
        )
        print(f"\n✅ 공통 공간 임베딩 추출 완료!")
        
        # 임베딩 시각화
        print("\n" + "="*50)
        print("🎨 임베딩 시각화 시작")
        print("="*50)
        embedding_method = getattr(args, 'embedding_method', 'pca')
        max_samples = getattr(args, 'max_embedding_samples', 5000)
        
        print(f"   - 시각화 방법: {embedding_method}")
        print(f"   - 최대 샘플 수: {max_samples}")
        
        # 저장 경로를 절대 경로로 변환
        current_dir = os.getcwd()
        print(f"   - 저장 디렉토리: {current_dir}")
        
        if embedding_method in ['pca', 'both']:
            try:
                pca_path = os.path.join(current_dir, 'embeddings_pca.png')
                print(f"\n   📊 PCA 시각화 생성 중...")
                print(f"   📁 저장 경로: {pca_path}")
                visualize_embeddings(
                    embedding_data, args, 
                    save_path=pca_path, 
                    method='pca', 
                    max_samples=max_samples
                )
                print(f"   ✅ PCA 시각화 완료: {pca_path}")
            except Exception as e:
                print(f"   ❌ PCA 시각화 중 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if embedding_method in ['tsne', 'both']:
            try:
                tsne_path = os.path.join(current_dir, 'embeddings_tsne.png')
                print(f"\n   📊 t-SNE 시각화 생성 중...")
                print(f"   📁 저장 경로: {tsne_path}")
                visualize_embeddings(
                    embedding_data, args, 
                    save_path=tsne_path, 
                    method='tsne', 
                    max_samples=max_samples
                )
                print(f"   ✅ t-SNE 시각화 완료: {tsne_path}")
            except Exception as e:
                print(f"   ❌ t-SNE 시각화 중 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Attention map 시각화
        if args.visualize_attention:
            print("\n" + "="*50)
            print("🎯 Attention Map 계산 및 시각화 시작")
            print("="*50)
            try:
                # 1. 기존 텍스트 프로토타입에 대한 attention map
                attention_scores, text_prototypes, batch_idx = compute_attention_map(
                    args.text_prototypes, model, data_loader, args, random_seed=args.random_seed
                )
                
                attention_path = os.path.join(current_dir, 'attention_map.png')
                visualize_attention_map(
                    attention_scores, text_prototypes, batch_idx, 
                    save_path=attention_path
                )
                print(f"\n✅ Attention Map 시각화 완료: {attention_path}")
                
                # 2. 시계열과 관련 없는 토큰들에 대한 attention map
                unrelated_tokens = [
                    "elephant",      # 코끼리 (동물)
                    "rainbow",       # 무지개 (색깔/자연 현상)
                    "pizza",         # 피자 (음식)
                    "happiness",     # 행복 (감정)
                    "basketball",    # 농구 (스포츠)
                    "guitar",        # 기타 (악기)
                    "novel",         # 소설 (문학)
                    "chemistry",     # 화학 (과학 분야)
                    "ocean",         # 바다 (지리/자연)
                    "butterfly"      # 나비 (곤충/자연)
                ]
                
                print("\n" + "="*50)
                print("🎯 시계열과 관련 없는 토큰들에 대한 Attention Map 계산 및 시각화 시작")
                print("="*50)
                print(f"   - 토큰 리스트: {unrelated_tokens}")
                
                attention_scores_unrelated, text_prototypes_unrelated, batch_idx_unrelated = compute_attention_map(
                    unrelated_tokens, model, data_loader, args, random_seed=args.random_seed
                )
                
                attention_path_unrelated = os.path.join(current_dir, 'attention_map_unrelated.png')
                visualize_attention_map(
                    attention_scores_unrelated, text_prototypes_unrelated, batch_idx_unrelated, 
                    save_path=attention_path_unrelated
                )
                print(f"\n✅ 시계열과 관련 없는 토큰들의 Attention Map 시각화 완료: {attention_path_unrelated}")
                
            except Exception as e:
                print(f"❌ Attention Map 계산 중 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*50)
        print("✅ 모든 작업 완료!")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 오류 발생")
        print(f"   오류 메시지: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
