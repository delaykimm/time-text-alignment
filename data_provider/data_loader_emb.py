import os, re, glob, json, h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path="./dataset/", flag='train', size=None, 
                 features='M', data_path='ETTh1', num_nodes=7,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h',
                 model_name="gpt2", stride=1, batch_size = 32):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.timeenc = timeenc
        self.num_nodes = num_nodes
        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = batch_size # 배치크기 저장
        self.samples_per_file = 32 # 각 파일에 들어있는 샘플 수 

        # Append '.csv' if not present
        if not data_path.endswith('.csv'):
            data_path += '.csv'

        # Normalize root path
        root_path = os.path.abspath(root_path)

        # If data_path is absolute, use as-is. Else, join with root_path
        if os.path.isabs(data_path):
            self.data_path = data_path
        else:
            self.data_path = os.path.normpath(os.path.join(root_path, data_path))

        # Save file name without extension (for embed path)
        self.data_path_file = os.path.splitext(os.path.basename(self.data_path))[0]

        self.model_name = model_name
        self.stride = stride
        self.embed_path = f"./MY/Embeddings/{self.data_path_file}/{flag}/"
        #self.embed_path = f"/NAS/jiyun/MY/{self.data_path_file}/{flag}/"

        self.__read_data__()
        
        # 임베딩 경로는 절대경로로 고정(상대경로 혼동 방지)
        #self.embed_path = os.path.abspath(f"./MY/Embeddings/{self.data_path_file}/{flag}/")
        
        # ✅ 여기서 유효 인덱스 수집
        self._build_valid_ids()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        # 👇 이 길이를 넘어서는 시작 인덱스는 시계열 슬라이딩에서 불가
        self.max_start = len(self.data_x) - self.seq_len - self.pred_len + 1
    
    
    def _build_valid_ids(self):
        """
        embed_path에 실제 존재하는 파일들 중,
        시계열 윈도우 시작 인덱스 범위(0..max_start-1)에 속하는 것만 수집
        stride를 고려하여 stride의 배수인 인덱스만 유효한 것으로 간주
        """
        if not os.path.isdir(self.embed_path):
            raise RuntimeError(f"Embedding path not found: {self.embed_path}")

        # 숫자 또는 제로패딩 숫자 파일 지원 (e.g., 7.h5, 0007.h5)
        paths = glob.glob(os.path.join(self.embed_path, "*.h5"))
        num_pat = re.compile(r"(\d+)\.h5$")

        # 파일 인덱스 수집
        file_indices = []
        for p in paths:
            m = num_pat.search(os.path.basename(p))
            if not m:
                continue
            file_idx = int(m.group(1))
            file_indices.append(file_idx)
        
        file_indices = sorted(set(file_indices))
        
        valid = []
        for file_idx in file_indices:  # [0, 1, 2]
            # 파일 0에서 가능한 s_begin: 0, 8, 16, 24 (stride=8이므로)
            # 하지만 실제로는 file_idx * per_file부터 시작
            # 파일 0: s_begin = 0, 8, 16, 24 (per_file=32 내에서 stride=8 간격)
            
            # 각 파일에서 stride 간격으로 s_begin 생성
            for i in range(0, self.samples_per_file, self.stride):  # [0, 8, 16, 24]
                s_begin = file_idx * self.samples_per_file + i
                if 0 <= s_begin < self.max_start:
                    valid.append(s_begin)

        valid = sorted(set(valid))
        if not valid:
            raise RuntimeError(f"No usable .h5 under {self.embed_path} (expected files covering range 0..{self.max_start-1} with stride={self.stride})")
        self.valid_ids = valid
        # 디버깅 도움:
        # print(f"[Embedding] usable files: {len(self.valid_ids)} / max_start={self.max_start}, last={self.valid_ids[-1]}")

    def __len__(self):
        # ✅ 실존하는 임베딩 파일에 맞춰 길이 정의
        return len(self.valid_ids)
    
    def __getitem__(self, index):
        # 1) 이 샘플의 시계열 구간 (x, y, x_mark, y_mark)
        s_begin = self.valid_ids[index]      # _build_valid_ids()에서 만든 시작 인덱스
        s_end   = s_begin + self.seq_len
        r_begin = s_end 
        r_end   = r_begin + self.pred_len

        seq_x      = self.data_x[s_begin:s_end]          # [L, N]
        seq_y      = self.data_y[r_begin:r_end]          # [pred_len, N]
        seq_x_mark = self.data_stamp[s_begin:s_end]      # [L, F]
        seq_y_mark = self.data_stamp[r_begin:r_end]      # [pred_len, F]

        # 2) 이 샘플에 해당하는 h5 파일 / 로컬 인덱스 계산
        #    ─ .h5 하나에 samples_per_file(=32)개 윈도우가 들어있다고 가정
        file_idx  = s_begin // self.samples_per_file
        local_idx = s_begin %  self.samples_per_file

        file_path = os.path.join(self.embed_path, f"{file_idx}.h5")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No embedding file found at {file_path}")
        try:
            hf = h5py.File(file_path, 'r')
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            raise FileNotFoundError(f"Error opening embedding file: {file_path}")

        # 3) last_token_embeddings: [B, d_llm, N] or [B, 1, d_llm, N] 등일 수 있음
        if "last_token_embeddings" in hf:
            dset = hf["last_token_embeddings"]
            last_np = dset[local_idx]        # 한 샘플만 읽기
            # shape 정리: [d_llm, N]로 맞춤
            if last_np.ndim == 3:            # [1, d_llm, N] 같은 경우
                last_np = last_np[0]
        elif "embeddings" in hf:
            # 예전 포맷: [B, T, d_llm, N]에서 마지막 토큰만 사용
            dset = hf["embeddings"]
            e_np = dset[local_idx]           # [T, d_llm, N]
            last_np = e_np[-1]               # [d_llm, N]
        else:
            # 안전장치: 없는 경우 0 텐서
            d_llm = getattr(self, "d_llm", 768)
            last_np = np.zeros((d_llm, self.num_nodes), dtype="float32")

        last_emb = torch.from_numpy(last_np).float()      # [d_llm, N]

        # 4) full prompt token embeddings: [T_i, d_llm, N] or [K, T_i, d_llm, N]
        if "prompt_token_embeddings" in hf:
            p_dset = hf["prompt_token_embeddings"]
            prompt_np = p_dset[local_idx]                 # 한 샘플
            # [K, T_i, d_llm, N]인 경우는 collate에서 K 평균 처리
            prompt_emb = torch.from_numpy(prompt_np).float()  # 그냥 그대로 넘김
        else:
            # 없으면 일단 last_emb를 1-step짜리로 사용
            prompt_emb = last_emb.unsqueeze(0)            # [1, d_llm, N]

        # 5) anchor_avg: [B, N, d_llm] 또는 [B, K, N, d_llm]
        if "anchor_avg" in hf:
            a_dset = hf["anchor_avg"]
            anchor_np = a_dset[local_idx]                 # 한 샘플
            anchor_avg = torch.from_numpy(anchor_np).float()
        else:
            # [N, d_llm] 기본 0값
            anchor_avg = torch.zeros(
                self.num_nodes, last_emb.shape[0], dtype=last_emb.dtype
            )  # [N, d_llm]

        # 6) num_token_idx: 이 샘플(local_idx)의 [N][L] 리스트
        if 'num_token_idx_json' in hf:
            try:
                raw_idx = json.loads(hf['num_token_idx_json'][()].decode())
                if isinstance(raw_idx, list):
                    selected_idx = raw_idx[local_idx:local_idx + 1]
                    if selected_idx:
                        json_all = selected_idx[0]  # [N][L]
                    else:
                        json_all = None
                else:
                    json_all = None
            except:
                json_all = None
        else:
            json_all = None
        
        if json_all is not None:
            # json_all: [B][N][L] 라고 가정
            per_sample = json_all  # [N][L]
            # L, N 길이 맞춰서 안전하게 잘라주기
            N = self.num_nodes
            L = self.seq_len
            
            num_token_idx = []
            for n in range(N):
                if n < len(per_sample):
                    row = per_sample[n]
                else:
                    row = []
                channel_list = []
                for t in range(L):
                    if t < len(row):
                        idx_list = row[t]
                        if not isinstance(idx_list, list):
                            idx_list = []
                    else:
                        idx_list = []
                    channel_list.append(idx_list)
                num_token_idx.append(channel_list)  # [L]
            # 최종: [N][L]
        else:
            # 기본값: [N][L], 모두 빈 리스트
            num_token_idx = [[[] for _ in range(self.seq_len)]
                            for _ in range(self.num_nodes)]

        hf.close()  # 파일 닫기

        # 7) 최종 반환: "샘플 1개"만 반환
        return (
            seq_x,          # [L, N]
            seq_y,          # [pred_len, N]
            seq_x_mark,     # [L, F]
            seq_y_mark,     # [pred_len, F]
            last_emb,       # [d_llm, N]
            prompt_emb,     # [T_i, d_llm, N] or [K, T_i, d_llm, N]
            anchor_avg,     # [N, d_llm] or [K, N, d_llm]
            num_token_idx   # [N][L] (list of list of list[int])
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
   
class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path="./dataset/", flag='train', size=None, 
                 features='M', data_path='ETTm1', model_name="gpt2",
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None, stride=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path

                # Append '.csv' if not present
        if not data_path.endswith('.csv'):
            data_path += '.csv'

        # Normalize root path
        root_path = os.path.abspath(root_path)

        # If data_path is absolute, use as-is. Else, join with root_path
        if os.path.isabs(data_path):
            self.data_path = data_path
        else:
            self.data_path = os.path.normpath(os.path.join(root_path, data_path))

        # Save file name without extension (for embed path)
        self.data_path_file = os.path.splitext(os.path.basename(self.data_path))[0]

        self.model_name = model_name
        self.stride = stride
        self.embed_path = f"./MY/Embeddings/{self.data_path_file}/{flag}/"

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        embeddings_stack = []
        # stride를 고려: 하나의 파일에 stride개만큼의 임베딩이 배치로 들어있음
        # 파일 인덱스 = s_begin // stride
        # 배치 내 인덱스 = s_begin % stride
        file_idx = s_begin // self.stride
        batch_idx = s_begin % self.stride
        file_path = os.path.join(self.embed_path, f"{file_idx}.h5")
        
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                data = hf['embeddings'][:]  # shape: [B, ...] 형태
                # 배치 차원에서 batch_idx 선택
                if data.ndim > 1:
                    data = data[batch_idx:batch_idx+1]  # 배치 차원 유지
                tensor = torch.from_numpy(data)
                embeddings_stack.append(tensor.squeeze(0))
        else:
            raise FileNotFoundError(f"No embedding file found at {file_path}")       
        
        embeddings = torch.stack(embeddings_stack, dim=-1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings
    
    def __len__(self):
        max_start = len(self.data_x) - self.seq_len - self.pred_len
        if max_start < 0:
            return 0
        return (max_start // self.stride) + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path="./dataset/", flag='train', size=None,
                 features='M', data_path='ECL',
                 target='OT', scale=True, timeenc=0, freq='h',
                 patch_len=16,percent=100,model_name="gpt2", stride=1, num_nodes=7, batch_size=32):
        # size [seq_len, label_len, pred_len]
        # info
        self.percent = percent
        self.patch_len = patch_len
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.num_nodes = num_nodes
        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = batch_size  # 배치크기 저장
        self.samples_per_file = 32  # 각 파일에 들어있는 샘플 수

                # Append '.csv' if not present
        if not data_path.endswith('.csv'):
            data_path += '.csv'

        # Normalize root path
        root_path = os.path.abspath(root_path)

        # If data_path is absolute, use as-is. Else, join with root_path
        if os.path.isabs(data_path):
            self.data_path = data_path
        else:
            self.data_path = os.path.normpath(os.path.join(root_path, data_path))

        # Save file name without extension (for embed path)
        self.data_path_file = os.path.splitext(os.path.basename(self.data_path))[0]

        self.model_name = model_name
        self.stride = stride
        self.embed_path = f"./MY/Embeddings/{self.data_path_file}/{flag}/"

        self.__read_data__()
        
        # 임베딩 경로는 절대경로로 고정(상대경로 혼동 방지)
        self.embed_path = os.path.abspath(f"./MY/Embeddings/{self.data_path_file}/{flag}/")
        
        # ✅ 여기서 유효 인덱스 수집
        self._build_valid_ids()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        # 👇 이 길이를 넘어서는 시작 인덱스는 시계열 슬라이딩에서 불가
        self.max_start = len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def _build_valid_ids(self):
        """
        embed_path에 실제 존재하는 파일들 중,
        시계열 윈도우 시작 인덱스 범위(0..max_start-1)에 속하는 것만 수집
        stride를 고려하여 stride의 배수인 인덱스만 유효한 것으로 간주
        """
        if not os.path.isdir(self.embed_path):
            raise RuntimeError(f"Embedding path not found: {self.embed_path}")

        # 숫자 또는 제로패딩 숫자 파일 지원 (e.g., 7.h5, 0007.h5)
        paths = glob.glob(os.path.join(self.embed_path, "*.h5"))
        num_pat = re.compile(r"(\d+)\.h5$")

        # 파일 인덱스 수집
        file_indices = []
        for p in paths:
            m = num_pat.search(os.path.basename(p))
            if not m:
                continue
            file_idx = int(m.group(1))
            file_indices.append(file_idx)
        
        file_indices = sorted(set(file_indices))
        
        valid = []
        for file_idx in file_indices:  # [0, 1, 2]
            # 파일 0에서 가능한 s_begin: 0, 8, 16, 24 (stride=8이므로)
            # 하지만 실제로는 file_idx * per_file부터 시작
            # 파일 0: s_begin = 0, 8, 16, 24 (per_file=32 내에서 stride=8 간격)
            
            # 각 파일에서 stride 간격으로 s_begin 생성
            for i in range(0, self.samples_per_file, self.stride):  # [0, 8, 16, 24]
                s_begin = file_idx * self.samples_per_file + i
                if 0 <= s_begin < self.max_start:
                    valid.append(s_begin)

        valid = sorted(set(valid))
        if not valid:
            raise RuntimeError(f"No usable .h5 under {self.embed_path} (expected files covering range 0..{self.max_start-1} with stride={self.stride})")
        self.valid_ids = valid
        # 디버깅 도움:
        # print(f"[Embedding] usable files: {len(self.valid_ids)} / max_start={self.max_start}, last={self.valid_ids[-1]}")

    def __len__(self):
        # ✅ 실존하는 임베딩 파일에 맞춰 길이 정의
        return len(self.valid_ids)

    def __getitem__(self, index):
        # 1) 이 샘플의 시계열 구간 (x, y, x_mark, y_mark)
        s_begin = self.valid_ids[index]      # _build_valid_ids()에서 만든 시작 인덱스
        s_end   = s_begin + self.seq_len
        r_begin = s_end
        r_end   = r_begin + self.pred_len

        seq_x      = self.data_x[s_begin:s_end]          # [L, N]
        seq_y      = self.data_y[r_begin:r_end]          # [pred_len, N]
        seq_x_mark = self.data_stamp[s_begin:s_end]      # [L, F]
        seq_y_mark = self.data_stamp[r_begin:r_end]      # [pred_len, F]

        # 2) 이 샘플에 해당하는 h5 파일 / 로컬 인덱스 계산
        #    ─ .h5 하나에 samples_per_file(=32)개 윈도우가 들어있다고 가정
        file_idx  = s_begin // self.samples_per_file
        local_idx = s_begin %  self.samples_per_file

        file_path = os.path.join(self.embed_path, f"{file_idx}.h5")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No embedding file found at {file_path}")
        try:
            hf = h5py.File(file_path, 'r')
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            raise FileNotFoundError(f"Error opening embedding file: {file_path}")

        # 3) last_token_embeddings: [B, d_llm, N] or [B, 1, d_llm, N] 등일 수 있음
        if "last_token_embeddings" in hf:
            dset = hf["last_token_embeddings"]
            last_np = dset[local_idx]        # 한 샘플만 읽기
            # shape 정리: [d_llm, N]로 맞춤
            if last_np.ndim == 3:            # [1, d_llm, N] 같은 경우
                last_np = last_np[0]
        elif "embeddings" in hf:
            # 예전 포맷: [B, T, d_llm, N]에서 마지막 토큰만 사용
            dset = hf["embeddings"]
            e_np = dset[local_idx]           # [T, d_llm, N]
            last_np = e_np[-1]               # [d_llm, N]
        else:
            # 안전장치: 없는 경우 0 텐서
            d_llm = getattr(self, "d_llm", 768)
            last_np = np.zeros((d_llm, self.num_nodes), dtype="float32")

        last_emb = torch.from_numpy(last_np).float()      # [d_llm, N]

        # 4) full prompt token embeddings: [T_i, d_llm, N] or [K, T_i, d_llm, N]
        if "prompt_token_embeddings" in hf:
            p_dset = hf["prompt_token_embeddings"]
            prompt_np = p_dset[local_idx]                 # 한 샘플
            # [K, T_i, d_llm, N]인 경우는 collate에서 K 평균 처리
            prompt_emb = torch.from_numpy(prompt_np).float()  # 그냥 그대로 넘김
        else:
            # 없으면 일단 last_emb를 1-step짜리로 사용
            prompt_emb = last_emb.unsqueeze(0)            # [1, d_llm, N]

        # 5) anchor_avg: [B, N, d_llm] 또는 [B, K, N, d_llm]
        if "anchor_avg" in hf:
            a_dset = hf["anchor_avg"]
            anchor_np = a_dset[local_idx]                 # 한 샘플
            anchor_avg = torch.from_numpy(anchor_np).float()
        else:
            # [N, d_llm] 기본 0값
            anchor_avg = torch.zeros(
                self.num_nodes, last_emb.shape[0], dtype=last_emb.dtype
            )  # [N, d_llm]

        # 6) num_token_idx: 이 샘플(local_idx)의 [N][L] 리스트
        if 'num_token_idx_json' in hf:
            try:
                raw_idx = json.loads(hf['num_token_idx_json'][()].decode())
                if isinstance(raw_idx, list):
                    selected_idx = raw_idx[local_idx:local_idx + 1]
                    if selected_idx:
                        json_all = selected_idx[0]  # [N][L]
                    else:
                        json_all = None
                else:
                    json_all = None
            except:
                json_all = None
        else:
            json_all = None
        
        if json_all is not None:
            # json_all: [B][N][L] 라고 가정
            per_sample = json_all  # [N][L]
            # L, N 길이 맞춰서 안전하게 잘라주기
            N = self.num_nodes
            L = self.seq_len

            num_token_idx = []
            for n in range(N):
                if n < len(per_sample):
                    row = per_sample[n]
                else:
                    row = []
                channel_list = []
                for t in range(L):
                    if t < len(row):
                        idx_list = row[t]
                        if not isinstance(idx_list, list):
                            idx_list = []
                    else:
                        idx_list = []
                    channel_list.append(idx_list)
                num_token_idx.append(channel_list)  # [L]
            # 최종: [N][L]
        else:
            # 기본값: [N][L], 모두 빈 리스트
            num_token_idx = [[[] for _ in range(self.seq_len)]
                            for _ in range(self.num_nodes)]

        hf.close()  # 파일 닫기

        # 7) 최종 반환: "샘플 1개"만 반환
        return (
            seq_x,          # [L, N]
            seq_y,          # [pred_len, N]
            seq_x_mark,     # [L, F]
            seq_y_mark,     # [pred_len, F]
            last_emb,       # [d_llm, N]
            prompt_emb,     # [T_i, d_llm, N] or [K, T_i, d_llm, N]
            anchor_avg,     # [N, d_llm] or [K, N, d_llm]
            num_token_idx   # [N][L] (list of list of list[int])
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)