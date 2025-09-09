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
                 model_name="gpt2"):
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
        """
        if not os.path.isdir(self.embed_path):
            raise RuntimeError(f"Embedding path not found: {self.embed_path}")

        # 숫자 또는 제로패딩 숫자 파일 지원 (e.g., 7.h5, 0007.h5)
        paths = glob.glob(os.path.join(self.embed_path, "*.h5"))
        num_pat = re.compile(r"(\d+)\.h5$")

        valid = []
        for p in paths:
            m = num_pat.search(os.path.basename(p))
            if not m:
                continue
            i = int(m.group(1))
            if 0 <= i < self.max_start:
                valid.append(i)

        valid = sorted(set(valid))
        if not valid:
            raise RuntimeError(f"No usable .h5 under {self.embed_path} (expected 0..{self.max_start-1})")
        self.valid_ids = valid
        # 디버깅 도움:
        # print(f"[Embedding] usable files: {len(self.valid_ids)} / max_start={self.max_start}, last={self.valid_ids[-1]}")

    def __len__(self):
        # ✅ 실존하는 임베딩 파일에 맞춰 길이 정의
        return len(self.valid_ids)
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]            # [L, N]
        seq_y = self.data_y[r_begin:r_end]            # [pred_len, N]
        seq_x_mark = self.data_stamp[s_begin:s_end]   # [L, F]
        seq_y_mark = self.data_stamp[r_begin:r_end]   # [pred_len, F]
        
        ## last-token + full prompt-token 임베딩 함께 읽기
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        
        if os.path.exists(file_path):
            try:
                with h5py.File(file_path, 'r') as hf:
                    # 마지막 토큰 임베딩
                    if 'last_token_embeddings' in hf:
                        last_np = hf['last_token_embeddings'][:]        # shape (1, d_model, N)
                    # 프롬프트 내 모든 토큰 임베딩
                    full_np = hf['prompt_token_embeddings'][:]      # shape (1, T_max, d_model, N)
                    prompt_np = hf['prompt_token_embeddings'][:]   # [B, T_max, d_llm, N] (선택)
                    anchor_np = hf['anchor_avg'][:]                # [B, N, d_llm]  (B=1)
                    
                    # (옵션) 메타도 필요하면 읽기
                    raw_idx = json.loads(hf['num_token_idx_json'][()].decode())
                    # num_vals= json.loads(hf['num_values_json'][()].decode())
                
                last_emb   = torch.from_numpy(last_np).float().squeeze(0)     # [d_llm, N]
                prompt_emb = torch.from_numpy(prompt_np).float().squeeze(0)   # [T_max, d_llm, N]
                anchor_avg = torch.from_numpy(anchor_np).float().squeeze(0)   # [N, d_llm]
                
                T_max = prompt_emb.shape[0]
                N     = prompt_emb.shape[-1]
                L     = seq_x.shape[0]  # 시점 길이

                # ---- num_token_idx를 [L][N]으로 전치 ----
                # 저장 당시 raw_idx = [B][N][L], 여기서는 B=1 가정 → raw_idx[0] = [N][L]
                if not isinstance(raw_idx, list) or len(raw_idx) == 0:
                    # 안전장치: 비었으면 빈 리스트 리턴
                    num_token_idx = [[[] for _ in range(N)] for _ in range(L)]
                else:
                    per_b = raw_idx[0]  # [N][L]
                    # 전치: [N][L] -> [L][N]
                    # 각 항목은 "그 숫자를 구성하는 토큰 인덱스 리스트"
                    # 또한 T_max 범위를 벗어나는 인덱스는 제거
                    num_token_idx = []
                    for t in range(L):
                        row_t = []
                        for n in range(N):
                            # per_b[n][t] 가 리스트(= 그 시점의 숫자 토큰 인덱스들)여야 함
                            idx_list = per_b[n][t] if (n < len(per_b) and t < len(per_b[n])) else []
                            if not isinstance(idx_list, list):
                                idx_list = []
                            # 클리핑: 0 <= idx < T_max
                            idx_list = [int(ix) for ix in idx_list if isinstance(ix, (int, float)) and 0 <= int(ix) < T_max]
                            row_t.append(idx_list)
                        num_token_idx.append(row_t)  # [N] (각 채널의 인덱스 리스트)
                    # 이제 num_token_idx는 [L][N] 구조   
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                # 기본값으로 빈 텐서 반환
                last_emb = torch.zeros(self.d_llm, self.num_nodes)
                prompt_emb = torch.zeros(1, self.d_llm, self.num_nodes)
                anchor_avg = torch.zeros(self.num_nodes, self.d_llm)
                num_token_idx = [[[] for _ in range(self.num_nodes)] for _ in range(self.seq_len)]
        else:
            raise FileNotFoundError(f"No embedding file found at {file_path}")       
        
        # 모델에서 어떤 것을 쓸지에 따라 반환
        # 여기서는 last_emb + prompt_emb + anchor_avg 모두 리턴
        # 추가:  num_token_idx  (shape: [L][N] = list of list of token_id list)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, last_emb, prompt_emb, anchor_avg, num_token_idx

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
   
class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path="./dataset/", flag='train', size=None, 
                 features='M', data_path='ETTm1', model_name="gpt2",
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
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
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        embeddings_stack = []
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                data = hf['embeddings'][:]
                tensor = torch.from_numpy(data)
                embeddings_stack.append(tensor.squeeze(0))
        else:
            raise FileNotFoundError(f"No embedding file found at {file_path}")       
        
        embeddings = torch.stack(embeddings_stack, dim=-1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path="./dataset/", flag='train', size=None,
                 features='M', data_path='ECL',
                 target='OT', scale=True, timeenc=0, freq='h',
                 patch_len=16,percent=100,model_name="gpt2"):
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
        self.embed_path = f"./MY/Embeddings/{self.data_path_file}/{flag}/"

        self.__read_data__()

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

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # auto_y = self.data_x[s_begin+self.patch_len:s_end+self.patch_len]

        embeddings_stack = []
        # for n in range(self.num_nodes):
        # print("embed_path in __getitem__:", self.embed_path)
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                data = hf['embeddings'][:]
                tensor = torch.from_numpy(data)
                embeddings_stack.append(tensor.squeeze(0))
        else:
            raise FileNotFoundError(f"No embedding file found at {file_path}")
                
        embeddings = torch.stack(embeddings_stack, dim=-1)
        # print("Shape of embeddings: " ,embeddings.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)