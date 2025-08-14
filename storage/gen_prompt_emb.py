import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Model
from num2words import num2words
from utils.prompt_parser import get_prompt_indices, find_numeric_phrases, debug_get_prompt_indices

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'ETTm1',
        model_name = "gpt2",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train'
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1
        
        # fast tokenizer & model (so offset_mapping works)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ------ position embedding extension -----
        new_max_pos = 2048
        self.model.config.n_positions = new_max_pos
        self.model.config.max_position_embeddings = new_max_pos
        self.model.config.ctx_len = 2048 
        old_wpe = self.model.wpe.weight.data  # [original_max, d_model]
        old_max_pos, d_model = old_wpe.shape

        new_wpe = nn.Embedding(new_max_pos, d_model).to(self.device)
        new_wpe.weight.data[:old_max_pos] = old_wpe
        new_wpe.weight.data[old_max_pos:] = old_wpe[-1].unsqueeze(0).repeat(new_max_pos - old_max_pos, 1)
        self.model.wpe = new_wpe
        
    def float_to_words(self, x: float) -> str:
        s = str(x)
        if '.' not in s:
            try:
                return num2words(int(x)).replace('-', ' ')
            except:
                return str(x)
        integer_part, decimal_part = s.split('.')
        parts = []
        try:
            parts.append(num2words(int(integer_part)).replace('-', ' '))
        except:
            parts.append(str(integer_part))
        parts.append('point')
        for d in decimal_part:
            try:
                parts.append(num2words(int(d)).replace('-', ' '))
            except:
                parts.append(d)
        return ' '.join(parts)
    
    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        # 원래 값들 가져오고 소수점 둘째 자리까지 반올림
        raw_values = in_data[i, :, j].flatten().tolist()
        values = [round(float(v), 2) for v in raw_values] 
        text_values = [self.float_to_words(v) for v in values]
        values_str = ", ".join(text_values)

        # trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        # trend_int = int(trends.item())
        trends = round(float(values[-1] - values[0]), 2)
        trends_str = self.float_to_words(trends)

        # Date formatting
        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)

        # Tokenize with truncation to avoid too-long sequences
        tokenized = self.tokenizer(
            in_prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        model_input = {k: v.to(self.device) for k, v in tokenized.items() if k != "offset_mapping"}

        # Collect value-index pairs using enhanced matching
        value_index_pairs = []
        for v in values:
            phrase_text = self.float_to_words(v)
            indices = get_prompt_indices(in_prompt, self.tokenizer, float(v), phrase_text=phrase_text)
            if not indices:
                # 디버그 필요할 때 주석 해제
                # debug_get_prompt_indices(in_prompt, self.tokenizer, float(v))
                pass
            if len(indices) > 0:
                value_index_pairs.append((float(v), indices))

        if not value_index_pairs:
            # fallback: 마지막 값과 마지막 토큰
            value_index_pairs.append((float(values[-1]), [-1]))
            
        # print(value_index_pairs)

        return model_input, value_index_pairs  # list of (value, [token positions])

    def forward(self, tokenized_prompt_dict):
        # tokenized_prompt_dict: output from tokenizer (input_ids, attention_mask, etc.)
        with torch.no_grad():
            outputs = self.model(**tokenized_prompt_dict)
            hidden = outputs.last_hidden_state  # [B, L, d_model]
        return hidden  # caller will select indices externally

    def generate_embeddings(self, in_data, in_data_mark):
            input_templates = {
                'FRED': "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
                'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
                'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends"
            }

            input_template = input_templates.get(self.data_path, input_templates['ETTh1'])
            
            tokenized_prompts = []
            max_token_count = 0
            
            B = len(in_data)
            N = in_data.shape[2]

            # 채널별 정렬 메타(숫자값, 토큰인덱스) 초기화: 배치마다 N개 채널
            # align_meta[b][n] = [(v_0, [idx...]), ..., (v_{L-1}, [idx...])]
            align_meta = [[None for _ in range(N)] for _ in range(B)]

            # Collect per-window prompt inputs
            for i in range(B):
                for j in range(N):
                    model_input, value_index_pairs = self._prepare_prompt(input_template, in_data, in_data_mark, i, j)
                    # we only need the tokenized prompt dict for embedding matrix building here
                    # ignore value_index_pairs for compatibility with previous return
                    input_ids = model_input.get("input_ids")
                    if input_ids is None:
                        continue
                    seq_len = input_ids.shape[1]
                    max_token_count = max(max_token_count, seq_len)
                    tokenized_prompts.append((i, model_input, j, seq_len))
                    # 채널 j의 시간순 (값, 토큰 idx 리스트) 저장
                    align_meta[i][j] = value_index_pairs

            in_prompt_emb = torch.zeros((B, max_token_count, self.d_model, N), 
                                        dtype=torch.float32, device=self.device
            )

            for i, tokenized_prompt, j, seq_len in tokenized_prompts:
                # (B=1, T_ij, d_model)
                prompt_embeddings = self.forward(tokenized_prompt)     # [1, T_ij, d_model]
                T_ij = prompt_embeddings.shape[1]
                # 길이 맞추기(padding)
                if T_ij < max_token_count:
                    last = prompt_embeddings[:, -1:, :]              # [1,1,d_model]
                    pad = last.repeat(1, max_token_count - T_ij, 1)
                    prompt_embeddings = torch.cat([prompt_embeddings, pad], dim=1)
                # [1, max_token_count, d_model] → 버퍼에 저장
                in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings

            # 5) 마지막 토큰 임베딩 (종전 반환값)
            last_token_emb = in_prompt_emb[:, max_token_count-1, :, :]   # [B, d_model, N]

            # 이제 full prompt-embedding matrix 도 함께 반환
            # in_prompt_emb: [B, max_token_count, d_model, N]
            # (변경점) align_meta도 같이 반환
            return last_token_emb, in_prompt_emb, align_meta