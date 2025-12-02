from transformers import AutoTokenizer
from word2number import w2n

def get_default_tokenizer(model_name="gpt2"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def find_numeric_phrases(prompt):
    """
    프롬프트 문자열에서 숫자 단어 조합을 찾아 (값, (단어 시작, 끝)) 튜플 리스트로 반환
    """
    words = prompt.lower().replace('-', ' ').split()
    phrases = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + 4, len(words)) + 1):  # 최대 3단어 숫자 표현
            phrase = " ".join(words[i:j])
            try:
                val = w2n.word_to_num(phrase)
                phrases.append((val, (i, j - 1)))
            except:
                continue
    return phrases

def normalize_tokens(tokens):
    """ subword prefix 제거하고 정규화된 토큰 문자열 리스트 반환 """
    return [t.lstrip("Ġ") for t in tokens]

def find_subsequence_indices(token_norms, phrase_words):
    """
    token_norms: normalized token list (e.g., ['seven','point','one','four',...])
    phrase_words: ['seven','point','one','four','three','four']
    서브시퀀스 일치하는 구간의 인덱스 반환. 없다면 빈 리스트.
    """
    L = len(token_norms)
    M = len(phrase_words)
    for start in range(L - M + 1):
        if token_norms[start : start + M] == phrase_words:
            return list(range(start, start + M))
    return []

def flexible_subsequence_match(token_norms, phrase_words, skip_tokens={'-'}):
    """
    phrase_words를 token_norms 안에서 skip_tokens를 무시하면서 순차적으로 매칭.
    성공하면 인덱스 리스트 반환.
    """
    L = len(token_norms)
    M = len(phrase_words)
    for start in range(L):
        ti = start
        pi = 0
        matched = []
        while ti < L and pi < M:
            tok = token_norms[ti]
            if tok in skip_tokens:
                ti += 1
                continue
            if tok == phrase_words[pi]:
                matched.append(ti)
                ti += 1
                pi += 1
            else:
                break
        if pi == M:
            return matched
    return []


def partial_fallback(token_norms, phrase_words, min_len=2, skip_tokens={'-'}):
    """
    전체 phrase가 안 걸리면 앞/뒤 일부 서브셋도 시도.
    """
    M = len(phrase_words)
    for l in range(M, min_len - 1, -1):
        # 앞쪽 l개
        sub_front = phrase_words[:l]
        m = flexible_subsequence_match(token_norms, sub_front, skip_tokens=skip_tokens)
        if m:
            return m
        # 뒤쪽 l개
        sub_back = phrase_words[-l:]
        m = flexible_subsequence_match(token_norms, sub_back, skip_tokens=skip_tokens)
        if m:
            return m
    return []


def get_prompt_indices(prompt, tokenizer, target_value, phrase_text=None, expand_window=1):
    """
    숫자값 target_value에 대응하는 token 인덱스 리스트 반환.
    phrase_text를 주면 그 표현을 기반으로 subsequence 매칭 우선 시도.
    """
    tokenized = tokenizer(
        prompt,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    if "offset_mapping" not in tokenized:
        return []

    offsets = tokenized["offset_mapping"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
    normed = normalize_tokens(tokens)
    words = prompt.split()
    phrases = find_numeric_phrases(prompt)

    # 0. phrase_text 기반 flexible subsequence match
    if phrase_text:
        phrase_words = phrase_text.lower().split()
        matched = flexible_subsequence_match(normed, phrase_words, skip_tokens={'-'})
        if matched:
            return matched
        # partial fallback
        matched = partial_fallback(normed, phrase_words, skip_tokens={'-'})
        if matched:
            return matched

    # 1. word-span 기반 strict/overlap match (기존)
    for val, (word_start, word_end) in phrases:
        if val == target_value:
            char_start = len(" ".join(words[:word_start])) + (1 if word_start > 0 else 0)
            char_end = len(" ".join(words[: word_end + 1]))
            strict = [
                i for i, (s, e) in enumerate(offsets) if s >= char_start and e <= char_end
            ]
            if strict:
                expanded = set()
                for idx in strict:
                    for delta in range(-expand_window, expand_window + 1):
                        if 0 <= idx + delta < len(tokens):
                            expanded.add(idx + delta)
                return sorted(expanded)
            loose = [
                i
                for i, (s, e) in enumerate(offsets)
                if not (e < char_start or s > char_end)
            ]
            if loose:
                return sorted(set(loose))

    # 2. fuzzy fallback: word-level phrase in concatenated normalized tokens
    for val, (word_start, word_end) in phrases:
        if val == target_value:
            phrase = "".join(prompt.split()[word_start:word_end + 1]).replace(" ", "").lower()
            concat = "".join(normed).lower()
            if phrase and phrase in concat:
                return list(range(len(tokens)))

    return []  # ultimate fallback caller에서 처리

def debug_get_prompt_indices(prompt, tokenizer, target_value):
    """
    디버깅용: 내부 상태 찍어주는 함수
    """
    print("=== DEBUG get_prompt_indices ===")
    print("Prompt:", prompt)
    tokenized = tokenizer(
        prompt,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
    normed = normalize_tokens(tokens)
    offsets = tokenized["offset_mapping"][0].tolist()
    print("Tokens:", tokens)
    print("Normalized:", normed)
    print("Offsets:", offsets)
    phrases = find_numeric_phrases(prompt)
    print("Numeric phrases:", phrases)
    for val, (ws, we) in phrases:
        if val == target_value:
            print(f"Target numeric phrase: {' '.join(prompt.split()[ws:we+1])}")
            # strict span
            words = prompt.split()
            char_start = len(" ".join(words[:ws])) + (1 if ws > 0 else 0)
            char_end = len(" ".join(words[: we + 1]))
            strict = [i for i, (s, e) in enumerate(offsets) if s >= char_start and e <= char_end]
            overlap = [i for i, (s, e) in enumerate(offsets) if not (e < char_start or s > char_end)]
            print("Strict indices:", strict)
            print("Overlap indices:", overlap)
            subseq = find_subsequence_indices(normed, prompt.split()[ws:we+1])
            print("Subsequence attempt (word-level):", subseq)
    print("=== end debug ===")