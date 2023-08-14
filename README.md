# japanese-large-lm-instruction-sft

This repository provides a 1.7B and 3.6B parameters Japanese language model, fine-tuned and trained by [LINE Corporation](https://linecorp.com/ja/).

- https://huggingface.co/line-corporation/japanese-large-lm-3.6b-sft
- https://huggingface.co/line-corporation/japanese-large-lm-1.7b-sft

## For Japanese

詳細な説明や実験に関しては「[Instruction Tuningにより対話性能を向上させた3.6B日本語言語モデルを公開します](https://engineering.linecorp.com/ja/blog/3.6b-japanese-language-model-with-improved-dialog-performance-by-instruction-tuning)」をご覧ください。

## How to use

```python
import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-3.6b-sft", use_fast=False)
model = AutoModel.from_pretrained("line-corporation/japanese-large-lm-3.6b-sft")
 
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
 
input_text = """四国の県名を全て列挙してください。"""
text = generator(
    f"ユーザー: {input_text}\nシステム: ",
    max_length = 256,
    do_sample = True,
    temperature = 0.7,
    top_p = 0.9,
    top_k = 0,
    repetition_penalty = 1.1,
    num_beams = 1,
    pad_token_id = tokenizer.pad_token_id,
    num_return_sequences = 1,
)

print(text)
# # [{'generated_text': 'ユーザー: 四国の県名を全て列挙してください。\nシステム:  高知県、徳島県、香川県、愛媛県'}]
```

## Tokenization

We use a sentencepiece tokenizer with a unigram language model and byte-fallback.
We **do not** apply pre-tokenization with Japanese tokenizer.
Thus, a user may directly feed raw sentences into the tokenizer.


## License

[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
