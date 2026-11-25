import numpy as np
import pickle
from pathlib import Path
from src.data import build_vocab, decode, encode
from src.model import GPT, GPTConfig
from src.sample import sample

ROOT = Path(__file__).resolve().parent
# 读取我们刚才训练好的权重
ckpt_path = ROOT / "checkpoints" / "final.pkl"
with ckpt_path.open("rb") as f:
    state = pickle.load(f)

model_config = state["config"]
model = GPT(model_config, rng=np.random.default_rng(42))
# 加载参数
for p, loaded_p in zip(model.params_and_grads(), state["params"]):
    p[0][:] = loaded_p

# 直接读取训练时保存的词表，不再重新构建
vocab_path = ROOT / "checkpoints" / "vocab.pkl"
with vocab_path.open("rb") as f:
    vocab_data = pickle.load(f)
stoi = vocab_data["stoi"]
itos = vocab_data["itos"]


print("✅ 化学小模型加载完成！")
print("👉 提问模板示例：问：燃烧的三个条件是什么？答：")
print("👉 输入 quit 退出\n")

while True:
    prompt = input("请输入：")
    if prompt.strip() == "quit":
        break
    prompt_ids = encode(prompt, stoi)
    out_ids = sample(
        model, prompt_ids,
        max_new_tokens=80,
        temperature=0.1,
        top_k=40,
        rng=np.random.default_rng()
    )
    output = decode(out_ids, itos)
    print("输出：", output, "\n")
