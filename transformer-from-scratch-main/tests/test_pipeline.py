"""Unit tests covering layers, attention, layer norm, and end-to-end model.

Run: python -m unittest discover tests
"""
from __future__ import annotations

import unittest

import numpy as np

from src.attention import MultiHeadSelfAttention
from src.block import TransformerBlock
from src.data import build_vocab, decode, encode
from src.embedding import TokenEmbedding, sinusoidal_position_encoding
from src.gradcheck import numerical_gradient, relative_error, upcast_module
from src.layer_norm import LayerNorm
from src.layers import Linear, ReLU
from src.losses import cross_entropy_loss, log_softmax, softmax
from src.model import GPT, GPTConfig, parameter_count


class TestTokenizer(unittest.TestCase):
    def test_round_trip(self):
        text = "Hello, world!\nGoodbye."
        stoi, itos = build_vocab(text)
        ids = encode(text, stoi)
        self.assertEqual(decode(ids, itos), text)

    def test_vocab_is_sorted_unique(self):
        stoi, itos = build_vocab("zaidan")
        self.assertEqual(itos, sorted(set("zaidan")))


class TestEmbedding(unittest.TestCase):
    def test_token_embedding_shape(self):
        rng = np.random.default_rng(0)
        emb = TokenEmbedding(vocab_size=10, d_model=4, rng=rng)
        idx = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        y = emb.forward(idx)
        self.assertEqual(y.shape, (2, 3, 4))

    def test_position_encoding_bounded(self):
        pe = sinusoidal_position_encoding(max_T=128, d_model=64)
        self.assertTrue(np.all(np.abs(pe) <= 1.0 + 1e-6))

    def test_position_encoding_shape(self):
        pe = sinusoidal_position_encoding(max_T=64, d_model=32)
        self.assertEqual(pe.shape, (64, 32))


class TestLossFunctions(unittest.TestCase):
    def test_log_softmax_overflow_safety(self):
        z = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        out = log_softmax(z)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_softmax_normalised(self):
        z = np.random.default_rng(0).standard_normal((4, 8))
        s = softmax(z)
        np.testing.assert_allclose(s.sum(axis=-1), np.ones(4), atol=1e-6)

    def test_cross_entropy_grad_rows_sum_to_zero(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((3, 5)).astype(np.float32)
        y = rng.integers(0, 5, size=3)
        _, grad = cross_entropy_loss(logits, y)
        np.testing.assert_allclose(grad.sum(axis=-1), 0.0, atol=1e-6)


class TestAttention(unittest.TestCase):
    def test_attention_weights_normalised(self):
        rng = np.random.default_rng(0)
        attn = MultiHeadSelfAttention(d_model=16, n_heads=2, rng=rng)
        x = rng.standard_normal((2, 4, 16)).astype(np.float32)
        attn.forward(x)
        A = attn._cache["A"]
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, atol=1e-5)

    def test_attention_causal(self):
        rng = np.random.default_rng(0)
        attn = MultiHeadSelfAttention(d_model=16, n_heads=2, rng=rng)
        x = rng.standard_normal((2, 4, 16)).astype(np.float32)
        attn.forward(x)
        A = attn._cache["A"]
        upper_tri = np.triu(np.ones((4, 4), dtype=bool), k=1)
        self.assertEqual((A * upper_tri).sum(), 0.0)

    def test_attention_gradient_check(self):
        rng = np.random.default_rng(0)
        attn = MultiHeadSelfAttention(d_model=16, n_heads=2, rng=rng)
        upcast_module(attn)
        x = rng.standard_normal((2, 4, 16))
        grad_out = rng.standard_normal((2, 4, 16))

        attn.forward(x)
        dx = attn.backward(grad_out)

        def loss_fn():
            return float((attn.forward(x) * grad_out).sum())

        for param, analytic in [
            (attn.W_qkv, attn.dW_qkv),
            (attn.W_o, attn.dW_o),
        ]:
            numerical = numerical_gradient(loss_fn, param, eps=1e-6)
            self.assertLess(relative_error(analytic, numerical), 1e-5)


class TestLayerNorm(unittest.TestCase):
    def test_layernorm_normalises(self):
        ln = LayerNorm(d_model=8)
        x = np.random.default_rng(0).standard_normal((2, 3, 8)).astype(np.float32) * 5
        y = ln.forward(x)
        # With default gamma=1, beta=0, output should have ~zero mean and unit variance per row.
        self.assertLess(float(np.abs(y.mean(-1)).max()), 1e-5)
        self.assertLess(float(np.abs(y.std(-1) - 1).max()), 1e-3)

    def test_layernorm_gradient_check(self):
        rng = np.random.default_rng(0)
        ln = LayerNorm(d_model=8)
        ln.gamma = (rng.standard_normal(8) * 0.5 + 1).astype(np.float32)
        upcast_module(ln)
        x = rng.standard_normal((2, 3, 8))
        grad_out = rng.standard_normal((2, 3, 8))

        ln.forward(x)
        ln.backward(grad_out)

        def loss_fn():
            return float((ln.forward(x) * grad_out).sum())

        for param, analytic in [(ln.gamma, ln.dgamma), (ln.beta, ln.dbeta)]:
            numerical = numerical_gradient(loss_fn, param, eps=1e-6)
            self.assertLess(relative_error(analytic, numerical), 1e-5)


class TestTransformerBlock(unittest.TestCase):
    def test_block_forward_shape(self):
        rng = np.random.default_rng(0)
        block = TransformerBlock(d_model=16, n_heads=2, rng=rng)
        x = rng.standard_normal((2, 4, 16)).astype(np.float32)
        out = block.forward(x)
        self.assertEqual(out.shape, x.shape)


class TestGPT(unittest.TestCase):
    def test_model_forward_shape(self):
        rng = np.random.default_rng(0)
        config = GPTConfig(vocab_size=20, d_model=16, n_heads=2, n_layers=2, block_size=8)
        model = GPT(config, rng=rng)
        idx = rng.integers(0, 20, size=(3, 5)).astype(np.int64)
        logits = model.forward(idx)
        self.assertEqual(logits.shape, (3, 5, 20))

    def test_model_backward_runs(self):
        rng = np.random.default_rng(0)
        config = GPTConfig(vocab_size=20, d_model=16, n_heads=2, n_layers=2, block_size=8)
        model = GPT(config, rng=rng)
        idx = rng.integers(0, 20, size=(3, 5)).astype(np.int64)
        logits = model.forward(idx)
        grad = rng.standard_normal(logits.shape).astype(np.float32)
        model.backward(grad)
        # All parameters should have gradients populated
        for param, grad_p in model.params_and_grads():
            self.assertIsNotNone(grad_p, f"missing grad for param shape {param.shape}")
            self.assertEqual(grad_p.shape, param.shape)

    def test_parameter_count_matches_sum(self):
        config = GPTConfig(vocab_size=20, d_model=16, n_heads=2, n_layers=2, block_size=8)
        model = GPT(config, rng=np.random.default_rng(0))
        explicit_sum = sum(p.size for p, _ in model.params_and_grads())
        self.assertEqual(parameter_count(model), explicit_sum)


if __name__ == "__main__":
    unittest.main()
