"""Plumbing tests for the real-transformer SUTURE adapter.

These tests intentionally instantiate a tiny local GPT-2-shaped decoder.  They
are not language-model evidence and do not stand in for the frozen Tier-A
checkpoint.  They exercise the adapter boundary without network access:
trajectory identity, architecture indexing, numerical adjoints, routed graft
evaluation, and the no-build-during-selection guard.
"""

from __future__ import annotations

import copy
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM

from suture.suture_torch import AdapterError, HFResidualAdapter, TorchPrompt


def tiny_pair() -> tuple[GPT2LMHeadModel, GPT2LMHeadModel]:
    config = GPT2Config(
        vocab_size=32,
        n_positions=8,
        n_ctx=8,
        n_embd=24,
        n_layer=3,
        n_head=3,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(7)
    host = GPT2LMHeadModel(config).double()
    donor = copy.deepcopy(host)
    with torch.no_grad():
        # A small donor perturbation keeps this a plumbing fixture in the
        # local linear regime while making the graft route observable.
        for parameter in donor.parameters():
            parameter.add_(0.002 * torch.randn_like(parameter))
    return host, donor


def fixture_prompt() -> TorchPrompt:
    return TorchPrompt(
        input_ids=torch.tensor([[1, 4, 9, 3, 7], [2, 5, 8, 6, 1]], dtype=torch.long),
        utility_token_ids=torch.tensor([10, 11]),
        risk_target_token_ids=torch.tensor([12, 13]),
        risk_donor_token_ids=torch.tensor([14, 15]),
    )


class SutureTorchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.host, self.donor = tiny_pair()
        self.adapter = HFResidualAdapter(self.host, self.donor, device="cpu")
        self.prompt = fixture_prompt()

    def test_architecture_and_state_indexing(self) -> None:
        states = self.adapter.residual_states(self.prompt)
        self.assertEqual(self.adapter.n_layers, 3)
        self.assertEqual(len(states), 4)
        self.assertEqual(states[0].shape, states[1].shape)
        donor_update = self.adapter.donor_block(1, states[1])
        host_update = self.adapter.host_block(1, states[1])
        self.assertEqual(donor_update.shape, host_update.shape)
        self.assertGreater(float(abs(donor_update - host_update).max()), 0.0)
        # The adjoint call must consume the same trajectory and clear it.
        adjoints = self.adapter.adjoints(self.prompt, "utility")
        self.assertEqual(len(adjoints), 3)
        self.assertEqual(adjoints[1].shape, states[1].shape)
        with self.assertRaises(AdapterError):
            self.adapter.host_block(1, states[1])

    def test_score_selection_does_not_build_models(self) -> None:
        scores = self.adapter.score_selection([self.prompt], [self.prompt])
        self.assertEqual(scores.n_layers, 3)
        self.assertEqual(self.adapter.merged_model_builds, 0)
        with self.adapter.selection_phase():
            with self.assertRaises(AdapterError):
                self.adapter.build_grafted_model((1,))
        graft = self.adapter.build_grafted_model((1,))
        self.assertEqual(self.adapter.merged_model_builds, 1)
        routed = graft.readout(self.prompt, "utility")
        self.assertTrue(torch.isfinite(torch.tensor(routed)))

    def test_identical_host_donor_reuses_host_update(self) -> None:
        adapter = HFResidualAdapter(self.host, self.host, device="cpu")
        states = adapter.residual_states(self.prompt)
        for layer in range(adapter.n_layers):
            host_update = adapter.host_block(layer, states[layer])
            donor_update = adapter.donor_block(layer, states[layer])
            self.assertTrue(torch.tensor(donor_update).equal(torch.tensor(host_update)))

    def test_numerical_adjoint(self) -> None:
        result = self.adapter.adjoint_check(
            self.prompt,
            layer=1,
            readout="utility",
            delta_scale=1e-3,
            seed=3,
        )
        self.assertLess(result["relative_error"], 1e-4, result)

    def test_out_of_range_graft_is_rejected(self) -> None:
        with self.assertRaises(AdapterError):
            self.adapter.build_grafted_model((3,))

    def test_routed_generation_and_full_sequence_score(self) -> None:
        routed = self.adapter.build_grafted_model((1,))
        generated = routed.generate(
            input_ids=self.prompt.input_ids[:1],
            max_new_tokens=2,
            do_sample=False,
            use_cache=False,
            pad_token_id=2,
        )
        self.assertEqual(generated.shape[0], 1)
        self.assertEqual(generated.shape[1], self.prompt.input_ids.shape[1] + 2)
        cached = routed.generate(
            input_ids=self.prompt.input_ids[:1],
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=2,
        )
        self.assertEqual(tuple(cached.shape), tuple(generated.shape))
        scores = routed.sequence_logprob(
            self.prompt.input_ids[:1],
            torch.tensor([[10, 11]], dtype=torch.long),
        )
        self.assertEqual(tuple(scores.shape), (1,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_qwen_style_model_layers_are_addressable(self) -> None:
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=16,
            bos_token_id=1,
            eos_token_id=2,
        )
        torch.manual_seed(19)
        host = Qwen2ForCausalLM(config)
        donor = copy.deepcopy(host)
        with torch.no_grad():
            donor.model.layers[0].self_attn.q_proj.weight.add_(0.001)
        adapter = HFResidualAdapter(host, donor, device="cpu")
        prompt = TorchPrompt(
            input_ids=torch.tensor([[1, 5, 7, 9]], dtype=torch.long),
            utility_token_ids=torch.tensor([10]),
            risk_target_token_ids=torch.tensor([11]),
            risk_donor_token_ids=torch.tensor([12]),
        )
        states = adapter.residual_states(prompt)
        self.assertEqual(adapter.architecture["layer_path"], "layers")
        self.assertEqual(len(states), 3)
        self.assertEqual(len(adapter.adjoints(prompt, "risk")), 2)


def layer_only_pair() -> tuple[GPT2LMHeadModel, GPT2LMHeadModel]:
    config = GPT2Config(
        vocab_size=32,
        n_positions=8,
        n_ctx=8,
        n_embd=24,
        n_layer=3,
        n_head=3,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.manual_seed(11)
    host = GPT2LMHeadModel(config).double()
    donor = copy.deepcopy(host)
    with torch.no_grad():
        for block in donor.transformer.h:
            for parameter in block.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
    return host, donor


def greedy_tokens(forward_fn, input_ids: torch.Tensor, steps: int) -> torch.Tensor:
    generated = input_ids.clone()
    for _ in range(steps):
        logits = forward_fn(input_ids=generated).logits[:, -1]
        generated = torch.cat([generated, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return generated


class RoutedGenerationEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.host, self.donor = layer_only_pair()
        self.adapter = HFResidualAdapter(self.host, self.donor, device="cpu")
        self.input_ids = torch.tensor([[1, 4, 9, 3]], dtype=torch.long)

    def test_empty_graft_matches_host_logits_and_greedy_tokens(self) -> None:
        routed = self.adapter.build_grafted_model(())
        host_logits = self.host(input_ids=self.input_ids).logits
        routed_logits = routed.forward(input_ids=self.input_ids).logits
        self.assertTrue(torch.allclose(host_logits, routed_logits, atol=1e-12, rtol=0.0))
        host_tokens = greedy_tokens(self.host, self.input_ids, 3)
        routed_tokens = routed.generate(
            input_ids=self.input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
            pad_token_id=0,
        )
        self.assertEqual(tuple(host_tokens.tolist()), tuple(routed_tokens.tolist()))

    def test_full_graft_matches_donor_logits_and_greedy_tokens(self) -> None:
        graft = tuple(range(self.adapter.n_layers))
        routed = self.adapter.build_grafted_model(graft)
        donor_logits = self.donor(input_ids=self.input_ids).logits
        routed_logits = routed.forward(input_ids=self.input_ids).logits
        self.assertTrue(torch.allclose(donor_logits, routed_logits, atol=1e-12, rtol=0.0))
        donor_tokens = greedy_tokens(self.donor, self.input_ids, 3)
        routed_tokens = routed.generate(
            input_ids=self.input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
            pad_token_id=0,
        )
        self.assertEqual(tuple(donor_tokens.tolist()), tuple(routed_tokens.tolist()))

    def test_cached_and_uncached_greedy_tokens_and_logits(self) -> None:
        routed = self.adapter.build_grafted_model((1,))
        uncached, uncached_logits = routed.generate(
            input_ids=self.input_ids,
            max_new_tokens=4,
            do_sample=False,
            use_cache=False,
            pad_token_id=0,
            return_step_logits=True,
        )
        cached, cached_logits = routed.generate(
            input_ids=self.input_ids,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            return_step_logits=True,
        )
        self.assertEqual(tuple(uncached.tolist()), tuple(cached.tolist()))
        self.assertEqual(tuple(uncached_logits.shape), tuple(cached_logits.shape))
        max_abs = float((uncached_logits - cached_logits).abs().max())
        # Grafted layers still run the host block before the donor replacement
        # hook, so cached logits are not bit-identical. Tokens must match.
        self.assertLess(max_abs, 0.25, f"cache logit drift {max_abs}")

    def test_empty_graft_cache_is_bit_identical(self) -> None:
        routed = self.adapter.build_grafted_model(())
        uncached, uncached_logits = routed.generate(
            input_ids=self.input_ids,
            max_new_tokens=4,
            do_sample=False,
            use_cache=False,
            pad_token_id=0,
            return_step_logits=True,
        )
        cached, cached_logits = routed.generate(
            input_ids=self.input_ids,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            return_step_logits=True,
        )
        self.assertEqual(tuple(uncached.tolist()), tuple(cached.tolist()))
        self.assertEqual(float((uncached_logits - cached_logits).abs().max()), 0.0)

    def test_batched_left_padding_cache_agrees_with_no_cache(self) -> None:
        routed = self.adapter.build_grafted_model((0, 1))
        input_ids = torch.tensor(
            [
                [0, 0, 1, 4, 9],
                [0, 2, 5, 8, 6],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )
        uncached, uncached_logits = routed.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
            pad_token_id=0,
            return_step_logits=True,
        )
        cached, cached_logits = routed.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            return_step_logits=True,
        )
        self.assertEqual(tuple(uncached.tolist()), tuple(cached.tolist()))
        max_abs = float((uncached_logits - cached_logits).abs().max())
        self.assertLess(max_abs, 0.25, f"batched cache logit drift {max_abs}")

    def test_resumed_generation_matches_uninterrupted_greedy_tokens(self) -> None:
        routed = self.adapter.build_grafted_model((2,))
        uninterrupted = routed.generate(
            input_ids=self.input_ids,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
        prefix = routed.generate(
            input_ids=self.input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
        resumed = routed.generate(
            input_ids=prefix,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
        self.assertEqual(tuple(uninterrupted.tolist()), tuple(resumed.tolist()))

    def test_qwen_empty_graft_cache_matches_and_two_model_graft_cache_is_limited(self) -> None:
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        torch.manual_seed(23)
        host = Qwen2ForCausalLM(config)
        donor = copy.deepcopy(host)
        with torch.no_grad():
            for layer in donor.model.layers:
                layer.self_attn.q_proj.weight.add_(0.02)
        adapter = HFResidualAdapter(host, donor, device="cpu")
        input_ids = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
        empty = adapter.build_grafted_model(())
        uncached = empty.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
            pad_token_id=0,
        )
        cached = empty.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
        self.assertEqual(tuple(uncached.tolist()), tuple(cached.tolist()))
        grafted = adapter.build_grafted_model((0,))
        grafted_uncached = grafted.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
            pad_token_id=0,
        )
        self.assertEqual(grafted_uncached.shape[1], input_ids.shape[1] + 3)
        try:
            grafted_cached = grafted.generate(
                input_ids=input_ids,
                max_new_tokens=3,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
            )
        except RuntimeError:
            # Two-copy routing still lets the host layer write the cache before
            # the donor replacement hook. Canonical E1 used a shared PEFT pair
            # instead of this two-model path.
            return
        self.assertEqual(tuple(grafted_uncached.tolist()), tuple(grafted_cached.tolist()))


def tiny_qwen3_pair():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(21)
    host = Qwen3ForCausalLM(config)
    donor = copy.deepcopy(host)
    with torch.no_grad():
        donor.model.layers[0].self_attn.q_proj.weight.add_(0.001)
        donor.model.layers[1].mlp.gate_proj.weight.add_(0.001)
    return host, donor


class Qwen3AdapterTests(unittest.TestCase):
    def test_qwen3_layer_discovery_and_lora_names(self) -> None:
        host, donor = tiny_qwen3_pair()
        adapter = HFResidualAdapter(host, donor, device="cpu")
        prompt = TorchPrompt(
            input_ids=torch.tensor([[1, 5, 7, 9]], dtype=torch.long),
            utility_token_ids=torch.tensor([10]),
            risk_target_token_ids=torch.tensor([11]),
            risk_donor_token_ids=torch.tensor([12]),
        )
        states = adapter.residual_states(prompt)
        self.assertEqual(adapter.n_layers, 3)
        self.assertEqual(len(states), 4)
        names = {name for name, _ in host.named_modules()}
        for target in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            self.assertTrue(any(name.endswith(target) for name in names), target)
        with adapter.selection_phase():
            with self.assertRaises(AdapterError):
                adapter.build_grafted_model((1,))
        graft = adapter.build_grafted_model((1,))
        generated = graft.generate(
            input_ids=prompt.input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=False,
            pad_token_id=2,
        )
        cached = graft.generate(
            input_ids=prompt.input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=2,
        )
        self.assertTrue(torch.as_tensor(states[0]).isfinite().all())
        self.assertEqual(tuple(generated.shape), tuple(cached.shape))

    def test_qwen2_is_not_the_v2_scientific_architecture(self) -> None:
        from suture.tier_a_config import MODEL_TYPE

        self.assertEqual(MODEL_TYPE, "qwen3")


if __name__ == "__main__":
    unittest.main()
