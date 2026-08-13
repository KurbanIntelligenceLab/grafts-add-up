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

from paper_iclr.suture_torch import AdapterError, HFResidualAdapter, TorchPrompt


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


if __name__ == "__main__":
    unittest.main()
