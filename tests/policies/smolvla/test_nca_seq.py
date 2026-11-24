"""SmolVLA Stage 1 on whole-grid NCA sequences (`stage1_input="seq"`): NTP, MAE and flow matching.

Tiny models on CPU: the VLM is built from its config without weights (the config and tokenizer must be in the
local Hugging Face cache) and cut to 2 layers.

    HF_HUB_OFFLINE=1 pytest tests/policies/smolvla/test_nca_seq.py
"""

import importlib
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import OBS_SEQ, SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

IMAGE_KEY = "observation.images.nca"
REF_COMMIT = "1b30b62c"  # nca-pretrain: the code every earlier Stage 1 and fine-tuning run used
FRAMES, VOCAB, BATCH = 3, 97, 2


def make_config(**kwargs) -> SmolVLAConfig:
    seq_len = FRAMES * 64 * (3 if kwargs.get("seq_token", "bin16") != "bin16" else 1)
    input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
    }
    if kwargs.get("stage1_input") == "seq":
        input_features[OBS_SEQ] = PolicyFeature(type=FeatureType.STATE, shape=(seq_len,))
    try:
        return SmolVLAConfig(
            input_features=input_features,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
            load_vlm_weights=False,
            num_vlm_layers=2,
            device="cpu",
            **kwargs,
        )
    except OSError as e:  # no network and the VLM config is not cached
        pytest.skip(f"SmolVLM2 config not available offline: {e}")


def make_policy(seed: int = 0, **kwargs) -> SmolVLAPolicy:
    cfg = make_config(**kwargs)
    torch.manual_seed(seed)
    try:
        return SmolVLAPolicy(cfg)
    except OSError as e:
        pytest.skip(f"SmolVLM2 config / tokenizer not available offline: {e}")


def seq_policy(objective: str, token: str = "bin16", **kwargs) -> SmolVLAPolicy:
    policy = make_policy(
        stage1_input="seq",
        stage1_objective=objective,
        seq_token=token,
        seq_frames=FRAMES,
        seq_vocab=VOCAB,
        **kwargs,
    )
    policy.eval()  # no dropout anywhere; keeps the frozen VLM in eval as in training
    return policy


def make_batch(seq: torch.Tensor | None = None, bsize: int = BATCH) -> dict:
    g = torch.Generator().manual_seed(1)
    batch = {
        IMAGE_KEY: torch.rand(bsize, 3, 512, 512, generator=g),
        OBS_STATE: torch.randn(bsize, 12, generator=g),
        ACTION: torch.randn(bsize, 50, 12, generator=g),
        OBS_LANGUAGE_TOKENS: torch.full((bsize, 1), 198, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(bsize, 1, dtype=torch.bool),
    }
    if seq is not None:
        batch[OBS_SEQ] = seq[:, None]  # the dataset adds a length-1 time axis
    return batch


def random_tokens(seed: int = 2, bsize: int = BATCH) -> torch.Tensor:
    return torch.randint(0, VOCAB, (bsize, FRAMES * 64), generator=torch.Generator().manual_seed(seed))


def captured_logits(policy: SmolVLAPolicy, batch: dict, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the loss without gradients and return (per-token losses [B, L], the logits seq_head produced)."""
    out = []
    hook = policy.model.seq_head.register_forward_hook(lambda _m, _i, o: out.append(o.detach().clone()))
    try:
        with torch.no_grad():
            images, img_masks = policy.prepare_images(batch)
            seq = batch[OBS_SEQ].reshape(batch[OBS_SEQ].shape[0], -1)
            losses, _, _ = policy.model.forward_seq(
                images,
                img_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                policy.prepare_state(batch),
                seq,
                **kwargs,
            )
    finally:
        hook.remove()
    return losses, torch.cat(out)


# --- 1. Defaults are unchanged ------------------------------------------------------------------------------------
def load_reference_module():
    """The smolvla policy as of REF_COMMIT, imported next to the current one as lerobot.policies.smolvla_ref."""
    repo = Path(__file__).resolve().parents[3]
    tmp = Path(tempfile.mkdtemp(prefix="smolvla_ref_"))
    for name in ("configuration_smolvla.py", "modeling_smolvla.py", "smolvlm_with_expert.py"):
        res = subprocess.run(
            ["git", "-C", str(repo), "show", f"{REF_COMMIT}:src/lerobot/policies/smolvla/{name}"],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            pytest.skip(f"commit {REF_COMMIT} not available: {res.stderr.strip()}")
        (tmp / name).write_text(
            res.stdout.replace('register_subclass("smolvla")', 'register_subclass("smolvla_ref")')
        )
    pkg = types.ModuleType("lerobot.policies.smolvla_ref")
    pkg.__path__ = [str(tmp)]
    sys.modules["lerobot.policies.smolvla_ref"] = pkg
    return importlib.import_module("lerobot.policies.smolvla_ref.modeling_smolvla")


def test_default_is_unchanged_bit_for_bit():
    """With every new option at its default, the model's weights after init and the loss on a fixed batch, noise
    and time are bit-identical to the code before this change."""
    ref = load_reference_module()
    new = make_policy(seed=0)
    cfg = make_config()
    ref_cfg = ref.SmolVLAConfig(
        input_features=cfg.input_features,
        output_features=cfg.output_features,
        load_vlm_weights=False,
        num_vlm_layers=2,
        device="cpu",
    )
    torch.manual_seed(0)
    old = ref.SmolVLAPolicy(ref_cfg)

    new_sd, old_sd = new.state_dict(), old.state_dict()
    assert list(new_sd) == list(old_sd)
    assert all(torch.equal(new_sd[k], old_sd[k]) for k in old_sd)

    g = torch.Generator().manual_seed(3)
    noise = torch.randn(BATCH, 50, 32, generator=g)
    time = torch.rand(BATCH, generator=g) * 0.999 + 0.001
    for policy in (new, old):
        policy.train()
    loss_new, dict_new = new.forward(make_batch(), noise=noise.clone(), time=time.clone())
    loss_old, dict_old = old.forward(make_batch(), noise=noise.clone(), time=time.clone())
    assert torch.equal(loss_new, loss_old) and dict_new == dict_old
    per_new, _ = new.forward(make_batch(), noise=noise.clone(), time=time.clone(), reduction="none")
    per_old, _ = old.forward(make_batch(), noise=noise.clone(), time=time.clone(), reduction="none")
    assert torch.equal(per_new, per_old)
    # sampling (the fine-tuned policy's eval path) too
    new.eval(), old.eval()
    with torch.no_grad():
        a_new = new.predict_action_chunk(make_batch(), noise=noise.clone())
        a_old = old.predict_action_chunk(make_batch(), noise=noise.clone())
    assert torch.equal(a_new, a_old)


def test_default_has_no_seq_modules_and_config_checks():
    cfg = make_config()
    assert cfg.stage1_input == "chunk" and cfg.stage1_objective == "flow"
    with pytest.raises(ValueError):
        make_config(stage1_input="seq", stage1_objective="flow", seq_token="bin16")
    with pytest.raises(ValueError):
        make_config(stage1_input="seq", stage1_objective="ntp", seq_token="rgb3")
    with pytest.raises(ValueError):
        make_config(stage1_input="seq", stage1_objective="mae", mae_mask="blocks")
    with pytest.raises(ValueError):
        make_config(stage1_input="sequence")
    assert make_config(stage1_input="seq", stage1_objective="flow", seq_token="rgb12").seq_len == 16 * 16
    assert make_config(stage1_input="seq", stage1_objective="mae").seq_len == 1024


# --- 2. NTP ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ntp():
    return seq_policy("ntp")


def test_ntp_modules_and_loss(ntp):
    keys = {k for k in ntp.state_dict() if k.startswith("model.seq_")}
    assert keys == {
        "model.seq_pos_embed.weight",
        "model.seq_embed.weight",
        "model.seq_head.weight",
        "model.seq_head.bias",
    }
    assert ntp.model.seq_embed.weight.shape[0] == VOCAB + 1 and ntp.model.seq_pos_embed.weight.shape[0] == 64
    tokens = random_tokens()
    loss, info = ntp.forward(make_batch(tokens))
    per_sample, _ = ntp.forward(make_batch(tokens), reduction="none")
    assert (
        loss.ndim == 0 and per_sample.shape == (BATCH,) and torch.allclose(per_sample.mean(), loss, atol=1e-5)
    )
    assert abs(loss.item() - torch.log(torch.tensor(float(VOCAB))).item()) < 1.0  # untrained: about ln(vocab)
    frames = [info[f"ntp_loss_frame_{i}"] for i in range(FRAMES)]
    assert abs(sum(frames) / FRAMES - info["loss"]) < 1e-4 and 0.0 <= info["seq_acc"] <= 1.0


def test_ntp_no_leak_from_later_tokens(ntp):
    """Changing tokens after position t changes no logit at positions <= t (and does change later ones)."""
    tokens = random_tokens()
    t = 70  # inside frame 1
    changed = tokens.clone()
    changed[:, t + 1 :] = (changed[:, t + 1 :] + 1 + torch.arange(changed.shape[1] - t - 1)) % VOCAB
    _, a = captured_logits(ntp, make_batch(tokens))
    _, b = captured_logits(ntp, make_batch(changed))
    a, b = a.view(BATCH, -1, VOCAB), b.view(BATCH, -1, VOCAB)
    # position p reads tokens < p, so logits up to and including t + 1 can't see the change
    assert torch.allclose(a[:, : t + 2], b[:, : t + 2], atol=1e-5)
    assert (a[:, t + 2 :] - b[:, t + 2 :]).abs().max() > 1e-3
    # and position p does read token p - 1: changing token t alone moves logits from t + 1 on, not before
    one = tokens.clone()
    one[:, t] = (one[:, t] + 1) % VOCAB
    _, c = captured_logits(ntp, make_batch(one))
    c = c.view(BATCH, -1, VOCAB)
    assert torch.allclose(a[:, : t + 1], c[:, : t + 1], atol=1e-5)
    assert (a[:, t + 1] - c[:, t + 1]).abs().max() > 1e-3


# --- 3. MAE ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mae():
    return seq_policy("mae")


def test_mae_no_leak_from_hidden_tokens(mae):
    assert "model.seq_mask_embed" in mae.state_dict() and mae.model.seq_embed.weight.shape[0] == VOCAB
    tokens = random_tokens()
    mask = mae.model.sample_mae_mask(BATCH, tokens.device)
    changed = torch.where(mask, (tokens + 5) % VOCAB, tokens)  # other true values under the mask
    la, a = captured_logits(mae, make_batch(tokens), mae_mask=mask)
    lb, b = captured_logits(mae, make_batch(changed), mae_mask=mask)
    assert a.shape == (int(mask.sum()), VOCAB)  # logits only for hidden tokens
    assert torch.equal(a, b)  # the input never contained them
    assert (la[~mask] == 0).all() and (la[mask] > 0).all() and not torch.equal(la, lb)  # targets did change
    # attention is bidirectional: a visible token late in the sequence changes outputs of hidden tokens before it
    late = int((~mask[0]).nonzero().max())
    assert mask[0, :late].any()
    moved = tokens.clone()
    moved[0, late] = (moved[0, late] + 1) % VOCAB
    _, c = captured_logits(mae, make_batch(moved), mae_mask=mask)
    n_before = int(mask[0, :late].sum())
    assert (a[:n_before] - c[:n_before]).abs().max() > 1e-4


def test_mae_loss_is_over_hidden_tokens_only(mae):
    tokens = random_tokens()
    mask = mae.model.sample_mae_mask(BATCH, tokens.device)
    batch = make_batch(tokens)
    loss, info = mae._forward_seq(batch, mae_mask=mask)
    losses, _ = captured_logits(mae, make_batch(tokens), mae_mask=mask)
    assert torch.allclose(loss, losses[mask].mean(), atol=1e-5) and "ntp_loss_frame_0" not in info


@pytest.mark.parametrize("kind", ["token", "frame", "tube"])
def test_mae_mask_patterns(kind):
    cfg = SimpleNamespace(seq_frames=16, seq_tokens_per_frame=64, mae_mask=kind, mae_mask_ratio=0.75)
    torch.manual_seed(0)
    mask = VLAFlowMatching.sample_mae_mask(SimpleNamespace(config=cfg), 8, "cpu")
    assert mask.shape == (8, 1024) and mask.dtype == torch.bool
    assert (mask.sum(dim=1) == 768).all()  # 75% of every sample
    grid = mask.view(8, 16, 64)
    if kind == "frame":  # whole frames: 12 of 16
        per_frame = grid.sum(dim=2)
        assert ((per_frame == 0) | (per_frame == 64)).all() and ((per_frame == 64).sum(dim=1) == 12).all()
    elif kind == "tube":  # the same 48 of 64 patch positions in every frame
        assert (grid == grid[:, :1]).all() and (grid[:, 0].sum(dim=1) == 48).all()
    else:  # random tokens: not frames, not tubes
        assert not (grid == grid[:, :1]).all() and len(grid.sum(dim=2).unique()) > 2
    assert not torch.equal(mask[0], mask[1])  # a new mask per sample


def test_chunked_cross_entropy_matches_plain(mae):
    """The checkpointed, chunked head gives the same loss and gradients as one plain cross-entropy."""
    model = mae.model
    g = torch.Generator().manual_seed(4)
    hidden = torch.randn(50, model.seq_head.in_features, generator=g, requires_grad=True)
    targets = torch.randint(0, VOCAB, (50,), generator=g)
    ce, correct = model._seq_cross_entropy(hidden, targets, chunk=7)
    ce.mean().backward()
    got = hidden.grad.clone(), model.seq_head.weight.grad.clone()
    hidden.grad = None
    model.seq_head.weight.grad = None
    logits = model.seq_head(hidden)
    plain = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    plain.mean().backward()
    assert torch.allclose(ce, plain, atol=1e-6) and torch.equal(correct, logits.argmax(-1) == targets)
    assert torch.allclose(got[0], hidden.grad, atol=1e-7) and torch.allclose(
        got[1], model.seq_head.weight.grad, atol=1e-7
    )
    model.seq_head.weight.grad = None
    model.seq_head.bias.grad = None


# --- 4. Flow matching on RGB sequences ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def flow3():
    return seq_policy("flow", "rgb3", seq_norm_mean=(0.1, 0.2, 0.3), seq_norm_std=(2.0, 4.0, 8.0))


@pytest.fixture(scope="module")
def flow12():
    return seq_policy("flow", "rgb12")


def rgb_seq(bsize: int = BATCH) -> torch.Tensor:
    return torch.randn(bsize, FRAMES * 64 * 3, generator=torch.Generator().manual_seed(5))


def test_flow_tokens_rgb3_normalized(flow3):
    seq = rgb_seq()
    tokens = flow3.model.seq_to_tokens(seq)
    assert tokens.shape == (BATCH, FRAMES * 64, 3)
    raw = seq.view(BATCH, FRAMES, 64, 3)
    f, patch = 2, 37
    want = (raw[1, f, patch] - torch.tensor([0.1, 0.2, 0.3])) / torch.tensor([2.0, 4.0, 8.0])
    assert torch.allclose(tokens[1, f * 64 + patch], want)


def test_flow_tokens_rgb12_grouping_by_hand(flow12):
    # value = 1000 * frame + 10 * patch + channel, so every number says where it came from
    f, p, c = torch.meshgrid(torch.arange(FRAMES), torch.arange(64), torch.arange(3), indexing="ij")
    seq = (1000 * f + 10 * p + c).float().reshape(1, -1)
    tokens = flow12.model.seq_to_tokens(seq)
    assert tokens.shape == (1, FRAMES * 16, 12)
    # frame 1, group (R=1, C=2) = patches (2, 4), (2, 5), (3, 4), (3, 5) = 20, 21, 28, 29: TL, TR, BL, BR x RGB
    want = [1000 + 10 * patch + ch for patch in (20, 21, 28, 29) for ch in range(3)]
    assert tokens[0, 16 + 4 * 1 + 2].tolist() == want
    assert tokens[0, 0].tolist() == [10 * patch + ch for patch in (0, 1, 8, 9) for ch in range(3)]


@pytest.mark.parametrize("name,dim,per_frame", [("flow3", 3, 64), ("flow12", 12, 16)])
def test_flow_loss_shape_and_padded_dims_ignored(name, dim, per_frame, request):
    policy = request.getfixturevalue(name)
    assert not any(k.startswith(("model.seq_embed", "model.seq_head")) for k in policy.state_dict())
    assert policy.model.seq_pos_embed.weight.shape[0] == per_frame
    length = FRAMES * per_frame
    g = torch.Generator().manual_seed(6)
    noise, time = torch.randn(BATCH, length, 32, generator=g), torch.rand(BATCH, generator=g) * 0.999 + 0.001
    batch = make_batch(rgb_seq())

    def run():
        with torch.no_grad():
            per_sample, info = policy.forward(dict(batch), noise=noise, time=time, reduction="none")
            loss, _ = policy.forward(dict(batch), noise=noise, time=time)
        return per_sample, loss, info

    per_sample, loss, info = run()
    assert (
        per_sample.shape == (BATCH,)
        and torch.allclose(per_sample.mean(), loss, atol=1e-6)
        and set(info) == {"loss"}
    )
    out = policy.model.action_out_proj
    saved = out.weight.data.clone()
    out.weight.data[dim:] += 1.0  # outputs for the padded dims
    assert torch.equal(run()[0], per_sample)
    out.weight.data[dim - 1] += 1.0  # a real dim
    assert not torch.allclose(run()[0], per_sample)
    out.weight.data.copy_(saved)


# --- 5. Gradients ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["ntp", "mae", "flow3"])
def test_gradients_reach_the_expert_not_the_vlm(name, request):
    policy = request.getfixturevalue(name)
    policy.zero_grad()
    batch = make_batch(rgb_seq() if name == "flow3" else random_tokens())
    loss, _ = policy.forward(batch)
    loss.backward()
    model = policy.model
    expert = dict(model.vlm_with_expert.lm_expert.named_parameters())
    with_grad = [n for n, p in expert.items() if p.grad is not None and p.grad.abs().sum() > 0]
    n_layers = len(model.vlm_with_expert.lm_expert.layers)
    for i in range(n_layers):
        for part in ("self_attn.q_proj", "self_attn.o_proj", "mlp"):
            assert any(n.startswith(f"layers.{i}.{part}") for n in with_grad), (
                f"no gradient in layer {i} {part}"
            )
    vlm = list(model.vlm_with_expert.vlm.parameters())
    assert all(not p.requires_grad and p.grad is None for p in vlm)
    assert model.state_proj.weight.grad is not None and model.seq_pos_embed.weight.grad.abs().sum() > 0
    if name == "flow3":
        assert (
            model.action_in_proj.weight.grad.abs().sum() > 0
            and model.action_time_mlp_in.weight.grad.abs().sum() > 0
        )
    else:
        assert model.seq_embed.weight.grad.abs().sum() > 0 and model.seq_head.weight.grad.abs().sum() > 0
        assert (
            model.action_in_proj.weight.grad is None
        )  # unused by MAE / NTP (and dropped at the swap anyway)
    policy.zero_grad()


# --- 6. The sequence reaches forward through the real processor pipeline ---------------------------------------------
def test_processor_leaves_seq_untouched_even_with_stats(flow3):
    """lerobot types observation.seq as a STATE vector; with stats present it would be normalized element-wise."""
    from lerobot.policies.factory import make_pre_post_processors

    cfg = flow3.config
    n = cfg.input_features[OBS_SEQ].shape[0]
    stats = {
        OBS_SEQ: {"mean": torch.full((n,), 5.0), "std": torch.full((n,), 2.0)},
        OBS_STATE: {"mean": torch.full((12,), 1.0), "std": torch.full((12,), 2.0)},
        ACTION: {"mean": torch.zeros(12), "std": torch.ones(12)},
    }
    try:
        pre, _ = make_pre_post_processors(cfg, dataset_stats=stats)
    except OSError as e:
        pytest.skip(f"tokenizer not available offline: {e}")
    seq = rgb_seq()
    raw = {
        IMAGE_KEY: torch.rand(BATCH, 3, 512, 512),
        OBS_STATE: torch.ones(BATCH, 12) * 3.0,
        ACTION: torch.zeros(BATCH, 50, 12),
        OBS_SEQ: seq[:, None].clone(),
        "task": [""] * BATCH,
    }
    out = pre(raw)
    assert torch.equal(out[OBS_SEQ], seq[:, None])  # untouched
    assert torch.allclose(out[OBS_STATE], torch.ones(BATCH, 12))  # (3 - 1) / 2: the state is still normalized
    loss, _ = flow3.forward(out)
    assert torch.isfinite(loss)


def test_seq_reaches_forward_from_a_real_dataset(tmp_path, monkeypatch):
    """A 10-rule binary dataset from the phase 2 builder -> lerobot dataset -> dataloader -> preprocessor ->
    policy.forward: the model receives exactly the stored token IDs, as int64."""
    nca = pytest.importorskip("nca")
    scripts = Path(nca.__file__).resolve().parents[1] / "scripts"
    if not (scripts / "build_seq_binary.py").exists():
        pytest.skip("nca sequence builders not found")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_cache"))
    env = {**__import__("os").environ, "HF_DATASETS_CACHE": str(tmp_path / "hf_cache")}
    for split, n in (("train", 8), ("heldout", 2)):
        subprocess.run(
            [sys.executable, str(scripts / "build_seq_binary.py"), "--root", str(tmp_path / f"{split}_000"),
             "--split", split, "--shard", "0", "--n-rules", str(n), "--rules-per-batch", "100"],
            check=True, capture_output=True, env=env, cwd=scripts,
        )  # fmt: skip
    root = tmp_path / "nca_seq_bin_test"
    subprocess.run(
        [sys.executable, str(scripts / "merge_seq_shards.py"), "--out", str(root),
         str(tmp_path / "train_000"), str(tmp_path / "heldout_000")],
        check=True, capture_output=True, env=env, cwd=scripts,
    )  # fmt: skip

    import numpy as np

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    meta = LeRobotDatasetMetadata("local/nca_seq_bin_test", root=root)
    cfg = SmolVLAConfig(
        stage1_input="seq", stage1_objective="ntp", load_vlm_weights=False, num_vlm_layers=2, device="cpu",
        push_to_hub=False,
    )  # fmt: skip
    torch.manual_seed(0)
    policy = make_policy(cfg, ds_meta=meta)  # features from the dataset, as lerobot-train does
    assert cfg.input_features[OBS_SEQ].shape == (1024,) and OBS_SEQ not in meta.stats
    dataset = LeRobotDataset(
        "local/nca_seq_bin_test", root=root, delta_timestamps=resolve_delta_timestamps(cfg, meta)
    )
    pre, _ = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)))
    stored = batch[OBS_SEQ].clone()
    assert stored.dtype == torch.int64 and stored.shape == (4, 1, 1024)
    batch = pre(batch)

    seen = {}
    original = policy.model.forward_seq

    def spy(images, img_masks, lang_tokens, lang_masks, state, seq, **kwargs):
        seen["seq"], seen["state"], seen["image"] = seq, state, images[0]
        return original(images, img_masks, lang_tokens, lang_masks, state, seq, **kwargs)

    policy.model.forward_seq = spy
    loss, info = policy.forward(batch)
    assert torch.isfinite(loss) and "ntp_loss_frame_15" in info
    assert seen["seq"].dtype == torch.int64 and torch.equal(seen["seq"], stored[:, 0])
    assert not seen["state"].any()  # zero state stays zero through the normalizer
    assert seen["image"].shape == (4, 3, 512, 512) and set(seen["image"].unique().tolist()) <= {-1.0, 1.0}
    verify = np.load(root / "nca_build/verify.npz")
    assert np.array_equal(stored[0, 0].numpy(), verify["seq"][0])
