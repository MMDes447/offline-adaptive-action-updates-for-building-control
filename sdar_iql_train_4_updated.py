"""
SDAR-IQL training script for offline RL building control â€” FLAT HYBRID ACTOR
============================================================================

This version matches the v8 SDAR-ready POOLED dataset (meter-free 29-dim obs,
per-zone lighting normalization) and uses a FLAT proposal policy.

Sequences are built PER-MINIBATCH (see build_sequence_batch); the full
(N, seq_len, obs_dim) tensors are never materialized. This is required for the
735,826-transition pooled dataset, where materializing them would need ~24.6 GB.

Observation: 29-dim (meter-free)
    [0:5]    zone temperatures                    (5 zones)
    [5:10]   lighting rates                       (5 zones)
    [10:14]  transmitted solar                    (4 windows)
    [14:18]  exterior irradiance                  (4 windows)
    [18:22]  WPI / glare proxy                    (4 windows)
    [22]     hvac_electricity_demand_rate         (facility HVAC, W)
    [23]     hour_sin
    [24]     hour_cos
    [25]     doy_sin
    [26]     doy_cos
    [27]     is_occupied
    [28]     is_preconditioning

Dataset action fields:
    actions:         (T, 19), normalized SDAR executed action
                     [0:4] glazing tier normalized from {0,1,2,3} to [-1,1]
                     [4:19] continuous actions normalized to [-1,1]
                     NOTE: lighting dims [4:9] are normalized PER ZONE by each
                     zone's installed max (see light_ranges_per_zone in the NPZ).
    prev_actions:    (T, 19), previous executed normalized action a^-
    selection_masks: (T, 19), b in {0,1}^19, where b=1 means act/change
    action_mixes:    (T, 19), a_mix=(1-b)*a^- + b*xi, xi=-2

Internal action format for proposal actor:
    [0:4]   glazing class IDs 0..3
    [4:19]  continuous normalized values

Networks:
    GRUEncoder:       z_t = GRU(o_{t-K:t})
    rep_t:            [z_t ; o_t]
    SDARSelector b:   beta(b_t | rep_t, prev_action_t)
    Flat proposal pi: pi(a_prop,t | rep_t, action_mix_t)
    TwinQ:            Q(rep_t, a_exec,t, b_t)
    ValueNet:         V(rep_t)

INFERENCE SAFEGUARDS (see load_and_act):
    * hard unoccupied-lights-off constraint, applied to the EXECUTED action and
      written back into prev_action_norm so later repeats repeat lights-off;
    * executed_selection_mask, so (a_exec - a_prev) * (1 - b_exec) == 0 holds
      even when the safeguard overrides a repeat dim;
    * optional deadband sync, keeping prev_action_norm equal to what is actually
      actuated after the cool >= heat + deadband_min correction.
"""

from __future__ import annotations

from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical, Normal


# ============================================================
# PATHS / RUN CONFIG
# ============================================================

DATASET_PATH = "pooled_sdar_experts.npz"
RUN_NAME = "sdar_iql_v9_14episodes_gru2_seq144"

CHECKPOINT_DIR = Path("checkpoints") / RUN_NAME
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("Checkpoint directory:", CHECKPOINT_DIR.resolve())
_test_file = CHECKPOINT_DIR / "_write_test.tmp"
try:
    with open(_test_file, "w") as f:
        f.write("ok")
    _test_file.unlink()
    print("Checkpoint directory write test: OK")
except PermissionError:
    raise PermissionError(f"No write permission for checkpoint directory: {CHECKPOINT_DIR.resolve()}")
except Exception as e:
    raise RuntimeError(f"Could not write to checkpoint directory: {CHECKPOINT_DIR.resolve()}\nError: {e}")

BEST_CKPT_PATH = CHECKPOINT_DIR / f"sdar_iql_best_{RUN_NAME}.pt"
FINAL_CKPT_PATH = CHECKPOINT_DIR / f"sdar_iql_final_{RUN_NAME}.pt"
CKPT_PREFIX = CHECKPOINT_DIR / f"sdar_iql_{RUN_NAME}_checkpoint_epoch"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DIMENSIONS â€” v8 SDAR POOLED DATASET (meter-free 29-dim obs)
# ============================================================

OBS_DIM = 29
ACT_DIM = 19
ACTION_DIM = 19
MASK_DIM = 19

NUM_GLAZING_HEADS = 4
NUM_GLAZING_CLASSES = 4
CONT_ACT_DIM = 15
GLAZING_ONEHOT_DIM = NUM_GLAZING_HEADS * NUM_GLAZING_CLASSES

# Q sees executed action plus mask:
#   onehot glazing 16 + continuous 15 + mask 19 = 50
Q_INPUT_ACT_DIM = GLAZING_ONEHOT_DIM + CONT_ACT_DIM + MASK_DIM

HIDDEN_DIM = 256
# Temporal (sin/cos + occupancy flags) starts here; these obs dims are passed
# through un-normalized. In the 29-dim meter-free layout this is index 23.
TEMPORAL_DIM_START = 23

LIGHT_DIM = 5
HEAT_DIM = 5
COOL_DIM = 5

# Per-zone lighting decode ranges (row order = 5 zones). This MUST match the
# controller's LIGHT_RANGE_PER_ZONE / the dataset's `light_ranges_per_zone`.
# The generic LIGHT_RANGE below is a nominal fallback only; the real decode
# uses LIGHT_RANGES_PER_ZONE, and load_and_act overrides it from the NPZ/ckpt.
LIGHT_RANGE = (0.0, 2500.0)
LIGHT_RANGES_PER_ZONE = np.array(
    [[0.0, 2231.0], [0.0, 2231.0], [0.0, 1412.0], [0.0, 1412.0], [0.0, 10586.0]],
    dtype=np.float32,
)
HEAT_RANGE = (16.0, 24.0)
COOL_RANGE = (22.0, 28.0)
# Equal-setpoint expert: allow cool_sp == heat_sp (no forced deadband at inference).
DEADBAND_MIN = 0.0

LATENT_DIM = 128
GRU_LAYERS = 2
SEQ_LEN = 144
REP_DIM = LATENT_DIM + OBS_DIM

SELECTOR_INPUT_DIM = REP_DIM + ACTION_DIM     # [rep, prev_action]
PROPOSAL_INPUT_DIM = REP_DIM + ACTION_DIM     # [rep, action_mix]


# ============================================================
# INFERENCE-TIME OPERATIONAL CONSTRAINTS
# ============================================================
# Used only by load_and_act (see the safeguard block in get_action).
# LIGHTS_OFF_NORM = -1.0 is 0 W in EVERY zone, because lighting dims are
# normalized per zone from [0, installed_W] -> [-1, 1].

OCCUPANCY_OBS_INDEX = 27
LIGHTING_ACTION_SLICE = slice(4, 9)
COOLING_ACTION_SLICE = slice(14, 19)
LIGHTS_OFF_NORM = -1.0


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 256          # 256 for the 735k pooled dataset; raise to 512 if GPU is comfortable
LR_ACTOR = 1e-4
LR_SELECTOR = 1e-4
LR_QVALUE = 1e-4
LR_VALUE = 1e-4
LR_ENCODER = 5e-5         # lower: every loss backprops into the shared GRU encoder

NUM_EPOCHS = 150          # 150 x 1000 = 150,000 updates (see note on 200k alternative)
STEPS_PER_EPOCH = 1000

EXPECTILE = 0.7
TEMPERATURE = 1.0
GAMMA = 0.99
TAU_TARGET = 0.005
MAX_ADV_WEIGHT = 20.0     # advantage-weight clamp (was 100); caps a possibly-miscalibrated critic

GRAD_CLIP_NORM = 1.0
LOG_EVERY = 1

USE_COMPILE = False
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# Actor-loss coefficients kept < 1: selector BCE is SUMMED over 19 dims and the
# proposal log-prob is also summed, so at 1.0 they dominate the shared GRU grads.
LAMBDA_SELECTOR = 0.25
LAMBDA_PROPOSAL = 0.25

ACTION_MASK_VALUE = -2.0
AVAILABLE_GLAZING_STATES = ["sr2_ec01", "sr2_ec02", "sr2_ec03", "sr2_ec04"]


# ============================================================
# ACTION HELPERS
# ============================================================

def decode_sdar_actions(actions_norm: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized SDAR action vector to actor-training format:
      first 4 dims: glazing class IDs 0..3
      last 15 dims: continuous normalized values in [-1,1]
    """
    glz_norm = actions_norm[:, :NUM_GLAZING_HEADS]
    glz_ids = torch.round((glz_norm + 1.0) * 0.5 * (NUM_GLAZING_CLASSES - 1))
    glz_ids = glz_ids.long().clamp(0, NUM_GLAZING_CLASSES - 1).float()
    cont = actions_norm[:, NUM_GLAZING_HEADS:]
    return torch.cat([glz_ids, cont], dim=-1)


def glazing_to_onehot(actions_actor_format: torch.Tensor) -> torch.Tensor:
    glazing_ids = actions_actor_format[:, :NUM_GLAZING_HEADS].long()
    onehot = F.one_hot(glazing_ids, NUM_GLAZING_CLASSES).float()
    onehot_flat = onehot.reshape(onehot.shape[0], -1)
    cont = actions_actor_format[:, NUM_GLAZING_HEADS:]
    return torch.cat([onehot_flat, cont], dim=-1)


def make_q_action_input(actions_actor_format: torch.Tensor, selection_masks: torch.Tensor) -> torch.Tensor:
    return torch.cat([glazing_to_onehot(actions_actor_format), selection_masks], dim=-1)


def ids_to_normalized_glazing(glz_ids: torch.Tensor) -> torch.Tensor:
    return 2.0 * glz_ids.float() / float(NUM_GLAZING_CLASSES - 1) - 1.0


def normalized_glazing_to_ids(glz_norm: np.ndarray) -> np.ndarray:
    ids = np.round((glz_norm + 1.0) * 0.5 * (NUM_GLAZING_CLASSES - 1))
    return np.clip(ids, 0, NUM_GLAZING_CLASSES - 1).astype(np.int64)


# ============================================================
# NORMALIZERS
# ============================================================

class Normalizer:
    def __init__(self, data: torch.Tensor | None = None, eps: float = 1e-6):
        if data is not None:
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0).clamp(min=eps)
        else:
            self.mean = None
            self.std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, sd):
        self.mean = sd["mean"]
        self.std = sd["std"]


class ScalarNormalizer:
    def __init__(self, data: torch.Tensor | None = None, eps: float = 1e-6):
        if data is not None:
            self.mean = data.mean()
            self.std = data.std().clamp(min=eps)
        else:
            self.mean = None
            self.std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, sd):
        self.mean = sd["mean"]
        self.std = sd["std"]


# ============================================================
# NETWORKS
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers += [nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(obs_dim, latent_dim, num_layers, batch_first=True)
        self.latent_dim = latent_dim
        self.num_layers = num_layers

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        output, h_n = self.gru(obs_seq, hidden)
        return output[:, -1, :], h_n

    def init_hidden(self, batch_size: int, device: str):
        return torch.zeros(self.num_layers, batch_size, self.latent_dim, device=device)


class SDARSelector(nn.Module):
    """beta(b_t | rep_t, prev_action_t): outputs 19 Bernoulli logits."""
    def __init__(self, input_dim: int, mask_dim: int = 19, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(input_dim, mask_dim, hidden_dim, n_layers=3)

    def forward(self, rep: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([rep, prev_action], dim=-1))

    def sample(self, rep: torch.Tensor, prev_action: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        probs = torch.sigmoid(self.forward(rep, prev_action))
        if deterministic:
            return (probs >= 0.5).float()
        return torch.bernoulli(probs)


class SDARFlatHybridActor(nn.Module):
    """
    FLAT proposal policy pi(a_prop | rep, action_mix).

    This intentionally does NOT condition glazing_2 on glazing_1, heat on light,
    cooling on heat, etc. All heads are produced from one shared trunk.
    """
    LOG_STD_MIN = -2.0
    LOG_STD_MAX = 1.0

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = MLP(input_dim, hidden_dim, hidden_dim, n_layers=2)
        self.glazing_logits_head = nn.Linear(hidden_dim, NUM_GLAZING_HEADS * NUM_GLAZING_CLASSES)
        self.cont_mean = nn.Linear(hidden_dim, CONT_ACT_DIM)
        self.cont_log_std_head = nn.Linear(hidden_dim, CONT_ACT_DIM)

    def _forward_dist(self, proposal_rep: torch.Tensor):
        h = self.trunk(proposal_rep)
        glazing_logits = self.glazing_logits_head(h).view(
            -1, NUM_GLAZING_HEADS, NUM_GLAZING_CLASSES
        )
        cont_mean = self.cont_mean(h)
        raw_log_std = self.cont_log_std_head(h)
        squashed = torch.tanh(raw_log_std)
        cont_log_std = (
            self.LOG_STD_MIN
            + 0.5 * (squashed + 1.0) * (self.LOG_STD_MAX - self.LOG_STD_MIN)
        )
        return glazing_logits, cont_mean, cont_log_std

    def forward(self, proposal_rep: torch.Tensor):
        return self._forward_dist(proposal_rep)

    def log_prob(self, proposal_rep: torch.Tensor, actions_actor_format: torch.Tensor) -> torch.Tensor:
        glazing_logits, cont_mean, cont_log_std = self._forward_dist(proposal_rep)
        glz_ids = actions_actor_format[:, :NUM_GLAZING_HEADS].long()
        cont = actions_actor_format[:, NUM_GLAZING_HEADS:]

        log_p_glz = Categorical(logits=glazing_logits).log_prob(glz_ids).sum(dim=-1)
        log_p_cont = Normal(cont_mean, cont_log_std.exp()).log_prob(cont).sum(dim=-1)
        return (log_p_glz + log_p_cont).unsqueeze(-1)

    def masked_log_prob(
        self,
        proposal_rep: torch.Tensor,
        actions_actor_format: torch.Tensor,
        selection_masks: torch.Tensor,
    ) -> torch.Tensor:
        glazing_logits, cont_mean, cont_log_std = self._forward_dist(proposal_rep)
        glz_ids = actions_actor_format[:, :NUM_GLAZING_HEADS].long()
        cont = actions_actor_format[:, NUM_GLAZING_HEADS:]

        m_glz = selection_masks[:, :NUM_GLAZING_HEADS]
        m_cont = selection_masks[:, NUM_GLAZING_HEADS:]

        log_p_glz_dims = Categorical(logits=glazing_logits).log_prob(glz_ids)
        log_p_cont_dims = Normal(cont_mean, cont_log_std.exp()).log_prob(cont)

        log_p_glz = (log_p_glz_dims * m_glz).sum(dim=-1)
        log_p_cont = (log_p_cont_dims * m_cont).sum(dim=-1)
        return (log_p_glz + log_p_cont).unsqueeze(-1)

    def log_prob_split(
        self,
        proposal_rep: torch.Tensor,
        actions_actor_format: torch.Tensor,
        selection_masks: torch.Tensor,
    ):
        glazing_logits, cont_mean, cont_log_std = self._forward_dist(proposal_rep)
        glz_ids = actions_actor_format[:, :NUM_GLAZING_HEADS].long()
        cont = actions_actor_format[:, NUM_GLAZING_HEADS:]

        m_glz = selection_masks[:, :NUM_GLAZING_HEADS]
        m_cont = selection_masks[:, NUM_GLAZING_HEADS:]

        log_p_glz_dims = Categorical(logits=glazing_logits).log_prob(glz_ids)
        log_p_cont_dims = Normal(cont_mean, cont_log_std.exp()).log_prob(cont)

        log_p_glz = (log_p_glz_dims * m_glz).sum(dim=-1)
        log_p_cont = (log_p_cont_dims * m_cont).sum(dim=-1)
        return log_p_glz.unsqueeze(-1), log_p_cont.unsqueeze(-1)

    def log_std_diag(self, proposal_rep: torch.Tensor, actions_actor_format: torch.Tensor | None = None) -> torch.Tensor:
        _, _, cont_log_std = self._forward_dist(proposal_rep)
        return cont_log_std

    def sample(self, proposal_rep: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        glazing_logits, cont_mean, cont_log_std = self._forward_dist(proposal_rep)
        glz_dist = Categorical(logits=glazing_logits)
        if deterministic:
            glazing_action = glz_dist.probs.argmax(dim=-1).float()
            cont_action = cont_mean
        else:
            glazing_action = glz_dist.sample().float()
            cont_action = Normal(cont_mean, cont_log_std.exp()).rsample()
        return torch.cat([glazing_action, cont_action], dim=-1)


class TwinQ(nn.Module):
    def __init__(self, input_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = MLP(input_dim + act_dim, 1, hidden_dim)
        self.q2 = MLP(input_dim + act_dim, 1, hidden_dim)

    def forward(self, rep: torch.Tensor, action_input: torch.Tensor):
        sa = torch.cat([rep, action_input], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, rep: torch.Tensor, action_input: torch.Tensor):
        q1, q2 = self.forward(rep, action_input)
        return torch.min(q1, q2)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.v = MLP(input_dim, 1, hidden_dim)

    def forward(self, rep: torch.Tensor):
        return self.v(rep)


# ============================================================
# IQL / SDAR LOSSES
# ============================================================

def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def iql_value_loss(qnet_target: TwinQ, vnet: ValueNet, rep: torch.Tensor, action_q_input: torch.Tensor, expectile: float):
    with torch.no_grad():
        q_target = qnet_target.min_q(rep, action_q_input)
    v = vnet(rep)
    return expectile_loss(q_target - v, expectile)


def iql_q_loss(qnet: TwinQ, vnet: ValueNet, rep: torch.Tensor, action_q_input: torch.Tensor, reward: torch.Tensor, rep_next: torch.Tensor, done: torch.Tensor, gamma: float):
    with torch.no_grad():
        v_next = vnet(rep_next)
        td_target = reward + gamma * (1.0 - done) * v_next
    q1, q2 = qnet(rep, action_q_input)
    return ((q1 - td_target).pow(2) + (q2 - td_target).pow(2)).mean()


def compute_iql_weights(qnet_target: TwinQ, vnet: ValueNet, rep: torch.Tensor, action_q_input: torch.Tensor, temperature: float):
    with torch.no_grad():
        q = qnet_target.min_q(rep, action_q_input)
        v = vnet(rep)
        advantage = q - v
        advantage_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        weights = (advantage_norm / temperature).exp().clamp(max=MAX_ADV_WEIGHT)
    return weights, advantage


def sdar_selector_loss(selector: SDARSelector, rep: torch.Tensor, prev_action: torch.Tensor, selection_mask: torch.Tensor, weights: torch.Tensor):
    logits = selector(rep, prev_action)
    bce = F.binary_cross_entropy_with_logits(logits, selection_mask, reduction="none")
    loss_per_sample = bce.sum(dim=-1, keepdim=True)
    return (weights * loss_per_sample).mean()


def sdar_proposal_loss(actor: SDARFlatHybridActor, rep: torch.Tensor, action_mix: torch.Tensor, action_actor_format: torch.Tensor, selection_mask: torch.Tensor, weights: torch.Tensor):
    proposal_rep = torch.cat([rep, action_mix], dim=-1)
    log_prob = actor.masked_log_prob(proposal_rep, action_actor_format, selection_mask)
    return -(weights * log_prob).mean()


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ============================================================
# DATA LOADING / SEQUENCES
# ============================================================

def load_dataset(path: str):
    if not path.endswith(".npz"):
        raise ValueError("SDAR setup expects the v8 NPZ dataset. Use .npz, not CSV.")

    data = np.load(path, allow_pickle=True)

    obs = torch.from_numpy(data["observations"]).float()
    actions_norm = torch.from_numpy(data["actions"]).float()
    prev_actions = torch.from_numpy(data["prev_actions"]).float()
    selection_masks = torch.from_numpy(data["selection_masks"]).float()
    action_mixes = torch.from_numpy(data["action_mixes"]).float()
    rew = torch.from_numpy(data["rewards"]).float().unsqueeze(-1)
    next_obs = torch.from_numpy(data["next_observations"]).float()
    done = torch.from_numpy(data["terminals"]).float().unsqueeze(-1)

    actions_actor_format = decode_sdar_actions(actions_norm)

    print("Loaded SDAR NPZ dataset:")
    print("  observations:     ", tuple(obs.shape))
    print("  actions_norm:     ", tuple(actions_norm.shape))
    print("  actions_actor_fmt:", tuple(actions_actor_format.shape))
    print("  prev_actions:     ", tuple(prev_actions.shape))
    print("  selection_masks:  ", tuple(selection_masks.shape))
    print("  action_mixes:     ", tuple(action_mixes.shape))
    print("  rewards:          ", tuple(rew.shape))
    print("  next_observations:", tuple(next_obs.shape))
    print("  terminals:        ", tuple(done.shape))

    assert obs.shape[1] == OBS_DIM, f"OBS_DIM={OBS_DIM}, dataset obs_dim={obs.shape[1]}"
    assert next_obs.shape == obs.shape, f"next_obs.shape={next_obs.shape}, obs.shape={obs.shape}"
    assert actions_norm.shape[1] == ACT_DIM, f"ACT_DIM={ACT_DIM}, dataset action_dim={actions_norm.shape[1]}"
    assert prev_actions.shape[1] == ACTION_DIM
    assert selection_masks.shape[1] == MASK_DIM
    assert action_mixes.shape[1] == ACTION_DIM

    if torch.isnan(obs).any() or torch.isnan(next_obs).any():
        raise ValueError("NaNs found in observations.")
    if torch.isnan(actions_norm).any() or torch.isnan(selection_masks).any() or torch.isnan(action_mixes).any():
        raise ValueError("NaNs found in SDAR action fields.")
    if selection_masks.min() < -1e-6 or selection_masks.max() > 1.0 + 1e-6:
        raise ValueError("selection_masks must be in [0,1].")

    return obs, actions_actor_format, prev_actions, selection_masks, action_mixes, rew, next_obs, done


def build_sequences(obs: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor, seq_len: int):
    """Reference full-dataset sequence builder. Kept for the unit test and as the
    canonical definition that build_sequence_batch reproduces per-minibatch."""
    N, obs_dim = obs.shape
    obs_seqs = torch.zeros(N, seq_len, obs_dim, dtype=obs.dtype)
    next_obs_seqs = torch.zeros(N, seq_len, obs_dim, dtype=obs.dtype)

    for i in range(N):
        t = seq_len - 1
        obs_seqs[i, t] = obs[i]
        j = i - 1
        while t > 0 and j >= 0 and done[j].item() == 0:
            t -= 1
            obs_seqs[i, t] = obs[j]
            j -= 1

        t = seq_len - 1
        next_obs_seqs[i, t] = next_obs[i]
        if done[i].item() == 0:
            j = i
            while t > 0 and j >= 0 and done[j].item() == 0:
                t -= 1
                next_obs_seqs[i, t] = obs[j]
                j -= 1

    return obs_seqs, next_obs_seqs


def make_episode_start_indices(done: torch.Tensor) -> torch.Tensor:
    """For each transition i, return the index of the first transition in the
    episode that contains i (i.e. the transition right after the previous
    terminal, or 0). Used to zero-pad sequences that would cross an episode
    boundary, exactly matching build_sequences' `while ... done[j]==0` stop."""
    done = done.squeeze(-1).bool()
    n = len(done)

    start_mask = torch.zeros(n, dtype=torch.bool)
    start_mask[0] = True
    start_mask[1:] = done[:-1]

    indices = torch.arange(n)
    markers = torch.where(start_mask, indices, torch.zeros_like(indices))

    return torch.cummax(markers, dim=0).values


def build_sequence_batch(
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
    episode_starts: torch.Tensor,
    indices: torch.Tensor,
    seq_len: int,
):
    """Build (obs_seq, next_obs_seq) for a minibatch of `indices` only.

    Bit-for-bit equivalent to build_sequences()[indices], but never materializes
    the full (N, seq_len, obs_dim) tensors. Zero-pads positions that fall before
    the episode start; the next-state sequence's history is fully zeroed when the
    transition is terminal (done=1), matching the reference builder."""
    device = indices.device
    batch_size = len(indices)
    obs_dim = obs.shape[1]

    starts = episode_starts[indices].unsqueeze(1)

    # Current-state sequence ending at observation[index]
    offsets = torch.arange(
        -(seq_len - 1), 1, device=device
    )

    positions = indices.unsqueeze(1) + offsets.unsqueeze(0)
    valid = positions >= starts

    obs_seq = obs[positions.clamp(min=0)]
    obs_seq = obs_seq * valid.unsqueeze(-1)

    # Next-state sequence ending at next_observation[index]
    previous_offsets = torch.arange(
        -(seq_len - 2), 1, device=device
    )

    previous_positions = (
        indices.unsqueeze(1)
        + previous_offsets.unsqueeze(0)
    )

    previous_valid = previous_positions >= starts
    previous_valid &= done[indices].reshape(-1, 1) < 0.5

    next_obs_seq = torch.zeros(
        batch_size,
        seq_len,
        obs_dim,
        dtype=obs.dtype,
        device=device,
    )

    next_obs_seq[:, :-1] = (
        obs[previous_positions.clamp(min=0)]
        * previous_valid.unsqueeze(-1)
    )

    next_obs_seq[:, -1] = next_obs[indices]

    return obs_seq, next_obs_seq


def test_build_sequences():
    print("=" * 60)
    print("UNIT TEST: build_sequences")
    print("=" * 60)

    obs = torch.tensor([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4], [5.0, 0.5]])
    next_obs = torch.tensor([[2.0, 0.2], [3.0, 0.3], [4.0, 0.4], [5.0, 0.5], [6.0, 0.6]])
    done = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]).unsqueeze(-1)

    obs_seqs, next_obs_seqs = build_sequences(obs, next_obs, done, seq_len=3)
    passed = True

    obs_tests = [
        (0, torch.tensor([[0, 0], [0, 0], [1.0, 0.1]])),
        (1, torch.tensor([[0, 0], [1.0, 0.1], [2.0, 0.2]])),
        (2, torch.tensor([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])),
        (3, torch.tensor([[0, 0], [0, 0], [4.0, 0.4]])),
        (4, torch.tensor([[0, 0], [4.0, 0.4], [5.0, 0.5]])),
    ]
    for idx, expected in obs_tests:
        if not torch.allclose(obs_seqs[idx], expected):
            passed = False

    next_tests = [
        (0, torch.tensor([[0, 0], [1.0, 0.1], [2.0, 0.2]])),
        (1, torch.tensor([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])),
        (2, torch.tensor([[0, 0], [0, 0], [4.0, 0.4]])),
        (3, torch.tensor([[0, 0], [4.0, 0.4], [5.0, 0.5]])),
        (4, torch.tensor([[4.0, 0.4], [5.0, 0.5], [6.0, 0.6]])),
    ]
    for idx, expected in next_tests:
        if not torch.allclose(next_obs_seqs[idx], expected):
            passed = False

    # Also assert the per-minibatch builder matches the reference on the same data.
    starts = make_episode_start_indices(done)
    all_idx = torch.arange(obs.shape[0])
    bat_o, bat_n = build_sequence_batch(obs, next_obs, done, starts, all_idx, seq_len=3)
    if not (torch.allclose(bat_o, obs_seqs) and torch.allclose(bat_n, next_obs_seqs)):
        passed = False
        print("  build_sequence_batch DIFFERS from build_sequences!")

    print(f"  {'ALL PASSED' if passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    print()
    return passed


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _sd(mod: nn.Module):
    return mod._orig_mod.state_dict() if hasattr(mod, "_orig_mod") else mod.state_dict()


def get_checkpoint_config():
    return {
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "action_dim": ACTION_DIM,
        "mask_dim": MASK_DIM,
        "latent_dim": LATENT_DIM,
        "gru_layers": GRU_LAYERS,
        "seq_len": SEQ_LEN,
        "rep_dim": REP_DIM,
        "q_input_act_dim": Q_INPUT_ACT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_glazing_heads": NUM_GLAZING_HEADS,
        "num_glazing_classes": NUM_GLAZING_CLASSES,
        "cont_act_dim": CONT_ACT_DIM,
        "light_dim": LIGHT_DIM,
        "heat_dim": HEAT_DIM,
        "cool_dim": COOL_DIM,
        "temporal_dim_start": TEMPORAL_DIM_START,
        "expectile": EXPECTILE,
        "temperature": TEMPERATURE,
        "gamma": GAMMA,
        "max_adv_weight": MAX_ADV_WEIGHT,
        "actor_class": "SDARFlatHybridActor",
        "selector_class": "SDARSelector",
        "proposal_input_dim": PROPOSAL_INPUT_DIM,
        "selector_input_dim": SELECTOR_INPUT_DIM,
        "q_uses_mask": True,
        "dataset_path": DATASET_PATH,
        "run_name": RUN_NAME,
        "action_mask_value": ACTION_MASK_VALUE,
        "deadband_min": DEADBAND_MIN,
        # Setpoint decode ranges are recorded so inference can assert they match.
        "heat_range": list(HEAT_RANGE),
        "cool_range": list(COOL_RANGE),
        "light_ranges_per_zone": LIGHT_RANGES_PER_ZONE.tolist(),
        "obs_layout": {
            "temp": [0, 5],
            "light_rate": [5, 10],
            "solar": [10, 14],
            "ext_irr": [14, 18],
            "wpi": [18, 22],
            "hvac_demand_rate": [22, 23],
            "temporal": [23, 29],
        },
        "action_layout": {
            "glazing": [0, 4],
            "lighting": [4, 9],
            "heating": [9, 14],
            "cooling": [14, 19],
        },
    }


def save_checkpoint(path: Path, epoch: int, gru_encoder, selector, actor, qnet, qnet_target, vnet, obs_normalizer, rew_normalizer, opt_encoder=None, opt_selector=None, opt_actor=None, opt_q=None, opt_v=None):
    payload = {
        "epoch": epoch,
        "gru_encoder": _sd(gru_encoder),
        "selector": _sd(selector),
        "actor": _sd(actor),
        "qnet": _sd(qnet),
        "qnet_target": qnet_target.state_dict(),
        "vnet": _sd(vnet),
        "obs_normalizer": obs_normalizer.state_dict(),
        "rew_normalizer": rew_normalizer.state_dict(),
        "config": get_checkpoint_config(),
    }
    if opt_encoder is not None:
        payload.update({
            "opt_encoder": opt_encoder.state_dict(),
            "opt_selector": opt_selector.state_dict(),
            "opt_actor": opt_actor.state_dict(),
            "opt_q": opt_q.state_dict(),
            "opt_v": opt_v.state_dict(),
        })
    torch.save(payload, path)


# ============================================================
# TRAINING
# ============================================================

def train():
    if not test_build_sequences():
        print("Aborting: sequence unit test failed.")
        return

    print(f"Device: {DEVICE}")
    print("Actor: SDARFlatHybridActor")
    print("Selector: SDARSelector")
    print("Proposal policy is FLAT: glazing logits + continuous Gaussian are produced in parallel from the same trunk.")
    print(f"POMDP: GRU encoder (latent={LATENT_DIM}, layers={GRU_LAYERS}, seq_len={SEQ_LEN})")
    print(f"Observation: {OBS_DIM} dims (temporal starts at {TEMPORAL_DIM_START})")
    print(f"Representation: [z_t ; o_t] dim={REP_DIM}")
    print(f"Q action input dim: {Q_INPUT_ACT_DIM} = onehot_glz({GLAZING_ONEHOT_DIM}) + cont({CONT_ACT_DIM}) + mask({MASK_DIM})")
    print(f"IQL: expectile={EXPECTILE}, temperature={TEMPERATURE}, gamma={GAMMA}, tau_target={TAU_TARGET}, max_adv_w={MAX_ADV_WEIGHT}")
    print(f"Optim: AMP={USE_AMP} ({AMP_DTYPE}), batch={BATCH_SIZE}, lambda_sel={LAMBDA_SELECTOR}, lambda_prop={LAMBDA_PROPOSAL}")
    print()

    obs_raw, action_actor_raw, prev_actions_raw, masks_raw, mixes_raw, rew_raw, next_obs_raw, done_raw = load_dataset(DATASET_PATH)
    N = obs_raw.shape[0]
    print(f"Dataset: {N} transitions ({DATASET_PATH})")

    print("\n=== Terminal / episode checks ===")
    terminal_indices = torch.where(done_raw.squeeze(-1) > 0.5)[0]
    terminal_sum = int(done_raw.sum().item())
    print("terminal sum:", terminal_sum)
    print("terminal indices:", terminal_indices.tolist())
    assert terminal_sum >= 1, "Dataset must contain at least one terminal flag."
    assert terminal_indices[-1].item() == N - 1, (
        f"Last transition should be terminal. Got last terminal at {terminal_indices[-1].item()}, expected {N - 1}."
    )
    print("Episode terminal check: OK")

    print("\n=== Raw data ===")
    print(f"  Obs (sensors)    mean: {obs_raw[:, :TEMPORAL_DIM_START].mean():.4f}, std: {obs_raw[:, :TEMPORAL_DIM_START].std():.4f}")
    print(f"  Obs (temporal)   mean: {obs_raw[:, TEMPORAL_DIM_START:].mean():.4f}, std: {obs_raw[:, TEMPORAL_DIM_START:].std():.4f}")
    print(f"  Action glz ids   mean: {action_actor_raw[:, :4].mean():.4f}, std: {action_actor_raw[:, :4].std():.4f}")
    print(f"  Action cont norm mean: {action_actor_raw[:, 4:].mean():.4f}, std: {action_actor_raw[:, 4:].std():.4f}")
    print(f"  Mask             mean: {masks_raw.mean():.4f}, min: {masks_raw.min():.1f}, max: {masks_raw.max():.1f}")
    print(f"  Reward           mean: {rew_raw.mean():.4f}, std: {rew_raw.std():.4f}, min: {rew_raw.min():.4f}, max: {rew_raw.max():.4f}")

    glz_ids_raw = action_actor_raw[:, :NUM_GLAZING_HEADS].long().reshape(-1)
    counts = torch.bincount(glz_ids_raw, minlength=NUM_GLAZING_CLASSES)
    pct = 100.0 * counts.float() / counts.sum()
    print("  Glazing dist:", {f"sr2_ec0{i+1}": f"{p:.1f}%" for i, p in enumerate(pct.tolist())})

    print("\n=== Mask diagnostics ===")
    groups = {
        "glazing 0:4": masks_raw[:, 0:4],
        "lighting 4:9": masks_raw[:, 4:9],
        "heating 9:14": masks_raw[:, 9:14],
        "cooling 14:19": masks_raw[:, 14:19],
        "overall": masks_raw,
    }
    for name, m in groups.items():
        mean_b = float(m.mean())
        apr = 1.0 / max(mean_b, 1e-9)
        print(f"  {name:16s} mean(b)={mean_b:.4f} APR={apr:.2f}")

    obs_normalizer = Normalizer(obs_raw)
    obs_normalizer.mean[TEMPORAL_DIM_START:] = 0.0
    obs_normalizer.std[TEMPORAL_DIM_START:] = 1.0

    obs_norm = obs_normalizer.normalize(obs_raw)
    next_obs_norm = obs_normalizer.normalize(next_obs_raw)

    # SDAR fields are already in normalized action space. Continuous actions are not re-normalized.
    action_actor_norm = action_actor_raw
    prev_actions_norm = prev_actions_raw
    selection_masks = masks_raw
    action_mixes = mixes_raw

    rew_normalizer = ScalarNormalizer(rew_raw)
    rew_norm = rew_normalizer.normalize(rew_raw)

    print("\n=== After normalization ===")
    print(f"  Obs (sensors)    mean: {obs_norm[:, :TEMPORAL_DIM_START].mean():.4f}, std: {obs_norm[:, :TEMPORAL_DIM_START].std():.4f}")
    print(f"  Obs (temporal)   mean: {obs_norm[:, TEMPORAL_DIM_START:].mean():.4f}, std: {obs_norm[:, TEMPORAL_DIM_START:].std():.4f}  (pass-through)")
    print(f"  Action cont norm mean: {action_actor_norm[:, 4:].mean():.4f}, std: {action_actor_norm[:, 4:].std():.4f}")
    print(f"  Reward           mean: {rew_norm.mean():.4f}, std: {rew_norm.std():.4f}")

    # ------------------------------------------------------------------
    # Per-minibatch sequences: keep only FLAT (N, dim) tensors on-device.
    # The (N, seq_len, obs_dim) sequence tensors are NEVER materialized
    # (they would need ~24.6 GB for the 735k pooled dataset). Instead,
    # build_sequence_batch reconstructs each minibatch's windows on the fly.
    # ------------------------------------------------------------------
    print(f"\nSequences built per-minibatch (seq_len={SEQ_LEN}); "
          f"full (N, seq_len, obs_dim) tensors are never materialized.")

    obs_all = obs_norm.to(DEVICE)
    next_obs_all = next_obs_norm.to(DEVICE)
    done_all = done_raw.to(DEVICE)

    episode_start_all = make_episode_start_indices(done_raw).to(DEVICE)

    action_actor_all = action_actor_norm.to(DEVICE)
    prev_actions_all = prev_actions_norm.to(DEVICE)
    selection_masks_all = selection_masks.to(DEVICE)
    action_mixes_all = action_mixes.to(DEVICE)

    action_q_all = make_q_action_input(action_actor_all, selection_masks_all)

    rew_all = rew_norm.to(DEVICE)

    # Sanity check: a sequence built AT an episode start must be zero before the
    # current step and must never reach back across the preceding terminal.
    if len(terminal_indices) > 1:
        check_starts = terminal_indices[:-1] + 1
        check_starts = check_starts[check_starts < N].to(DEVICE)
        if len(check_starts) > 0:
            obs_seq_chk, _ = build_sequence_batch(
                obs_all, next_obs_all, done_all, episode_start_all, check_starts, SEQ_LEN
            )
            assert torch.allclose(
                obs_seq_chk[:, :-1], torch.zeros_like(obs_seq_chk[:, :-1])
            ), "build_sequence_batch crossed an episode boundary at an episode start!"
            print("Episode-boundary sequence check (per-batch builder): OK")

    flat_mem_mb = (
        obs_all.nbytes + next_obs_all.nbytes + action_actor_all.nbytes
        + prev_actions_all.nbytes + selection_masks_all.nbytes + action_mixes_all.nbytes
        + action_q_all.nbytes + rew_all.nbytes + done_all.nbytes + episode_start_all.nbytes
    ) / 1e6
    per_batch_seq_mb = 2 * BATCH_SIZE * SEQ_LEN * OBS_DIM * obs_all.element_size() / 1e6
    print(f"Dataset GPU memory (flat tensors): {flat_mem_mb:.1f} MB")
    print(f"Per-minibatch sequence tensors:    {per_batch_seq_mb:.1f} MB (built on the fly)\n")

    del obs_raw, action_actor_raw, prev_actions_raw, masks_raw, mixes_raw, rew_raw, next_obs_raw, done_raw
    del obs_norm, next_obs_norm, action_actor_norm, prev_actions_norm, selection_masks, action_mixes, rew_norm

    gru_encoder = GRUEncoder(OBS_DIM, LATENT_DIM, GRU_LAYERS).to(DEVICE)
    selector = SDARSelector(SELECTOR_INPUT_DIM, MASK_DIM, HIDDEN_DIM).to(DEVICE)
    actor = SDARFlatHybridActor(PROPOSAL_INPUT_DIM, HIDDEN_DIM).to(DEVICE)
    qnet = TwinQ(REP_DIM, Q_INPUT_ACT_DIM, HIDDEN_DIM).to(DEVICE)
    qnet_target = TwinQ(REP_DIM, Q_INPUT_ACT_DIM, HIDDEN_DIM).to(DEVICE)
    qnet_target.load_state_dict(qnet.state_dict())
    vnet = ValueNet(REP_DIM, HIDDEN_DIM).to(DEVICE)

    if USE_COMPILE:
        gru_encoder = torch.compile(gru_encoder)
        selector = torch.compile(selector)
        actor = torch.compile(actor)
        qnet = torch.compile(qnet)
        vnet = torch.compile(vnet)
        print("torch.compile applied to gru_encoder, selector, actor, qnet, vnet")

    total_params = (
        sum(p.numel() for p in gru_encoder.parameters())
        + sum(p.numel() for p in selector.parameters())
        + sum(p.numel() for p in actor.parameters())
        + sum(p.numel() for p in qnet.parameters())
        + sum(p.numel() for p in vnet.parameters())
    )
    print(f"Trainable params: {total_params:,}\n")

    opt_encoder = Adam(gru_encoder.parameters(), lr=LR_ENCODER)
    opt_selector = Adam(selector.parameters(), lr=LR_SELECTOR)
    opt_actor = Adam(actor.parameters(), lr=LR_ACTOR)
    opt_q = Adam(qnet.parameters(), lr=LR_QVALUE)
    opt_v = Adam(vnet.parameters(), lr=LR_VALUE)

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_q_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss_v = 0.0
        epoch_loss_q = 0.0
        epoch_loss_beta = 0.0
        epoch_loss_pi = 0.0
        epoch_loss_a = 0.0
        diag = {}

        for step in range(STEPS_PER_EPOCH):
            idx = torch.randint(0, N, (BATCH_SIZE,), device=DEVICE)

            obs_seq, next_obs_seq = build_sequence_batch(
                obs_all, next_obs_all, done_all, episode_start_all, idx, SEQ_LEN
            )
            action_actor = action_actor_all[idx]
            prev_action = prev_actions_all[idx]
            selection_mask = selection_masks_all[idx]
            action_mix = action_mixes_all[idx]
            action_q_input = action_q_all[idx]
            reward = rew_all[idx]
            done = done_all[idx]

            opt_encoder.zero_grad(set_to_none=True)
            opt_selector.zero_grad(set_to_none=True)
            opt_actor.zero_grad(set_to_none=True)
            opt_q.zero_grad(set_to_none=True)
            opt_v.zero_grad(set_to_none=True)

            with torch.autocast(device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP):
                z, _ = gru_encoder(obs_seq)
                z_next, _ = gru_encoder(next_obs_seq)

                obs_curr = obs_seq[:, -1, :]
                next_obs_curr = next_obs_seq[:, -1, :]
                rep = torch.cat([z, obs_curr], dim=-1)
                rep_next = torch.cat([z_next, next_obs_curr], dim=-1)

                loss_v = iql_value_loss(qnet_target, vnet, rep, action_q_input, EXPECTILE)
                loss_q = iql_q_loss(qnet, vnet, rep, action_q_input, reward, rep_next, done, GAMMA)

                weights, advantage = compute_iql_weights(qnet_target, vnet, rep, action_q_input, TEMPERATURE)

                loss_beta = sdar_selector_loss(selector, rep, prev_action, selection_mask, weights)
                loss_pi = sdar_proposal_loss(actor, rep, action_mix, action_actor, selection_mask, weights)
                loss_a = LAMBDA_SELECTOR * loss_beta + LAMBDA_PROPOSAL * loss_pi

                loss_total = loss_v + loss_q + loss_a

            scaler.scale(loss_total).backward()

            scaler.unscale_(opt_encoder)
            scaler.unscale_(opt_selector)
            scaler.unscale_(opt_actor)
            scaler.unscale_(opt_q)
            scaler.unscale_(opt_v)

            torch.nn.utils.clip_grad_norm_(gru_encoder.parameters(), GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(selector.parameters(), GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(vnet.parameters(), GRAD_CLIP_NORM)

            scaler.step(opt_encoder)
            scaler.step(opt_selector)
            scaler.step(opt_actor)
            scaler.step(opt_q)
            scaler.step(opt_v)
            scaler.update()

            soft_update(qnet_target, qnet, TAU_TARGET)

            epoch_loss_v += loss_v.item()
            epoch_loss_q += loss_q.item()
            epoch_loss_beta += loss_beta.item()
            epoch_loss_pi += loss_pi.item()
            epoch_loss_a += loss_a.item()

            if step == STEPS_PER_EPOCH - 1:
                with torch.no_grad():
                    z_diag, _ = gru_encoder(obs_seq)
                    rep_diag = torch.cat([z_diag, obs_seq[:, -1, :]], dim=-1)

                    q_val = qnet_target.min_q(rep_diag, action_q_input)
                    v_val = vnet(rep_diag)
                    adv = q_val - v_val
                    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-6)
                    w = (adv_norm / TEMPERATURE).exp().clamp(max=MAX_ADV_WEIGHT)

                    selector_logits = selector(rep_diag, prev_action)
                    selector_probs = torch.sigmoid(selector_logits)
                    pred_mask = (selector_probs >= 0.5).float()
                    mask_acc = (pred_mask == selection_mask).float().mean()

                    proposal_rep_diag = torch.cat([rep_diag, action_mix], dim=-1)
                    glz_lp, cont_lp = actor.log_prob_split(proposal_rep_diag, action_actor, selection_mask)
                    cont_log_std = actor.log_std_diag(proposal_rep_diag)

                    diag = {
                        "q_mean": q_val.mean().item(),
                        "v_mean": v_val.mean().item(),
                        "adv_mean": adv.mean().item(),
                        "adv_std": adv.std().item(),
                        "w_mean": w.mean().item(),
                        "w_max": w.max().item(),
                        "clip_frac": (w >= 0.999 * MAX_ADV_WEIGHT).float().mean().item(),
                        "mask_acc": mask_acc.item(),
                        "mask_pred": selector_probs.mean().item(),
                        "mask_data": selection_mask.mean().item(),
                        "glz_logp_masked": glz_lp.mean().item(),
                        "cont_logp_masked": cont_lp.mean().item(),
                        "logstd_mean": cont_log_std.mean().item(),
                        "logstd_min": cont_log_std.min().item(),
                        "logstd_max": cont_log_std.max().item(),
                        "z_mean": z_diag.mean().item(),
                        "z_std": z_diag.std().item(),
                    }

        avg_v = epoch_loss_v / STEPS_PER_EPOCH
        avg_q = epoch_loss_q / STEPS_PER_EPOCH
        avg_beta = epoch_loss_beta / STEPS_PER_EPOCH
        avg_pi = epoch_loss_pi / STEPS_PER_EPOCH
        avg_a = epoch_loss_a / STEPS_PER_EPOCH

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | V: {avg_v:.6f} | Q: {avg_q:.6f} | "
            f"Beta: {avg_beta:.4f} | Pi: {avg_pi:.4f} | ActorTotal: {avg_a:.4f}"
        )

        if epoch % LOG_EVERY == 0 and diag:
            print(
                f"  diag: q={diag['q_mean']:.3f} v={diag['v_mean']:.3f} "
                f"adv={diag['adv_mean']:.3f}+/-{diag['adv_std']:.3f} "
                f"w={diag['w_mean']:.2f}(max={diag['w_max']:.1f}, clip={diag['clip_frac']:.2%}) "
                f"mask_acc={diag['mask_acc']:.3f} mask_p={diag['mask_pred']:.3f} mask_data={diag['mask_data']:.3f} "
                f"logp_glz(masked)={diag['glz_logp_masked']:.2f} "
                f"logp_cont(masked)={diag['cont_logp_masked']:.2f} "
                f"logstd={diag['logstd_mean']:.2f}[{diag['logstd_min']:.2f},{diag['logstd_max']:.2f}]"
            )
            print(f"  latent: z_mean={diag['z_mean']:.4f} z_std={diag['z_std']:.4f}")

        if avg_q < best_q_loss:
            best_q_loss = avg_q
            save_checkpoint(BEST_CKPT_PATH, epoch, gru_encoder, selector, actor, qnet, qnet_target, vnet, obs_normalizer, rew_normalizer)

        if epoch % 50 == 0:
            save_checkpoint(
                Path(f"{CKPT_PREFIX}{epoch}.pt"),
                epoch,
                gru_encoder,
                selector,
                actor,
                qnet,
                qnet_target,
                vnet,
                obs_normalizer,
                rew_normalizer,
                opt_encoder=opt_encoder,
                opt_selector=opt_selector,
                opt_actor=opt_actor,
                opt_q=opt_q,
                opt_v=opt_v,
            )
            print("  -> Saved checkpoint")

    save_checkpoint(FINAL_CKPT_PATH, NUM_EPOCHS, gru_encoder, selector, actor, qnet, qnet_target, vnet, obs_normalizer, rew_normalizer)

    print(f"\nTraining complete. Saved {FINAL_CKPT_PATH}")
    print(f"Best Q loss: {best_q_loss:.6f} ({BEST_CKPT_PATH})")

    return gru_encoder, selector, actor, qnet, vnet, obs_normalizer, rew_normalizer


# ============================================================
# SDAR INFERENCE HELPER  (safeguard-aware)
# ============================================================

def load_and_act(
    checkpoint_path=BEST_CKPT_PATH,
    enforce_unoccupied_lights_off: bool = True,
    sync_prev_with_deadband: bool = True,
    selector_mode: str = "sample",
    proposal_deterministic: bool = True,
    selector_seed: int = 20260728,
    constant_update_rates: dict[str, float] | None = None,
    periodic_phase_seed: int | None = None,
    lighting_change_tolerance_w: float = 1.0,
    setpoint_change_tolerance_c: float = 0.05,
):
    """
    Build a stateful (get_action, reset) pair from a checkpoint.

    enforce_unoccupied_lights_off
        Hard operational constraint: when obs[OCCUPANCY_OBS_INDEX] < 0.5, the
        five lighting dims of the EXECUTED action are forced to LIGHTS_OFF_NORM
        (= 0 W in every zone). The safeguarded action is written back into
        prev_action_norm, so a later b=0 repeats lights-off rather than a stale
        nonzero value. Doing this here (not in the callback) is what keeps the
        agent's internal memory equal to what the building actually receives.

    sync_prev_with_deadband
        Keep prev_action_norm equal to the actuated setpoints after the
        cool_sp >= heat_sp + deadband_min correction. Set False to reproduce the
        old behavior exactly (useful when isolating the lighting fix).

    selector_mode
        "threshold" reproduces the old deterministic b = 1[p >= 0.5] rule.
        "sample" draws b from the learned Bernoulli probabilities. This is the
        policy-faithful mode when update probabilities encode action frequency.
        "constant" samples a state-independent Bernoulli mask using fixed
        per-subsystem rates. "periodic" uses the same rates in a deterministic
        fractional-period scheduler. "always" updates every action dimension.

    proposal_deterministic
        Keep the categorical glazing and continuous control proposals
        deterministic independently of selector sampling.

    selector_seed
        Seed for reproducible Bernoulli selector samples. reset() restores this
        seed, so repeated evaluations are exactly reproducible.

    constant_update_rates
        Per-subsystem target rates used by "constant" and "periodic". Required
        keys are glazing, lighting, heating, and cooling. Defaults are the
        observed E100 rates and should be replaced by validation-set means for
        a final rate-matched experiment.

    periodic_phase_seed
        Seed used only to assign deterministic initial phase offsets to the 19
        periodic schedules. Once initialized, periodic execution is entirely
        deterministic. Defaults to selector_seed.

    Returned dict:
        glazing / lighting_power / heating_sp / cooling_sp   -> actuate these
        selection_mask            policy mask b
        executed_selection_mask   mask consistent with the executed action, so
                                  (a_exec - a_prev) * (1 - b_exec) == 0 holds
        actual_change_mask         physical command changed beyond tolerance
        proposal_action_norm       deterministic/stochastic actor proposal
        prev_action_norm_before    action remembered before this decision
        policy_executed_action_norm  before any safeguard
        executed_action_norm         after safeguards (what was actuated)
        selection_logits / selection_probability  active selector diagnostics
        learned_selection_logits / learned_selection_probability
                                  learned-network diagnostics in every mode
        override_dims, deadband_applied, lighting_safety_* , occupied
    """
    selector_mode = str(selector_mode).lower()
    if selector_mode not in {"threshold", "sample", "constant", "periodic", "always"}:
        raise ValueError(
            "selector_mode must be one of 'threshold', 'sample', 'constant', "
            "'periodic', or 'always', "
            f"got {selector_mode!r}."
        )
    selector_seed = int(selector_seed)
    periodic_phase_seed = (
        selector_seed if periodic_phase_seed is None else int(periodic_phase_seed)
    )
    lighting_change_tolerance_w = float(lighting_change_tolerance_w)
    setpoint_change_tolerance_c = float(setpoint_change_tolerance_c)
    if lighting_change_tolerance_w < 0.0 or setpoint_change_tolerance_c < 0.0:
        raise ValueError("Action-change tolerances must be non-negative.")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]

    obs_dim = cfg["obs_dim"]
    seq_len = cfg["seq_len"]
    latent_dim = cfg["latent_dim"]
    gru_layers = cfg.get("gru_layers", 1)
    rep_dim = cfg["rep_dim"]
    hidden_dim = cfg["hidden_dim"]
    action_dim = cfg.get("action_dim", ACTION_DIM)
    mask_dim = cfg.get("mask_dim", MASK_DIM)
    proposal_input_dim = cfg.get("proposal_input_dim", rep_dim + action_dim)
    selector_input_dim = cfg.get("selector_input_dim", rep_dim + action_dim)
    deadband_min = cfg.get("deadband_min", DEADBAND_MIN)

    if constant_update_rates is None:
        constant_update_rates = {
            "glazing": 0.02777,
            "lighting": 0.12440,
            "heating": 0.26803,
            "cooling": 0.26908,
        }
    required_rate_keys = {"glazing", "lighting", "heating", "cooling"}
    missing_rate_keys = required_rate_keys.difference(constant_update_rates)
    if missing_rate_keys:
        raise ValueError(
            "constant_update_rates is missing keys: "
            + ", ".join(sorted(missing_rate_keys))
        )
    rates = {k: float(constant_update_rates[k]) for k in required_rate_keys}
    if any((value < 0.0 or value > 1.0) for value in rates.values()):
        raise ValueError("Every constant update rate must lie in [0, 1].")
    fixed_rate_np = np.asarray(
        [rates["glazing"]] * 4
        + [rates["lighting"]] * 5
        + [rates["heating"]] * 5
        + [rates["cooling"]] * 5,
        dtype=np.float32,
    )
    if fixed_rate_np.shape != (mask_dim,):
        raise ValueError(
            f"Expected {mask_dim} fixed selector rates, got {fixed_rate_np.shape}."
        )
    fixed_rate_vector = torch.from_numpy(fixed_rate_np).unsqueeze(0).to(DEVICE)

    # Setpoint decode ranges: prefer the checkpoint's, fall back to module
    # globals for older checkpoints that did not record them.
    heat_range = tuple(cfg.get("heat_range", list(HEAT_RANGE)))
    cool_range = tuple(cfg.get("cool_range", list(COOL_RANGE)))
    if heat_range != tuple(HEAT_RANGE) or cool_range != tuple(COOL_RANGE):
        print(
            f"[load_and_act] WARNING: setpoint ranges differ from module globals.\n"
            f"               ckpt heat={heat_range} cool={cool_range}\n"
            f"               code heat={tuple(HEAT_RANGE)} cool={tuple(COOL_RANGE)}\n"
            f"               Using the CHECKPOINT ranges (decode must match the dataset)."
        )

    # Per-zone lighting decode ranges: prefer the checkpoint's saved ranges so
    # decode ALWAYS matches the per-zone normalization used to encode the data.
    light_ranges = np.array(
        cfg.get("light_ranges_per_zone", LIGHT_RANGES_PER_ZONE.tolist()),
        dtype=np.float32,
    )

    gru_encoder = GRUEncoder(obs_dim, latent_dim, gru_layers).to(DEVICE)
    gru_encoder.load_state_dict(ckpt["gru_encoder"])
    gru_encoder.eval()

    selector = SDARSelector(selector_input_dim, mask_dim, hidden_dim).to(DEVICE)
    selector.load_state_dict(ckpt["selector"])
    selector.eval()

    actor = SDARFlatHybridActor(proposal_input_dim, hidden_dim).to(DEVICE)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_normalizer = Normalizer()
    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    # Raw observations arrive as CPU tensors; keep the stats on CPU too.
    obs_normalizer.mean = obs_normalizer.mean.detach().cpu().float()
    obs_normalizer.std = obs_normalizer.std.detach().cpu().float()

    print(
        f"[load_and_act] ckpt={checkpoint_path}\n"
        f"               run={cfg.get('run_name')} epoch={ckpt.get('epoch')} "
        f"dataset={cfg.get('dataset_path')}\n"
        f"               seq_len={seq_len} latent={latent_dim} layers={gru_layers} "
        f"deadband_min={deadband_min}\n"
        f"               enforce_unoccupied_lights_off={enforce_unoccupied_lights_off} "
        f"sync_prev_with_deadband={sync_prev_with_deadband}\n"
        f"               selector_mode={selector_mode} "
        f"selector_seed={selector_seed} "
        f"periodic_phase_seed={periodic_phase_seed} "
        f"proposal_deterministic={proposal_deterministic}\n"
        f"               fixed_rates={rates}"
    )

    zero_obs = torch.zeros(obs_dim)
    obs_window = deque([zero_obs.clone() for _ in range(seq_len)], maxlen=seq_len)

    def _init_prev():
        p = torch.zeros(action_dim, dtype=torch.float32)
        p[:4] = -1.0      # glazing tier 0 normalized
        p[4:9] = -1.0     # zero lighting normalized (0 W -> -1 in each zone's range)
        p[9:14] = -1.0    # heat lower bound normalized
        p[14:19] = 1.0    # cool upper bound normalized
        return p

    prev_action_norm = _init_prev()
    selector_generator = torch.Generator(device=DEVICE)
    selector_generator.manual_seed(selector_seed)

    # Per-dimension offsets prevent every zone in a subsystem from updating at
    # exactly the same periodic timestep. The phase is random only at setup;
    # the resulting schedule is deterministic and resettable.
    phase_rng = np.random.default_rng(periodic_phase_seed)
    initial_periodic_phase = torch.from_numpy(
        phase_rng.uniform(0.0, 1.0, size=(1, mask_dim)).astype(np.float32)
    ).to(DEVICE)
    periodic_accumulator = initial_periodic_phase.clone()

    def reset():
        nonlocal obs_window, prev_action_norm, periodic_accumulator
        obs_window = deque([zero_obs.clone() for _ in range(seq_len)], maxlen=seq_len)
        prev_action_norm = _init_prev()
        selector_generator.manual_seed(selector_seed)
        periodic_accumulator = initial_periodic_phase.clone()

    def _cont_norm_to_physical(cont_norm_np: np.ndarray):
        def denorm_range(x, lo, hi):
            return 0.5 * (x + 1.0) * (hi - lo) + lo

        # lighting: PER-ZONE ranges (each zone has its own installed max)
        light_norm = cont_norm_np[:5]
        light_power = np.empty(5, dtype=np.float32)
        for z in range(5):
            lo, hi = light_ranges[z]
            light_power[z] = np.clip(denorm_range(light_norm[z], lo, hi), lo, hi)

        heat_sp = np.clip(denorm_range(cont_norm_np[5:10], *heat_range), heat_range[0], heat_range[1])
        cool_sp = np.clip(denorm_range(cont_norm_np[10:15], *cool_range), cool_range[0], cool_range[1])

        deadband_applied = np.zeros(5, dtype=bool)
        for i in range(5):
            if cool_sp[i] < heat_sp[i] + deadband_min - 1e-9:
                cool_sp[i] = np.clip(heat_sp[i] + deadband_min, cool_range[0], cool_range[1])
                deadband_applied[i] = True
        return light_power, heat_sp, cool_sp, deadband_applied

    def _cool_physical_to_norm(cool_sp: np.ndarray) -> np.ndarray:
        lo, hi = cool_range
        return np.clip(2.0 * (cool_sp - lo) / (hi - lo) - 1.0, -1.0, 1.0).astype(np.float32)

    def get_action(
        raw_obs: np.ndarray,
        deterministic: bool | None = None,
        force_first_update: bool = False,
    ):
        nonlocal prev_action_norm, periodic_accumulator

        raw_obs = np.asarray(raw_obs, dtype=np.float32)
        if raw_obs.shape != (obs_dim,):
            raise ValueError(f"Expected observation shape ({obs_dim},), got {raw_obs.shape}")

        occupied = bool(raw_obs[OCCUPANCY_OBS_INDEX] >= 0.5)

        obs_t = torch.from_numpy(raw_obs).float()
        obs_norm = obs_normalizer.normalize(obs_t.unsqueeze(0)).squeeze(0)
        obs_window.append(obs_norm)

        window_tensor = torch.stack(list(obs_window)).unsqueeze(0).to(DEVICE)
        prev_action_batch = prev_action_norm.unsqueeze(0).to(DEVICE)
        prev_action_before = prev_action_batch.clone()

        with torch.no_grad():
            z, _ = gru_encoder(window_tensor)
            rep = torch.cat([z, window_tensor[:, -1, :]], dim=-1)

            learned_selection_logits = selector(rep, prev_action_batch)
            learned_selection_probability = torch.sigmoid(learned_selection_logits)

            # Passing deterministic explicitly preserves the old coupled API.
            # Leaving it as None uses the independently configured selector
            # and proposal modes.
            if deterministic is None:
                active_selector_mode = selector_mode
                active_proposal_deterministic = proposal_deterministic
            else:
                # Preserve the legacy threshold/sample coupling only for the
                # learned selector. Explicit ablation modes must not be
                # silently replaced by this backwards-compatible argument.
                active_selector_mode = selector_mode
                if selector_mode in {"threshold", "sample"}:
                    active_selector_mode = "threshold" if deterministic else "sample"
                active_proposal_deterministic = bool(deterministic)

            if active_selector_mode in {"threshold", "sample"}:
                active_selection_logits = learned_selection_logits
                active_selection_probability = learned_selection_probability
            elif active_selector_mode in {"constant", "periodic"}:
                active_selection_probability = fixed_rate_vector
                active_selection_logits = torch.logit(
                    fixed_rate_vector.clamp(1e-6, 1.0 - 1e-6)
                )
            elif active_selector_mode == "always":
                active_selection_probability = torch.ones(
                    1, mask_dim, device=DEVICE
                )
                active_selection_logits = torch.logit(
                    active_selection_probability.clamp(1e-6, 1.0 - 1e-6)
                )
            else:
                raise RuntimeError(f"Unhandled selector mode {active_selector_mode!r}.")

            if force_first_update or active_selector_mode == "always":
                mask = torch.ones(1, mask_dim, device=DEVICE)
            elif active_selector_mode == "threshold":
                mask = (active_selection_probability >= 0.5).float()
            elif active_selector_mode in {"sample", "constant"}:
                mask = torch.bernoulli(
                    active_selection_probability,
                    generator=selector_generator,
                )
            elif active_selector_mode == "periodic":
                periodic_accumulator = periodic_accumulator + fixed_rate_vector
                mask = (periodic_accumulator >= 1.0).float()
                periodic_accumulator = periodic_accumulator - mask

            action_mix = (1.0 - mask) * prev_action_batch + mask * ACTION_MASK_VALUE
            proposal_rep = torch.cat([rep, action_mix], dim=-1)
            proposal_actor_fmt = actor.sample(
                proposal_rep,
                deterministic=active_proposal_deterministic,
            )

            prop_glz_ids = proposal_actor_fmt[:, :NUM_GLAZING_HEADS].long().clamp(0, NUM_GLAZING_CLASSES - 1)
            prop_glz_norm = ids_to_normalized_glazing(prop_glz_ids)
            prop_cont_norm = proposal_actor_fmt[:, NUM_GLAZING_HEADS:].clamp(-1.0, 1.0)
            proposal_norm = torch.cat([prop_glz_norm, prop_cont_norm], dim=-1)

            # Normal SDAR execution: b=0 -> repeat previous, b=1 -> proposal
            executed_norm = (1.0 - mask) * prev_action_batch + mask * proposal_norm

            # Untouched policy output, kept for attribution/diagnostics.
            policy_executed_norm = executed_norm.clone()

            # ---------- HARD CONSTRAINT: lights off when unoccupied ----------
            lighting_safety_active = bool(enforce_unoccupied_lights_off and not occupied)
            override_dims = torch.zeros(1, action_dim, device=DEVICE)

            if lighting_safety_active:
                light_slice = executed_norm[:, LIGHTING_ACTION_SLICE]
                changed = (light_slice > LIGHTS_OFF_NORM + 1e-6).float()
                override_dims[:, LIGHTING_ACTION_SLICE] = changed
                executed_norm[:, LIGHTING_ACTION_SLICE] = LIGHTS_OFF_NORM

            lighting_safety_override = bool(override_dims.sum().item() > 0)

            # Mask describing what ACTUALLY happened: a dim the selector wanted
            # to repeat but the safeguard changed is, in SDAR terms, an "act".
            executed_mask = torch.clamp(mask + override_dims, max=1.0)

            # CRITICAL: remember the SAFEGUARDED action so the next repeat
            # repeats lights-off instead of a stale nonzero value.
            prev_action_norm = executed_norm.squeeze(0).detach().cpu()

        executed_np = prev_action_norm.numpy().copy()
        prev_action_before_np = prev_action_before.squeeze(0).detach().cpu().numpy().copy()
        proposal_np = proposal_norm.squeeze(0).detach().cpu().numpy().copy()
        policy_executed_np = policy_executed_norm.squeeze(0).detach().cpu().numpy()
        mask_np = mask.squeeze(0).detach().cpu().numpy()
        selection_logits_np = active_selection_logits.squeeze(0).detach().cpu().numpy()
        selection_probability_np = active_selection_probability.squeeze(0).detach().cpu().numpy()
        learned_selection_logits_np = learned_selection_logits.squeeze(0).detach().cpu().numpy()
        learned_selection_probability_np = learned_selection_probability.squeeze(0).detach().cpu().numpy()
        executed_mask_np = executed_mask.squeeze(0).detach().cpu().numpy()
        override_dims_np = override_dims.squeeze(0).detach().cpu().numpy()

        glz_ids = normalized_glazing_to_ids(executed_np[:4])
        glazing_names = [AVAILABLE_GLAZING_STATES[g] for g in glz_ids]

        light_power, heat_sp, cool_sp, deadband_applied = _cont_norm_to_physical(executed_np[4:])

        # ---------- keep internal state == actuated state ----------
        if sync_prev_with_deadband and deadband_applied.any():
            executed_np[COOLING_ACTION_SLICE] = _cool_physical_to_norm(cool_sp)
            prev_action_norm = torch.from_numpy(executed_np.copy())
            db = deadband_applied.astype(np.float32)
            executed_mask_np[COOLING_ACTION_SLICE] = np.clip(
                executed_mask_np[COOLING_ACTION_SLICE] + db, 0.0, 1.0
            )
            override_dims_np[COOLING_ACTION_SLICE] = np.maximum(
                override_dims_np[COOLING_ACTION_SLICE], db
            )

        # A selected update is not necessarily a physical command change. Use
        # discrete equality for glazing and engineering-unit tolerances for
        # continuous actuators to avoid counting floating-point noise.
        previous_glazing_ids = normalized_glazing_to_ids(prev_action_before_np[:4])
        actual_change_mask = np.zeros(action_dim, dtype=np.float32)
        actual_change_mask[:4] = (previous_glazing_ids != glz_ids).astype(np.float32)

        previous_light, previous_heat, previous_cool, _ = _cont_norm_to_physical(
            prev_action_before_np[4:]
        )
        actual_change_mask[4:9] = (
            np.abs(light_power - previous_light) > lighting_change_tolerance_w
        ).astype(np.float32)
        actual_change_mask[9:14] = (
            np.abs(heat_sp - previous_heat) > setpoint_change_tolerance_c
        ).astype(np.float32)
        actual_change_mask[14:19] = (
            np.abs(cool_sp - previous_cool) > setpoint_change_tolerance_c
        ).astype(np.float32)

        return {
            # physical actions to actuate
            "glazing": glazing_names,
            "lighting_power": light_power,
            "heating_sp": heat_sp,
            "cooling_sp": cool_sp,

            # policy view
            "selection_mask": mask_np,
            "selection_logits": selection_logits_np,
            "selection_probability": selection_probability_np,
            "learned_selection_logits": learned_selection_logits_np,
            "learned_selection_probability": learned_selection_probability_np,
            "selector_mode": active_selector_mode,
            "selector_stochastic": active_selector_mode in {"sample", "constant"},
            "proposal_deterministic": active_proposal_deterministic,
            "force_first_update": bool(force_first_update),
            "target_update_rate": selection_probability_np.copy(),
            "proposal_action_norm": proposal_np,
            "prev_action_norm_before": prev_action_before_np,
            "policy_executed_action_norm": policy_executed_np,

            # what actually happened
            "executed_action_norm": executed_np.copy(),
            "executed_selection_mask": executed_mask_np,
            "actual_change_mask": actual_change_mask,

            # safeguard diagnostics
            "lighting_safety_active": lighting_safety_active,
            "lighting_safety_override": lighting_safety_override,
            "override_dims": override_dims_np,
            "deadband_applied": deadband_applied.astype(np.float32),
            "occupied": occupied,
        }

    return get_action, reset


if __name__ == "__main__":
    train()
