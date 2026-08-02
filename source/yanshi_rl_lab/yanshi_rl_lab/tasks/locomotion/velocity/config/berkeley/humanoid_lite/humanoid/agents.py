# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Berkeley Humanoid Lite PPO runner configuration (attempt 3).

Verbatim transcription of the vendor's humanoid runner (HybridRobotics/
Berkeley-Humanoid-Lite @ 984741a3,
``.../tasks/locomotion/velocity/config/humanoid/agents/rsl_rl_ppo_cfg.py``
L6-31), with two documented deviations:

- ``max_iterations`` 6000 -> 10000: the fleet trains every robot for 10000
  iterations so Yanshi Rank curves stay comparable (the training command and
  seeds are fleet-uniform); the official value is recorded here, not used.
- ``experiment_name`` "humanoid" -> "": fleet convention (the runner cfg
  leaves it to the task name, same as ``YanshiPPORunnerCfg``).
- ``actor_obs_normalization`` / ``critic_obs_normalization`` = False are
  required fields on our pinned IsaacLab 2.3.2 / rsl-rl-lib >= 4.0.0 (see
  ``tasks/locomotion/agents.py``); the vendor's older stack predates them.

Everything else (24 steps/env, entropy 0.008, lr 1e-3 adaptive with
desired_kl 0.01, init_noise_std 1.0, hidden [256, 128, 128], elu,
empirical_normalization False, gamma 0.99, lam 0.95, clip 0.2, 5 epochs,
4 mini-batches, grad-norm 1.0) is the official value; most of them coincide
with the fleet ``YanshiPPORunnerCfg`` defaults and are restated so this
class stands alone as the official transcription.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Official hyperparameters (rsl_rl_ppo_cfg.py @984741a3, line refs in docstring)
OFFICIAL_ENTROPY_COEF = 0.008  # L22
OFFICIAL_HIDDEN_DIMS = [256, 128, 128]  # L14-15
FLEET_MAX_ITERATIONS = 10000  # deviation: official 6000 (L8), fleet-uniform budget


@configclass
class BerkeleyHumanoidLitePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # official L7
    max_iterations = FLEET_MAX_ITERATIONS
    save_interval = 100  # official L9
    experiment_name = ""  # fleet convention; official "humanoid" (L10)
    empirical_normalization = False  # official L11
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # official L13
        actor_obs_normalization=False,  # required on IsaacLab 2.3.2 / rsl-rl-lib >= 4.0.0
        critic_obs_normalization=False,
        actor_hidden_dims=OFFICIAL_HIDDEN_DIMS,  # official L14
        critic_hidden_dims=OFFICIAL_HIDDEN_DIMS,  # official L15
        activation="elu",  # official L16
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # official L19
        use_clipped_value_loss=True,  # official L20
        clip_param=0.2,  # official L21
        entropy_coef=OFFICIAL_ENTROPY_COEF,  # official L22
        num_learning_epochs=5,  # official L23
        num_mini_batches=4,  # official L24
        learning_rate=1.0e-3,  # official L25
        schedule="adaptive",  # official L26
        gamma=0.99,  # official L27
        lam=0.95,  # official L28
        desired_kl=0.01,  # official L29
        max_grad_norm=1.0,  # official L30
    )
