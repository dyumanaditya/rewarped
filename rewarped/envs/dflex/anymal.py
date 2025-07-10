import os.path

import torch
import math
import collections

import warp as wp
import warp.sim
from torch.onnx.symbolic_opset9 import detach

from ...environment import IntegratorType, run_env, RenderMode
from ...warp_env import WarpEnv

from .utils.torch_utils import normalize, quat_conjugate, quat_from_angle_axis, quat_mul, quat_rotate


class Anymal(WarpEnv):
    sim_name = "Anymal" + "Rewarped"
    env_offset = (5.0, 0.0, 5.0)

    render_mode = RenderMode.OPENGL

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 16
    # euler_settings = dict(angular_damping=0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 16
    featherstone_settings = dict(angular_damping=0.05, update_mass_matrix_every=sim_substeps_featherstone)

    eval_fk = True
    eval_ik = False if integrator_type == IntegratorType.FEATHERSTONE else True

    frame_dt = 1.0 / 60.0
    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "joint_qd")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=4, episode_length=1000, early_termination=True, **kwargs):
        # TODO: Check if this holds for go2
        num_obs = 49
        num_act = 12
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        # MDP Parameters
        self.action_scale = 50.0
        self.joint_vel_obs_scaling = 0.1

        # Reward parameters
        self.termination_height = 0.25
        self.termination_height_fly = 1.0
        self.action_penalty = -0.005
        self.up_rew_scale = 0.1
        self.heading_rew_scale = 1.0
        self.height_rew_scale = 1.0

        # how many past obs to keep
        self.obs_history_len = 10
        # a deque per env, each entry is a list of floats
        self.obs_history = [
            collections.deque(maxlen=self.obs_history_len)
            for _ in range(self.num_envs)
        ]

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.05
        return builder

    def _obs_dim(self):
        # replace 49 by however many dims your final obs has
        return 49

    # def create_env(self, builder):
    #     self.create_articulation(builder)

    def create_articulation(self, builder):
        # TODO: Change this to correct file

        wp.sim.parse_urdf(
            os.path.join(self.asset_dir, "dflex/anymal_c/urdf/anymal_c.urdf"),
            builder,
            floating=True,
            density=1000.0,
            stiffness=0.0,
            damping=1.0,
            contact_ke=4.0e3,
            contact_kd=1.0e3,
            contact_kf=3.0e2,
            contact_mu=0.75,
            contact_restitution=0.0,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            armature=0.01,
            # armature_scale=5,
            enable_self_collisions=True,
            collapse_fixed_joints=True,
        )
        # wp.sim.parse_mjcf(
        #     os.path.join(self.asset_dir, "dflex/go2.xml"),
        #     builder,
        #     density=1000.0,
        #     stiffness=85.0,
        #     damping=2.0,
        #     contact_ke=2.0e3,
        #     contact_kd=5.0e2,
        #     contact_kf=1.0e2,
        #     contact_mu=0.75,
        #     contact_restitution=0.0,
        #     limit_ke=1.0e3,
        #     limit_kd=1.0e1,
        #     armature=0.006,
        #     # armature_scale=5,
        #     enable_self_collisions=True,
        #     up_axis="y"
        # )

        builder.joint_axis_mode = [wp.sim.JOINT_MODE_FORCE] * len(builder.joint_axis_mode)
        builder.joint_act[:] = [0.0] * len(builder.joint_act)

        self.start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
        self.inv_start_rot = wp.quat_inverse(self.start_rot)

        # TODO: Check with isaac sim to see if these joint positions are correct
        self.start_height = 0.85
        builder.joint_q[:7] = [
            0.0,
            self.start_height,
            0.0,
            *self.start_rot,
        ]

        builder.joint_q[7:] = [
            0.03,  # LF_HAA
            0.4,  # LF_HFE
            -0.8,  # LF_KFE
            -0.03,  # RF_HAA
            0.4,  # RF_HFE
            -0.8,  # RF_KFE
            0.03,  # LH_HAA
            -0.4,  # LH_HFE
            0.8,  # LH_KFE
            -0.03,  # RH_HAA
            -0.4,  # RH_HFE
            0.8,  # RH_KFE
        ]

    def init_sim(self):
        super().init_sim()
        # self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...

            self.start_joint_q = self.state.joint_q.view(self.num_envs, -1).clone()
            self.start_joint_qd = self.state.joint_qd.view(self.num_envs, -1).clone()

            self.start_pos = self.start_joint_q[:, :3]
            self.start_rot = list(wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5))
            self.start_rotation = torch.tensor(self.start_rot, device=self.device)

            # Unit vectors
            self.x_unit = torch.tensor([1.0, 0, 0], device=self.device).repeat(self.num_envs, 1)
            self.y_unit = torch.tensor([0, 1.0, 0], device=self.device).repeat(self.num_envs, 1)
            self.z_unit = torch.tensor([0, 0, 1.0], device=self.device).repeat(self.num_envs, 1)

            # Basis and target
            self.up_vec = self.y_unit.clone()
            self.heading_vec = self.x_unit.clone()
            self.inv_start_rot = quat_conjugate(self.start_rotation).repeat(self.num_envs, 1)
            self.basis_vec0 = self.heading_vec.clone()
            self.basis_vec1 = self.up_vec.clone()
            self.targets = torch.tensor([10000.0, 0.0, 0.0], device=self.device, requires_grad=False).repeat(self.num_envs, 1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    @torch.no_grad()
    def randomize_init(self, env_ids):
        # Stochastic initialization similar to DFlex
        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)

        N = len(env_ids)

        # Randomize base position
        joint_q[env_ids, 0:3] = self.start_pos[env_ids] + 0.1 * (torch.rand(N, 3, device=self.device) - 0.5) * 2.0

        # Random small rotation
        angle = (torch.rand(N, device=self.device) - 0.5) * math.pi / 12.0
        axis = torch.nn.functional.normalize(torch.rand(N, 3, device=self.device) - 0.5, dim=-1)
        joint_q[env_ids, 3:7] = quat_mul(joint_q[env_ids, 3:7], quat_from_angle_axis(angle, axis))

        # Randomize joints
        joint_q[env_ids, 7:] = self.start_joint_q[env_ids, 7:] + 0.2 * (
                    torch.rand(N, 12, device=self.device) - 0.5) * 2.0

        # Velocity randomization
        joint_qd[env_ids, :] = 0.5 * (torch.rand(N, 12+6, device=self.device) - 0.5)

        # Convert COM twist
        ang_vel = joint_qd[env_ids, 0:3]
        lin_vel = joint_qd[env_ids, 3:6]
        joint_qd[env_ids, 3:6] = lin_vel + torch.cross(joint_q[env_ids, 0:3], ang_vel, dim=-1)

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions.clone()
        acts = self.action_scale * actions

        # Invert the action direction to match dflex
        # acts = -acts

        # hard‐limit to ±80 Nm
        max_tau = 80.0
        acts = torch.clamp(acts, -max_tau, max_tau)

        if self.joint_act_indices is ...:
            self.control.assign("joint_act", acts.flatten())
        else:
            joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, acts)
            self.control.assign("joint_act", joint_act.flatten())
        # self.control.assign("joint_act", wp.to_torch(self.model.joint_act).clone().flatten())

    def compute_observations(self):
        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)

        # Torso
        _pos = joint_q[:, 0:3]
        pos = _pos - self.env_offsets
        rot = joint_q[:, 3:7]
        lin = joint_qd[:, 3:6]
        ang = joint_qd[:, 0:3]

        # Convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin = lin - torch.cross(_pos, ang, dim=-1)

        to_target = self.targets + (self.start_pos - self.env_offsets) - pos
        to_target[:, 1] = 0.0
        dirs = normalize(to_target)
        quat_rel = quat_mul(rot, self.inv_start_rot)
        up = quat_rotate(quat_rel, self.basis_vec1)
        head = quat_rotate(quat_rel, self.basis_vec0)

        obs_list = [
            pos[:, 1:2],     # 0 torso height
            rot,             # 1:5 torso rotation
            lin,             # 5:8 torso linear velocity
            ang,             # 8:11 torso angular velocity
            joint_q[:, 7:],  # 11:23 joint positions (legs)
            # joint_qd[:, 6:],  # 23:35 joint velocities (legs)
            self.joint_vel_obs_scaling * joint_qd[:, 6:],
            up[:, 1:2],      # 35 up vector in Y direction
            (head * dirs).sum(dim=-1, keepdim=True),    # 36 heading
            self.actions.clone()                        # 37:49 (12) actions
        ]
        self.obs_buf = torch.cat(obs_list, dim=-1)
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        # print("===========================")
        # print(obs_list)
        # print("===========================")
        # 3) check for any NaNs
        if torch.isnan(self.obs_buf).any():
            mask = torch.isnan(self.obs_buf)
            bad_envs = mask.any(dim=1).nonzero(as_tuple=False).squeeze(1).tolist()
            for env in bad_envs:
                # ** Extra debug info **
                step = int(self.progress_buf[env].item())
                is_trunc = bool(self.truncated_buf[env].item())
                is_term = bool(self.terminated_buf[env].item())
                is_reset = bool(self.reset_buf[env].item())
                raw_q = joint_q[env].detach().cpu().numpy()
                raw_qd = joint_qd[env].detach().cpu().numpy()
                act = self.actions[env].detach().cpu().numpy()
                offset = (self.env_offsets[env].cpu().detach().numpy()
                          if torch.is_tensor(self.env_offsets)
                          else self.env_offsets[env])
                target = self.targets[env].detach().cpu().numpy()

                print(f"\n>>> NaN detected in env #{env}")
                print(f"    step={step}/{self.episode_length}, "
                      f"truncated={is_trunc}, terminated={is_term}, reset={is_reset}")
                print(f"    env_offset={offset}, target={target}")
                print(f"    raw joint_q  = {raw_q}")
                print(f"    raw joint_qd = {raw_qd}")
                print(f"    action       = {act}")

                # which dims are NaN
                dims = mask[env].nonzero(as_tuple=False).squeeze(1).tolist()
                print(f"    NaN dims     = {dims}")

                # last few clean obs
                hist = list(self.obs_history[env])
                print(f"    last {len(hist)} clean obs:")
                for i, o in enumerate(hist, 1):
                    print(f"      t-{len(hist) - i + 1}: {o}")

                # current obs
                curr = self.obs_buf[env].detach().cpu().numpy().tolist()
                print(f"    curr (with NaNs) = {curr}\n")

        # 3) check for any NaNs
        # if torch.isnan(self.obs_buf).any():
        #     mask = torch.isnan(self.obs_buf)
        #     # which envs have *any* NaNs?
        #     bad_envs = mask.any(dim=1).nonzero(as_tuple=False).squeeze(1).tolist()
        #     for env in bad_envs:
        #         # which dims in this env are NaN?
        #         dims = mask[env].nonzero(as_tuple=False).squeeze(1).tolist()
        #         print(f"\n>>> NaN detected in env #{env}, dims {dims}")
        #
        #         # print the stored trajectory of the last obs_history_len steps
        #         print(f"Last {len(self.obs_history[env])} clean observations for env {env}:")
        #         for step, hist_obs in enumerate(self.obs_history[env], start=1):
        #             print(f"  t-{len(self.obs_history[env]) - step + 1}: {hist_obs}")
        #
        #         # finally, show the current obs with NaN
        #         curr = self.obs_buf[env].cpu().numpy().tolist()
        #         print(f"Current (contains NaN): {curr}\n")

    def compute_reward(self):
        up_r = self.up_rew_scale * self.obs_buf[:, 35]
        head_r = self.heading_rew_scale * self.obs_buf[:, 36]
        height_r = self.height_rew_scale * (self.obs_buf[:, 0] - self.termination_height)
        prog_r = self.obs_buf[:, 5]
        act_pen = self.action_penalty * torch.sum(self.actions ** 2, dim=-1)
        rew = prog_r + up_r + head_r + height_r + act_pen

        # 2) detect NaNs or Infs in obs, joint_q, joint_qd
        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)

        nonfinite_mask = ~torch.isfinite(self.obs_buf).all(dim=-1)
        nonfinite_mask |= ~torch.isfinite(joint_q).all(dim=-1)
        nonfinite_mask |= ~torch.isfinite(joint_qd).all(dim=-1)

        invalid_mask = (torch.abs(joint_q) > 1e6).any(dim=-1)
        invalid_mask |= (torch.abs(joint_qd) > 1e6).any(dim=-1)

        nan_terminated = nonfinite_mask | invalid_mask

        # Termination and reset logic
        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            terminated = (self.obs_buf[:, 0] < self.termination_height) | (self.obs_buf[:, 0] > self.termination_height_fly)
            reset = torch.where(terminated, torch.ones_like(reset), reset)
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)

        # 4) incorporate NaN/∞ termination into both reset & terminated
        reset = reset | nan_terminated
        terminated = terminated | nan_terminated

        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated

        # reset = self.reset_buf.clone()
        # truncated = self.progress_buf > self.episode_length - 1
        # reset = torch.where(truncated, torch.ones_like(reset), reset)
        # if self.early_termination:
        #     term = self.obs_buf[:, 0] < self.termination_height
        #     reset = torch.where(term, torch.ones_like(reset), reset)
        # else:
        #     term = torch.zeros_like(reset)
        # self.rew_buf = rew
        # self.reset_buf = reset
        # self.terminated_buf = term
        # self.truncated_buf = truncated
        #
        #
        #
        # rew = None
        # raise NotImplementedError
        #
        # reset_buf, progress_buf = self.reset_buf, self.progress_buf
        # max_episode_steps, early_termination = self.episode_length, self.early_termination
        # truncated = progress_buf > max_episode_steps - 1
        # reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        # if early_termination:
        #     raise NotImplementedError
        # else:
        #     terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        # self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated


if __name__ == "__main__":
    run_env(Anymal)
