# from https://github.com/NVIDIA/warp/blob/release-1.3/warp/examples/optim/example_bounce.py

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Grad Bounce
#
# Shows how to use Warp to optimize the initial velocity of a particle
# such that it bounces off the wall and floor in order to hit a target.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import numpy as np
import torch

import warp as wp

from rewarped.environment import IntegratorType, run_env
from rewarped.warp_env import WarpEnv
from rewarped.utils.plotter import Plotter


class Bounce(WarpEnv):
    sim_name = "Bounce" + "WarpExamples"
    env_offset = (0.0, 0.0, 2.5)

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 8
    # euler_settings = dict(angular_damping=0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 8
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=1)

    eval_fk = True
    eval_ik = False

    frame_dt = 1.0 / 60.0
    episode_duration = 1.0
    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "joint_qd")

    def __init__(self, num_envs=8, episode_length=-1, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 0

        episode_length = int(self.episode_duration / self.frame_dt)
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = 1.0
        self.render_traj = True

        self.plotter = Plotter()

    def create_env(self, builder):
        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2
        restitution = 1.0

        box_ke = 1.0e4
        box_kf = 0.0
        box_kd = 1.0e1
        box_mu = 0.2

        # ground_ke = 1e7
        ground_ke = ke
        ground_kf = kf
        ground_kd = kd
        ground_mu = mu
        ground_restitution = 1.0

        # Dummy shape due to rendering bug
        builder.add_shape_box(body=-1, pos=wp.vec3(400.0, 1.0, 0.0), hx=0.25, hy=1.0, hz=1.0)

        builder.add_shape_box(body=-1, pos=wp.vec3(2.0, 1.0, 0.0), hx=0.25, hy=1.0, hz=1.0, ke=box_ke, kf=box_kf, kd=box_kd, mu=box_mu)

        self.ground_plane_settings = dict(ke=ground_ke, kd=ground_kd, kf=ground_kf, mu=ground_mu, restitution=ground_restitution)

        sphere_mass = 1.0
        sphere_volume = 4.0 / 3.0 * np.pi * 0.1 ** 3
        sphere_density = sphere_mass / sphere_volume
        sphere = builder.add_body(origin=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), m=1.0)
        builder.add_shape_sphere(body=sphere, radius=0.1, density=sphere_density, has_ground_collision=True,
                                 ke=ke, kf=kf, kd=kd, mu=mu, restitution=restitution,
                                 is_solid=True)

        # Add free joint to sphere
        # builder.add_joint_free(child=sphere, armature=0.01)
        p = wp.vec3(-0.5, 1.0, 0.0)
        q = wp.quat_identity()
        object_start_pose = wp.transform(p, q)
        builder.add_joint_prismatic(
            parent=-1,
            child=sphere,
            armature=0.05,
            name="joint_prismatic",
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform((-0.5, 1.0, 0.0), wp.quat_identity()),
        )

        # Initial velocity
        builder.joint_q[0:1] = object_start_pose.p
        builder.joint_qd[0:3] = [5.0]
        builder.body_q[0] = object_start_pose
        builder.body_qd = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0]
        # builder.joint_qd[3:6] = [5.0, -5.0, 0.0]
        # builder.joint_q[0:3] = [-0.5, 1.0, 0.0]
        # builder.body_qd[0] = (0.0, 0.0, 0.0, 5.0, -5.0, 0.0)

    def create_model(self):
        model = super().create_model()

        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2

        model.soft_contact_ke = ke
        model.soft_contact_kf = kf
        model.soft_contact_kd = kd
        model.soft_contact_mu = mu
        model.soft_contact_margin = 10.0
        model.soft_contact_restitution = 1.0

        return model

    def init_sim(self):
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.target = (-2.0, 1.5, 0.0)
            self.target = torch.tensor(self.target, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.early_termination:
            raise NotImplementedError
        else:
            self.traj_verts = []

    @torch.no_grad()
    def randomize_init(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        pass

    def compute_observations(self):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1, 1)
        joint_qd = self.state.joint_qd.clone().view(self.num_envs, -1, 1)
        # joint_q[:, :, :3] -= self.env_offsets.view(self.num_envs, 1, 3)
        # delta = joint_q[:, 0, :3] - self.target
        self.obs_buf = {
            "joint_q": joint_q,
            "joint_qd": joint_qd,
            # "delta": delta,
        }

    def compute_reward(self):
        # delta = self.obs_buf["delta"]
        loss = self.obs_buf["joint_qd"][:, 0, 0:1] # velocity in x direction
        # print(loss)
        loss = torch.einsum("bi,bi->b", loss, loss)  # dot product
        rew = -1.0 * loss

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            raise NotImplementedError
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated

    def render(self, state=None):
        if not self.render_traj:
            super().render(state)
        else:
            # render state 1 (swapped with state 0 just before)
            state = state or self.state_1
            # traj_vert = self.state.joint_q.view(self.num_envs, -1, 1)[:, 0, :3].tolist()
            # self.traj_verts.append(traj_vert)

            if self.renderer is not None:
                with wp.ScopedTimer("render", False):
                    self.render_time += self.frame_dt
                    self.renderer.begin_frame(self.render_time)
                    self.renderer.render(state)

                    # traj_verts = np.array(self.traj_verts).transpose(1, 0, 2)  # -> B, T, 3
                    for i in range(self.num_envs):
                        if self.progress_buf[i] == 0:
                            continue

                        pos = (self.target + self.env_offsets[i]).tolist()
                        self.renderer.render_box(
                            pos=pos,
                            rot=wp.quat_identity(),
                            extents=(0.1, 0.1, 0.1),
                            color=(0.0, 0.0, 0.0),
                            name=f"target{i}",
                        )

                        # if traj_verts.shape[1] > 1:
                        #     self.renderer.render_line_strip(
                        #         vertices=traj_verts[i],
                        #         color=wp.render.bourke_color_map(0.0, getattr(self, "max_iter", 250.0), self.iter),
                        #         radius=0.02,
                        #         name=f"iter{self.iter}_traj{i}",
                        #     )

                self.renderer.end_frame()

    def check_grad(self):
        raise NotImplementedError

    def run_cfg(self):
        assert self.requires_grad

        state_0 = self.state_0
        body_q_0, body_qd_0 = self.state_tensors

        train_iters = 1
        train_rate = 0.02
        opt = torch.optim.SGD([body_qd_0], lr=train_rate)
        opt.add_param_group({"params": [body_q_0], "lr": 0})  # frozen params, adding so grads are zeroed

        self._run_state = state_0
        self._run_state_tensors = (body_q_0, body_qd_0)

        policy = None

        return train_iters, opt, policy

    def run_reset(self):
        obs = self.reset(clear_grad=self.requires_grad)

        state_0 = self._run_state
        body_q_0, body_qd_0 = self._run_state_tensors

        self.state_0 = state_0
        self.state_tensors = [body_q_0, body_qd_0]
        self.render_traj = self.iter % 16 == 0

        return obs

    def run_loss(self, traj, opt, policy):
        obses, actions, rewards, dones, infos = traj

        loss = -rewards[-1]

        opt.zero_grad()
        loss.sum().backward()
        opt.step()

        body_q_0, body_qd_0 = self._run_state_tensors

        print(f"Iter: {self.iter} Loss: {loss.tolist()}")
        # print(f"Grad body q, qd: {body_q_0.grad.norm().item()}, {body_qd_0.grad.norm().item()}")
        print(f"Grad qd: {body_qd_0.grad[0].item()}")

        # Store loss and gradient for plotting
        self.plotter.add_loss(loss.tolist()[0])
        self.plotter.add_gradient(body_qd_0.grad.detach().cpu().numpy().flatten()[3:5])
        self.plotter.add_initial_velocity(body_qd_0.detach().cpu().numpy().flatten()[3:5])


if __name__ == "__main__":
    run_env(Bounce, no_grad=False)
