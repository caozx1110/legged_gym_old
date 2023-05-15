# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgAlg

class A1RoughCfg( LeggedRobotCfg ):
    class terrain( LeggedRobotCfg.terrain ):
        # TODO: 在这里修改地形
        mesh_type = 'plane'
        
        '''所有地形可视化'''
        # num_rows = 10
        # num_cols = 8
        # terrain_proportions = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        
        '''上金字塔楼梯 + 课程学习'''
        # num_rows = 10
        # num_cols = 40
        # terrain_proportions = [0, 0, 1, 0, 0, 0, 0, 0]
        
        '''不平坦地形 + 课程学习'''
        # num_rows = 10
        # num_cols = 40
        # terrain_proportions = [0, 1, 0, 0, 0, 0, 0, 0]
        
        '''阶梯障碍 + 课程学习'''
        # num_rows = 10
        # num_cols = 40
        # terrain_proportions = [0, 0, 0, 0, 1, 0, 0, 0]
        
        '''湿滑地面'''
        # static_friction = 0.45
        # dynamic_friction = 0.05
        # terrain_kwargs = {'type': 'sloped_terrain', 'slope': 0}
        
        '''选择特定地形'''
        # curriculum = False
        # selected = True
        # terrain_kwargs = {'type': 'random_uniform_terrain', 'min_height': 0.0, 'max_height': 0.05, 'step': 0.005}
        # terrain_kwargs = {'type': 'sloped_terrain', 'slope': 0.1}
        # terrain_kwargs = {'type': 'pyramid_sloped_terrain', 'slope': 0.2, 'platform_size': 4}
        # terrain_kwargs = {'type': 'discrete_obstacles_terrain', 'max_height': 1, 'min_size': 0.2, 'max_size': 1, 'num_rects': 1, 'platform_size': 2}
        # terrain_kwargs = {'type': 'wave_terrain', 'num_waves': 1, 'amplitude': 0.5}
        # terrain_kwargs = {'type': 'stairs_terrain', 'step_height': 0.1, 'step_width': 0.5}
        # terrain_kwargs = {'type': 'pyramid_stairs_terrain', 'step_width': 0.5, 'step_height': 0.1, 'platform_size': 2}
        # terrain_kwargs = {'type': 'stepping_stones_terrain', 'stone_size': 0.5, 'stone_distance': 0.1, 'max_height': 0.1, 'platform_size': 2, 'depth': -10}
        
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100. # forces above this value are penalized
        
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -50.0
            
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            # tracking_lin_vel = 5.0
            # tracking_ang_vel = 2.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            dof_vel = -0.001
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.01


class A1RoughCfgPPO( LeggedRobotCfgAlg ):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
        
    class algorithm:
        # training params
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        gamma = 0.99
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        schedule = 'adaptive' # could be adaptive, fixed
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
    class runner:
        run_name = ''
        experiment_name = 'rough_a1'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates
        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt 


class A1RoughCfgSAC( LeggedRobotCfgAlg ):
    seed = 20021110
    runner_class_name = 'OffPolicyRunner'
    class algorithm( LeggedRobotCfgAlg.algorithm ):
        # for PPO
        # entropy_coef = 0.01
        # sac
        # training params
        num_learning_epochs=2
        num_mini_batches=4
        gamma=0.99
        reward_scale=1
        learning_rate=5e-4
        soft_target_tau=0.005
        target_update_period=1
        target_entropy=None
    class runner( LeggedRobotCfgAlg.runner ):
        run_name = ''
        experiment_name = 'rough_a1'
        policy_class_name = 'SoftActorCritic'
        algorithm_class_name = 'SAC'
        num_steps_per_env = 256 # per iteration
        max_iterations = 1500 # number of policy updates
