import numpy as np
import torch
import os
from typing import Dict, List, Tuple, Optional
import math
from pathlib import Path

from common.skeleton import Skeleton
from common.quaternion import *
from paramUtil import t2m_raw_offsets
from motion_utils import (
    face_joint_indx,
    fid_r,
    fid_l,
    l_idx1,
    l_idx2,
    qrot_np,
    qbetween_np,
    qmul_np,
    qinv_np,
    quaternion_to_cont6d_np,
    t2m_kinematic_chain
)

def process_file(positions: np.ndarray, feet_thre: float = 0.002) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process motion data to extract features."""
    # Create skeleton with raw offsets
    src_skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    tgt_offsets = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    
    # Uniform Skeleton
    positions = uniform_skeleton(positions, tgt_offsets)
    
    # Put on Floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    
    # XZ at origin
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz
    
    # All initially face Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    
    positions = qrot_np(root_quat_init, positions)
    
    # Get Foot Contacts
    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)
        
        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    
    feet_l, feet_r = foot_detect(positions, feet_thre)
    
    # Get Joint Rotations and Velocities
    skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
    cont_6d_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    cont_6d_params = quaternion_to_cont6d_np(cont_6d_params)
    
    # Root rotation and velocities
    r_rot = cont_6d_params[:, 0].copy()
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    
    # Root height
    root_y = positions[:, 0, 1:2]
    
    # Root rotation and linear velocity
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
    
    # Joint rotations
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    
    # Local joint positions
    ric_data = positions[:, 1:].reshape(len(positions), -1)
    
    # Joint velocities
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], positions.shape[1], axis=1),
                       positions[1:] - positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)
    
    # Concatenate all features
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    
    return data, positions, positions, l_velocity

def uniform_skeleton(positions, target_offset):
    """Uniformize skeleton to match target offset."""
    src_skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    
    # Calculate Scale Ratio as the ratio of legs
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    
    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt
    
    # Forward Kinematics
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(positions, tgt_root_pos)
    return new_joints

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npy", type=str, required=True, help="Input npy file in SMPL format")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    joints_dir = output_dir / "new_joints"
    vecs_dir = output_dir / "new_joint_vecs"
    joints_dir.mkdir(parents=True, exist_ok=True)
    vecs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process the motion data
    positions = np.load(args.input_npy)
    
    # Process the file
    try:
        data, ground_positions, positions, l_velocity = process_file(positions, 0.002)
        
        # Save processed data
        output_name = Path(args.input_npy).stem
        np.save(joints_dir / f"{output_name}.npy", positions)
        np.save(vecs_dir / f"{output_name}.npy", data)
        
        print(f"Successfully processed {output_name}")
        print(f"Positions shape: {positions.shape}")
        print(f"Feature vectors shape: {data.shape}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main() 