import numpy as np
import torch

def qmul_np(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = np.array([
        r[..., 0] * q[..., 0] - r[..., 1] * q[..., 1] - r[..., 2] * q[..., 2] - r[..., 3] * q[..., 3],
        r[..., 0] * q[..., 1] + r[..., 1] * q[..., 0] - r[..., 2] * q[..., 3] + r[..., 3] * q[..., 2],
        r[..., 0] * q[..., 2] + r[..., 1] * q[..., 3] + r[..., 2] * q[..., 0] - r[..., 3] * q[..., 1],
        r[..., 0] * q[..., 3] - r[..., 1] * q[..., 2] + r[..., 2] * q[..., 1] + r[..., 3] * q[..., 0]
    ])

    return np.stack(terms, axis=-1)

def qrot_np(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    qw = q[..., :1]
    
    return 2.0 * np.cross(qvec, np.cross(qvec, v) + qw * v) + v

def qinv_np(q):
    """
    Inverse quaternion(s) q.
    """
    assert q.shape[-1] == 4
    
    q = q.copy()
    q[..., 1:] *= -1
    return q

def qbetween_np(v0, v1):
    """
    Find quaternion representing the rotation between two vectors.
    """
    assert v0.shape == v1.shape
    assert v0.shape[-1] == 3
    
    # Normalize input vectors
    v0_norm = np.sqrt(np.sum(v0 * v0, axis=-1))
    v1_norm = np.sqrt(np.sum(v1 * v1, axis=-1))
    v0 = v0 / v0_norm[..., np.newaxis]
    v1 = v1 / v1_norm[..., np.newaxis]
    
    # Compute the cross product
    w = np.cross(v0, v1)
    w_norm = np.sqrt(np.sum(w * w, axis=-1))
    
    # If vectors are parallel, return identity quaternion
    mask = w_norm < 1e-6
    if np.any(mask):
        result = np.zeros(v0.shape[:-1] + (4,))
        result[..., 0] = 1.0
        if mask.shape:
            result[~mask] = np.concatenate([
                (1 + np.sum(v0[~mask] * v1[~mask], axis=-1))[..., np.newaxis],
                w[~mask]], axis=-1)
        return result
    
    # Otherwise compute the rotation quaternion
    w = w / w_norm[..., np.newaxis]
    theta = np.arccos(np.sum(v0 * v1, axis=-1))
    sin_theta = np.sin(theta / 2)
    cos_theta = np.cos(theta / 2)
    
    return np.concatenate([cos_theta[..., np.newaxis], sin_theta[..., np.newaxis] * w], axis=-1)

def quaternion_to_cont6d_np(quaternions):
    """
    Convert quaternion(s) to 6D continuous rotation representation.
    Based on Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
    """
    assert quaternions.shape[-1] == 4
    
    # Convert to rotation matrix first
    r = quaternion_to_matrix_np(quaternions)
    
    # Extract the first two columns
    return r[..., :, :2].reshape(quaternions.shape[:-1] + (6,))

def quaternion_to_matrix_np(quaternions):
    """
    Convert quaternion(s) to rotation matrix.
    """
    assert quaternions.shape[-1] == 4
    
    # Normalize quaternion
    q = quaternions / np.sqrt(np.sum(quaternions * quaternions, axis=-1))[..., np.newaxis]
    
    # Extract components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Compute rotation matrix elements
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*w*z
    r02 = 2*x*z + 2*w*y
    r10 = 2*x*y + 2*w*z
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*w*x
    r20 = 2*x*z - 2*w*y
    r21 = 2*y*z + 2*w*x
    r22 = 1 - 2*x*x - 2*y*y
    
    # Stack into matrix
    return np.stack([
        np.stack([r00, r01, r02], axis=-1),
        np.stack([r10, r11, r12], axis=-1),
        np.stack([r20, r21, r22], axis=-1)
    ], axis=-2)

# Constants for the motion processing
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],  # Right leg
    [0, 1, 4, 7, 10],  # Left leg
    [0, 3, 6, 9, 12, 15],  # Spine
    [9, 14, 17, 19, 21],  # Right arm
    [9, 13, 16, 18, 20]   # Left arm
]

# Joint indices
l_idx1, l_idx2 = 5, 8  # Lower legs
fid_r, fid_l = [8, 11], [7, 10]  # Right/Left foot
face_joint_indx = [2, 1, 17, 16]  # Face direction (r_hip, l_hip, sdr_r, sdr_l)
r_hip, l_hip = 2, 1  # Right/Left hip 