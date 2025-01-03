import numpy as np
from bvh_to_joints import BVHReader
from typing import Dict, List, Optional

class SMPLConverter:
    def __init__(self):
        # Define SMPL joint indices and their corresponding names
        self.smpl_joints = {
            0: "Hips",           # Root
            1: "LeftHip",
            2: "RightHip",
            3: "Chest",          # Spine1
            4: "LeftKnee",
            5: "RightKnee",
            6: "Chest2",         # Spine2
            7: "LeftAnkle",
            8: "RightAnkle",
            9: "Chest4",         # Spine3
            10: "LeftToe",
            11: "RightToe",
            12: "Neck",
            13: "LeftCollar",
            14: "RightCollar",
            15: "Head",
            16: "LeftShoulder",
            17: "RightShoulder",
            18: "LeftElbow",
            19: "RightElbow",
            20: "LeftWrist",
            21: "RightWrist"
        }
        
        # Define kinematic chains for reference
        self.kinematic_chains = {
            "right_leg": [0, 2, 5, 8, 11],      # Hips -> RightHip -> RightKnee -> RightAnkle -> RightToe
            "left_leg": [0, 1, 4, 7, 10],       # Hips -> LeftHip -> LeftKnee -> LeftAnkle -> LeftToe
            "spine": [0, 3, 6, 9, 12, 15],      # Hips -> Chest -> Chest2 -> Chest4 -> Neck -> Head
            "right_arm": [9, 14, 17, 19, 21],   # Chest4 -> RightCollar -> RightShoulder -> RightElbow -> RightWrist
            "left_arm": [9, 13, 16, 18, 20]     # Chest4 -> LeftCollar -> LeftShoulder -> LeftElbow -> LeftWrist
        }

    def create_joint_mapping(self, bvh_joint_names: List[str]) -> Dict[str, int]:
        """Create a mapping between BVH joint names and SMPL joint indices."""
        mapping = {}
        
        # Create a lowercase version of joint names for case-insensitive matching
        lower_bvh_names = [name.lower() for name in bvh_joint_names]
        
        # Map joints based on name matching
        for smpl_idx, smpl_name in self.smpl_joints.items():
            smpl_lower = smpl_name.lower()
            
            # Try exact match first
            if smpl_lower in lower_bvh_names:
                idx = lower_bvh_names.index(smpl_lower)
                mapping[bvh_joint_names[idx]] = smpl_idx
                continue
            
            # Try partial matches
            for i, bvh_name in enumerate(bvh_joint_names):
                bvh_lower = bvh_name.lower()
                
                # Handle common naming variations
                if smpl_lower == "hips" and any(x in bvh_lower for x in ["hip", "pelvis", "root"]):
                    mapping[bvh_name] = smpl_idx
                elif smpl_lower == "chest" and "spine" in bvh_lower:
                    mapping[bvh_name] = smpl_idx
                elif smpl_lower == "chest2" and "spine2" in bvh_lower:
                    mapping[bvh_name] = smpl_idx
                elif smpl_lower == "chest4" and "spine3" in bvh_lower:
                    mapping[bvh_name] = smpl_idx
                elif smpl_lower in bvh_lower:
                    mapping[bvh_name] = smpl_idx
        
        return mapping

    def convert_positions(self, bvh_positions: Dict[str, np.ndarray], joint_mapping: Dict[str, int]) -> np.ndarray:
        """Convert BVH joint positions to SMPL format."""
        # Get number of frames from first position array
        num_frames = next(iter(bvh_positions.values())).shape[0]
        
        # Initialize output array with shape (frames, 22, 3)
        smpl_positions = np.zeros((num_frames, 22, 3))
        
        # Fill in the positions using the mapping
        for bvh_name, smpl_idx in joint_mapping.items():
            smpl_positions[:, smpl_idx] = bvh_positions[bvh_name]
        
        return smpl_positions

    def normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize the positions according to HumanML3D requirements."""
        # Make a copy to avoid modifying the original
        normalized = positions.copy()
        
        # 1. Center the skeleton at the root (Hips)
        root_position = normalized[:, 0:1]  # Keep broadcasting dimension
        normalized = normalized - root_position
        
        # 2. Scale based on leg length
        left_hip = normalized[:, 1]
        left_knee = normalized[:, 4]
        left_ankle = normalized[:, 7]
        
        leg_length = (np.linalg.norm(left_knee - left_hip, axis=1) + 
                     np.linalg.norm(left_ankle - left_knee, axis=1)).mean()
        
        # Scale to make average leg length = 1
        normalized = normalized / leg_length
        
        # 3. Rotate to face Z+ direction
        # Calculate facing direction from hips
        left_hip = normalized[:, 1]
        right_hip = normalized[:, 2]
        facing = np.cross(right_hip - left_hip, np.array([0, 1, 0]))
        facing_angle = np.arctan2(facing[:, 0], facing[:, 2])
        
        # Create rotation matrices
        cos_a = np.cos(-facing_angle)
        sin_a = np.sin(-facing_angle)
        zeros = np.zeros_like(cos_a)
        ones = np.ones_like(cos_a)
        
        # Rotation matrices around Y axis
        rot_matrices = np.stack([
            np.stack([cos_a, zeros, sin_a], axis=1),
            np.stack([zeros, ones, zeros], axis=1),
            np.stack([-sin_a, zeros, cos_a], axis=1)
        ], axis=1)
        
        # Apply rotation
        normalized = np.matmul(rot_matrices, normalized.transpose(0, 2, 1)).transpose(0, 2, 1)
        
        return normalized

def process_bvh_file(bvh_path: str, output_path: str):
    """Process a BVH file and save the SMPL format positions."""
    # Read BVH file
    reader = BVHReader(bvh_path)
    reader.read_bvh()
    bvh_positions = reader.compute_joint_positions()
    
    # Convert to SMPL format
    converter = SMPLConverter()
    joint_mapping = converter.create_joint_mapping(list(bvh_positions.keys()))
    
    # Print mapping for verification
    print("\nJoint mapping:")
    for bvh_name, smpl_idx in joint_mapping.items():
        print(f"{bvh_name} -> {converter.smpl_joints[smpl_idx]} (SMPL joint {smpl_idx})")
    
    # Convert and normalize positions
    smpl_positions = converter.convert_positions(bvh_positions, joint_mapping)
    normalized_positions = converter.normalize_positions(smpl_positions)
    
    # Save to file
    np.save(output_path, normalized_positions)
    print(f"\nProcessed positions saved to {output_path}")
    print(f"Shape: {normalized_positions.shape}")

def main(bvh_file, output_file):
    # Example usage
    # Replace with desired output path
    process_bvh_file(bvh_file, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    bvh_file = args.bvh_file
    output_file = args.output_file
    main(bvh_file, output_file) 