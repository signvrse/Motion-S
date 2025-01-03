import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import math

class BVHJoint:
    def __init__(self, name: str, parent=None):
        self.name = name
        self.parent = parent
        self.children: List[BVHJoint] = []
        self.offset = np.zeros(3)
        self.channels: List[str] = []
        self.channel_indices: List[int] = []

class BVHReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.joints: Dict[str, BVHJoint] = {}
        self.root: Optional[BVHJoint] = None
        self.frames = 0
        self.frame_time = 0.0
        self.motion_data = None
        self.joint_positions = None

    def read_hierarchy(self, lines: List[str], index: int) -> Tuple[int, Optional[BVHJoint]]:
        """Read the HIERARCHY section of the BVH file."""
        joint_stack = []
        current_joint = None
        
        while index < len(lines):
            line = lines[index].strip()
            tokens = line.split()
            
            if not tokens:
                index += 1
                continue
                
            if tokens[0] == "MOTION":
                break
                
            if tokens[0] in ["JOINT", "End", "ROOT"]:
                name = tokens[1] if tokens[0] != "End" else f"{current_joint.name}_End"
                joint = BVHJoint(name, current_joint)
                self.joints[name] = joint
                
                if tokens[0] == "ROOT":
                    self.root = joint
                elif current_joint:
                    current_joint.children.append(joint)
                    
                joint_stack.append(joint)
                current_joint = joint
                
            elif tokens[0] == "OFFSET":
                current_joint.offset = np.array([float(x) for x in tokens[1:4]])
                
            elif tokens[0] == "CHANNELS":
                num_channels = int(tokens[1])
                current_joint.channels = tokens[2:2+num_channels]
                
            elif tokens[0] == "}":
                joint_stack.pop()
                if joint_stack:
                    current_joint = joint_stack[-1]
                else:
                    current_joint = None
                    
            index += 1
            
        return index, current_joint

    def read_motion(self, lines: List[str], index: int):
        """Read the MOTION section of the BVH file."""
        index += 1  # Skip "MOTION" line
        self.frames = int(lines[index].split()[-1])
        index += 1
        self.frame_time = float(lines[index].split()[-1])
        index += 1
        
        motion_data = []
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1
                continue
            values = [float(x) for x in line.split()]
            motion_data.append(values)
            index += 1
            
        self.motion_data = np.array(motion_data)

    def read_bvh(self):
        """Read the BVH file and extract hierarchy and motion data."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if line.startswith("HIERARCHY"):
                index += 1
                index, _ = self.read_hierarchy(lines, index)
            elif line.startswith("MOTION"):
                self.read_motion(lines, index)
                break
            else:
                index += 1

    def compute_joint_positions(self):
        """Compute global joint positions for each frame."""
        if self.motion_data is None:
            raise ValueError("No motion data loaded. Call read_bvh() first.")
            
        num_frames = len(self.motion_data)
        positions = {name: np.zeros((num_frames, 3)) for name in self.joints}
        
        def process_joint(joint: BVHJoint, frame_idx: int, parent_matrix=None):
            if parent_matrix is None:
                parent_matrix = np.eye(4)
                
            # Create local transformation matrix
            local_matrix = np.eye(4)
            
            # Apply translation from offset
            local_matrix[:3, 3] = joint.offset
            
            # Apply channel transformations
            if joint.channels:
                channel_values = []
                channel_start_idx = 0
                if joint == self.root:  # Root joint might have position channels
                    if "Xposition" in joint.channels:
                        local_matrix[0, 3] = self.motion_data[frame_idx][channel_start_idx]
                        channel_start_idx += 1
                    if "Yposition" in joint.channels:
                        local_matrix[1, 3] = self.motion_data[frame_idx][channel_start_idx]
                        channel_start_idx += 1
                    if "Zposition" in joint.channels:
                        local_matrix[2, 3] = self.motion_data[frame_idx][channel_start_idx]
                        channel_start_idx += 1
                
                # Handle rotation channels
                rotation_matrix = np.eye(3)
                for channel in joint.channels[channel_start_idx:]:
                    angle = self.motion_data[frame_idx][channel_start_idx]
                    channel_start_idx += 1
                    
                    if "rotation" in channel:
                        c = math.cos(math.radians(angle))
                        s = math.sin(math.radians(angle))
                        
                        if channel.startswith("X"):
                            rot = np.array([[1, 0, 0],
                                         [0, c, -s],
                                         [0, s, c]])
                        elif channel.startswith("Y"):
                            rot = np.array([[c, 0, s],
                                         [0, 1, 0],
                                         [-s, 0, c]])
                        elif channel.startswith("Z"):
                            rot = np.array([[c, -s, 0],
                                         [s, c, 0],
                                         [0, 0, 1]])
                        rotation_matrix = rotation_matrix @ rot
                
                local_matrix[:3, :3] = rotation_matrix
            
            # Compute global transformation
            global_matrix = parent_matrix @ local_matrix
            
            # Store joint position
            positions[joint.name][frame_idx] = global_matrix[:3, 3]
            
            # Process children
            for child in joint.children:
                process_joint(child, frame_idx, global_matrix)
        
        # Process all frames
        for frame_idx in range(num_frames):
            process_joint(self.root, frame_idx)
            
        self.joint_positions = positions
        return positions

def main(bvh_file):
    # Example usage

    reader = BVHReader(bvh_file)
    reader.read_bvh()
    joint_positions = reader.compute_joint_positions()
    
    # Print joint names and their positions at frame 0
    for joint_name, positions in joint_positions.items():
        print(f"{joint_name}: {positions[0]}")  # Print position at first frame

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", type=str, required=True)
    args = parser.parse_args()
    bvh_file = args.bvh_file
    main(bvh_file) 