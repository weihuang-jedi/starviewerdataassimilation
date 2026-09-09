#!/usr/bin/env python
import argparse
import os
import numpy as np
import torch
from scipy.spatial import cKDTree

class Advanced3DIcosahedralHierarchyBuilder:
    """
    Generates a 3D multi-scale hierarchical weather mesh topology pipeline.
    Handles simultaneous horizontal icosahedral subdivision and vertical striding,
    compiling flattened 3D spatial edges and cross-tier bipartite skip connections.
    """
    def __init__(self, max_level: int = 4, output_dir: str = "."):
        self.max_level = max_level
        self.output_dir = output_dir

        self.full_vertical_levels = np.array([
            2, 10, 20, 50, 75, 100, 150, 200, 300, 400,
            500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000,
            4500, 5000, 6000, 7000, 8000, 9000, 10000, 11500, 13000, 15000,
            17500, 20000
        ], dtype=np.float32)

        self.vertical_strides_by_level = {
            4: 1,   # M4 uses all 32 levels
            3: 2,   # M3 strides by 2 -> 16 levels
            2: 4,   # M2 strides by 4 -> 8 levels
            1: 8,   # M1 strides by 8 -> 4 levels
            0: 8    # M0 strides by 8 -> 4 levels
        }

    def _get_base_icosahedron(self):
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ], dtype=np.float64)
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)
        return vertices, faces

    def _compile_horizontal_edges(self, vertices: np.ndarray, faces: np.ndarray, level: int) -> torch.Tensor:
        edges_src, edges_dst = [], []
        for face in faces:
            n0, n1, n2 = face[0], face[1], face[2]
            edges_src.extend([n0, n1, n1, n2, n2, n0])
            edges_dst.extend([n1, n0, n2, n1, n0, n2])

        # 3D spatial radius search to guarantee seam-free boundary connectivity
        tree = cKDTree(vertices)
        max_dist = 1.8 / (2 ** level)
        pairs = tree.query_pairs(r=max_dist, output_type='set')
        for u, v in pairs:
            edges_src.extend([u, v])
            edges_dst.extend([v, u])

        edge_stack = np.vstack((edges_src, edges_dst))
        unique_edges = np.unique(edge_stack, axis=1)
        return torch.from_numpy(unique_edges).long()

    def _get_vertical_layer_indices(self, level: int):
        stride = self.vertical_strides_by_level.get(level, 1)
        return np.arange(0, len(self.full_vertical_levels), stride)

    def compile_hierarchy(self, requested_cross_pairs: list):
        os.makedirs(self.output_dir, exist_ok=True)

        vertices, faces = self._get_base_icosahedron()
        horizontal_counts = {0: len(vertices)}
        subdivided_layers = {0: (vertices.copy(), faces.copy())}

        for level in range(1, self.max_level + 1):
            midpoint_cache = {}
            verts_list = vertices.tolist()

            def get_midpoint(p1_idx, p2_idx):
                edge = tuple(sorted((p1_idx, p2_idx)))
                if edge in midpoint_cache:
                    return midpoint_cache[edge]
                v1 = np.array(verts_list[p1_idx])
                v2 = np.array(verts_list[p2_idx])
                mid = v1 + v2
                mid /= np.linalg.norm(mid)
                verts_list.append(mid.tolist())
                new_idx = len(verts_list) - 1
                midpoint_cache[edge] = new_idx
                return new_idx

            new_faces = []
            for face in faces:
                v0, v1, v2 = face[0], face[1], face[2]
                m01 = get_midpoint(v0, v1)
                m12 = get_midpoint(v1, v2)
                m20 = get_midpoint(v2, v0)
                new_faces.append([v0, m01, m20])
                new_faces.append([v1, m12, m01])
                new_faces.append([v2, m20, m12])
                new_faces.append([m01, m12, m20])

            vertices = np.array(verts_list)
            faces = np.array(new_faces, dtype=np.int32)
            horizontal_counts[level] = len(vertices)
            subdivided_layers[level] = (vertices.copy(), faces.copy())

        # Save horizontal intra-level topology maps
        print("[STAGE 1] Serializing horizontal intra-level topology components...")
        for level in range(self.max_level + 1):
            l_verts, l_faces = subdivided_layers[level]
            edge_idx = self._compile_horizontal_edges(l_verts, l_faces, level)
            torch.save(edge_idx, os.path.join(self.output_dir, f"edge_index_m{level}.pt"))

        # =================================================================
        # 3D COMPILATION AND CROSS-TIER MAPPING (3D EUCLIDEAN NEAREST NEIGHBOR)
        # =================================================================
        print(f"\n[STAGE 2] Building 3D cross-tier vertical + horizontal mapping arrays...")

        for x, y in requested_cross_pairs:
            if x not in horizontal_counts or y not in horizontal_counts:
                continue

            x_verts, _ = subdivided_layers[x]
            y_verts, _ = subdivided_layers[y]

            x_h_count = len(x_verts)
            y_h_count = len(y_verts)

            x_v_indices = self._get_vertical_layer_indices(x)
            y_v_indices = self._get_vertical_layer_indices(y)

            # Build 3D spatial tree for destination grid y
            tree_y = cKDTree(y_verts)
            # Find nearest node in y for each node in x
            _, h_nearest_y = tree_y.query(x_verts, k=1)

            map_src = []
            map_dst = []

            for x_v_local_idx, v_layer_idx in enumerate(x_v_indices):
                if v_layer_idx not in y_v_indices:
                    continue
                y_v_local_idx = np.where(y_v_indices == v_layer_idx)[0][0]

                for h_node_x, h_node_y in enumerate(h_nearest_y):
                    flat_index_x = (x_v_local_idx * x_h_count) + h_node_x
                    flat_index_y = (y_v_local_idx * y_h_count) + h_node_y

                    map_src.append(flat_index_x)
                    map_dst.append(flat_index_y)

            bipartite_3d_skip_map = torch.tensor([map_src, map_dst], dtype=torch.long)
            out_filename = f"map_m{x}_to_m{y}.pt"
            torch.save(bipartite_3d_skip_map, os.path.join(self.output_dir, out_filename))

            print(f" -> Generated 3D skip connection: {out_filename}")

        print(f"\nSUCCESS: 3D Multiscale Graph Hierarchy finalized inside '{self.output_dir}'!\n")

def main():
    parser = argparse.ArgumentParser(description="Compile customizable 3D multi-scale graphs across icosahedral grid steps.")
    parser.add_argument("-k", "--max_level", type=int, default=4, help="Maximum target resolution scale level")
    parser.add_argument("-o", "--output_dir", default=".", help="Target output workspace directory for PT tensors")
    args = parser.parse_args()

    custom_skip_pairs = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
        (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)
    ]

    builder = Advanced3DIcosahedralHierarchyBuilder(max_level=args.max_level, output_dir=args.output_dir)
    builder.compile_hierarchy(requested_cross_pairs=custom_skip_pairs)

if __name__ == "__main__":
    main()
