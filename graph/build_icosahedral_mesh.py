import argparse
import os
import numpy as np
import torch
import xarray as xr
from scipy.spatial import cKDTree

class IcosahedralMeshGenerator:
    """
    Constructs a recursively subdivided spherical icosahedral grid (Mk)
    matching the multi-mesh structure of Google GraphCast.
    """
    def __init__(self, subdivisions: int = 4, output_dir: str = "."):
        self.subdivisions = subdivisions
        self.output_dir = output_dir

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

    def subdivide(self) -> tuple:
        print(f"[INIT] Constructing base M0 Icosahedron...")
        vertices, faces = self._get_base_icosahedron()

        for level in range(1, self.subdivisions + 1):
            print(f"[SUBDIVIDE] Refining level M{level-1} -> M{level}...")
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

        return vertices, faces

    def build_and_export(self):
        vertices, faces = self.subdivide()
        num_nodes = len(vertices)
        num_faces = len(faces)
        print(f"\nCompleted Icosahedral M{self.subdivisions} compilation.")
        print(f" -> Total Spherical Nodes : {num_nodes:,}")
        print(f" -> Total Triangular Faces: {num_faces:,}")

        # Convert 3D Cartesian coordinates back to standard Latitude / Longitude
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        lats = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
        lons_raw = np.degrees(np.arctan2(y, x))

        lons_360 = np.mod(lons_raw, 360)

        # =================================================================
        # 1. COMPILE PYTORCH DIRECTED EDGES (FACE EDGES + 3D RADIUS EDGES)
        # =================================================================
        print("[STAGE 1] Formatting directed graph edge connections in 3D Euclidean space...")
        edges_src = []
        edges_dst = []
        for face in faces:
            n0, n1, n2 = face[0], face[1], face[2]
            edges_src.extend([n0, n1, n1, n2, n2, n0])
            edges_dst.extend([n1, n0, n2, n1, n0, n2])

        # Add 3D Euclidean spatial KD-Tree query to close any geometric neighbor gaps
        tree = cKDTree(vertices)
        # Distance cutoff scales inversely with subdivision level
        max_dist = 1.8 / (2 ** self.subdivisions)
        pairs = tree.query_pairs(r=max_dist, output_type='set')
        for u, v in pairs:
            edges_src.extend([u, v])
            edges_dst.extend([v, u])

        edge_stack = np.vstack((edges_src, edges_dst))
        unique_edges = np.unique(edge_stack, axis=1)
        edge_index = torch.from_numpy(unique_edges).long()

        os.makedirs(self.output_dir, exist_ok=True)
        pt_path = os.path.join(self.output_dir, f"icosahedral_edge_index_m{self.subdivisions}.pt")
        torch.save(edge_index, pt_path)
        print(f" -> Saved graph topology file to: {pt_path}")

        # =================================================================
        # 2. COMPILE UGRID NETCDF FILES
        # =================================================================
        print("[STAGE 2] Packaging UGRID NetCDF geometry structures...")
        ds_mesh = xr.Dataset(
            data_vars={
                "icosahedral_mesh": (
                    [],
                    0,
                    {
                        "cf_role": "mesh_topology",
                        "topology_dimension": 2,
                        "node_coordinates": "longitude latitude",
                        "face_node_connectivity": "face_nodes",
                        "face_dimension": "face"
                    }
                ),
                "face_nodes": (
                    ["face", "three"],
                    faces,
                    {
                        "cf_role": "face_node_connectivity",
                        "start_index": 0,
                        "long_name": "Indices of nodes defining each triangle face"
                    }
                ),
                "longitude": (["node"], lons_360, {"units": "degrees_east", "standard_name": "longitude"}),
                "latitude": (["node"], lats, {"units": "degrees_north", "standard_name": "latitude"}),
                "x_cartesian": (["node"], x, {"units": "m", "long_name": "Cartesian Coordinate X"}),
                "y_cartesian": (["node"], y, {"units": "m", "long_name": "Cartesian Coordinate Y"}),
                "z_cartesian": (["node"], z, {"units": "m", "long_name": "Cartesian Coordinate Z"}),
            },
            coords={
                "node": np.arange(num_nodes),
                "face": np.arange(num_faces),
                "three": np.arange(3)
            },
            attrs={
                "title": f"Global Spherical Icosahedral Grid (Level M{self.subdivisions})",
                "conventions": "CF-1.8 UGRID-1.0",
                "subdivision_level": self.subdivisions
            }
        )

        nc_path = os.path.join(self.output_dir, f"global_icosahedral_mesh_m{self.subdivisions}.nc")
        ds_mesh.to_netcdf(nc_path, format="NETCDF4")
        print(f" -> Saved UGRID NetCDF file to: {nc_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Create a recursively subdivided spherical icosahedral mesh (M_k).")
    parser.add_argument("-k", "--subdivisions", type=int, default=4, help="Subdivision level k (default: 4, 2562 nodes)")
    parser.add_argument("-o", "--output_dir", default=".", help="Target folder for computed files")
    args = parser.parse_args()

    generator = IcosahedralMeshGenerator(subdivisions=args.subdivisions, output_dir=args.output_dir)
    generator.build_and_export()

if __name__ == "__main__":
    main()
