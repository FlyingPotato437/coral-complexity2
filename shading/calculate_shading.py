import numpy as np
import pyvista as pv
import os
import time
from tqdm import tqdm
import multiprocessing as mp
import csv
from functools import partial
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate shading scores for plots.')
    parser.add_argument(
        '--input_dir', help='Input directory containing .ply files.')
    parser.add_argument('--cpu_limit', type=int,
                        help='CPU limit for parallel processing.', default=5)
    args = parser.parse_args()
    return parser, args


class AABB:
    def __init__(self, min_bound, max_bound):
        self.min_bound = min_bound
        self.max_bound = max_bound

    def intersect(self, ray_origin, ray_direction):
        t_min = np.zeros(3)
        t_max = np.zeros(3)
        for i in range(3):
            if ray_direction[i] != 0:
                t_min[i] = (self.min_bound[i] - ray_origin[i]) / \
                    ray_direction[i]
                t_max[i] = (self.max_bound[i] - ray_origin[i]) / \
                    ray_direction[i]
                if t_min[i] > t_max[i]:
                    t_min[i], t_max[i] = t_max[i], t_min[i]
            else:
                t_min[i] = float(
                    '-inf') if ray_origin[i] >= self.min_bound[i] else float('inf')
                t_max[i] = float(
                    'inf') if ray_origin[i] <= self.max_bound[i] else float('-inf')
        t_enter = max(t_min)
        t_exit = min(t_max)
        return t_enter <= t_exit and t_exit > 0


class BVHNode:
    def __init__(self, start, end, aabb):
        self.start = start
        self.end = end
        self.aabb = aabb
        self.left = None
        self.right = None


def build_bvh(triangles, indices, start, end, depth=0, max_depth=20):
    if start >= end or depth > max_depth:
        return None

    aabb_min = np.min(triangles[indices[start:end]], axis=(0, 1))
    aabb_max = np.max(triangles[indices[start:end]], axis=(0, 1))
    node = BVHNode(start, end, AABB(aabb_min, aabb_max))

    if end - start <= 4 or depth == max_depth:  # Leaf node
        return node

    # Choose longest axis to split
    axis = np.argmax(aabb_max - aabb_min)
    mid = (start + end) // 2

    # Sort indices based on triangle centroids
    centroids = np.mean(triangles[indices[start:end]], axis=1)
    indices[start:end] = indices[start:end][np.argsort(centroids[:, axis])]

    node.left = build_bvh(triangles, indices, start, mid, depth + 1, max_depth)
    node.right = build_bvh(triangles, indices, mid, end, depth + 1, max_depth)
    return node


def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2):
    epsilon = 1e-6
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if abs(a) < epsilon:
        return False
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(edge2, q)
    return t > epsilon


def intersect_bvh(node, ray_origin, ray_direction, triangles, indices):
    if node is None or not node.aabb.intersect(ray_origin, ray_direction):
        return False

    if node.left is None and node.right is None:
        for i in range(node.start, node.end):
            triangle = triangles[indices[i]]
            if ray_triangle_intersect(ray_origin, ray_direction, triangle[0], triangle[1], triangle[2]):
                return True
        return False

    return intersect_bvh(node.left, ray_origin, ray_direction, triangles, indices) or \
        intersect_bvh(node.right, ray_origin,
                      ray_direction, triangles, indices)


def process_chunk(args):
    chunk, bvh_root, triangles, indices, light_dir = args
    shadowed = np.zeros(len(chunk), dtype=bool)
    for i, point in enumerate(chunk):
        if intersect_bvh(bvh_root, point, light_dir, triangles, indices):
            shadowed[i] = True
    return shadowed


def point_in_box(point, box_min, box_max):
    return np.all(point >= box_min) and np.all(point <= box_max)


def triangle_intersects_box(v0, v1, v2, box_min, box_max):
    # Check if any vertex is inside the box
    if any(point_in_box(v, box_min, box_max) for v in [v0, v1, v2]):
        return True

    # Check if the triangle intersects any of the 12 edges of the box
    edges = [
        (box_min, [box_max[0], box_min[1], box_min[2]]),
        (box_min, [box_min[0], box_max[1], box_min[2]]),
        (box_min, [box_min[0], box_min[1], box_max[2]]),
        (box_max, [box_min[0], box_max[1], box_max[2]]),
        (box_max, [box_max[0], box_min[1], box_max[2]]),
        (box_max, [box_max[0], box_max[1], box_min[2]]),
        ([box_min[0], box_max[1], box_min[2]],
         [box_max[0], box_max[1], box_min[2]]),
        ([box_min[0], box_max[1], box_min[2]],
         [box_min[0], box_max[1], box_max[2]]),
        ([box_min[0], box_min[1], box_max[2]],
         [box_max[0], box_min[1], box_max[2]]),
        ([box_min[0], box_min[1], box_max[2]],
         [box_min[0], box_max[1], box_max[2]]),
        ([box_max[0], box_min[1], box_min[2]],
         [box_max[0], box_max[1], box_min[2]]),
        ([box_max[0], box_min[1], box_min[2]],
         [box_max[0], box_min[1], box_max[2]])
    ]

    for edge_start, edge_end in edges:
        if ray_triangle_intersect(edge_start, np.array(edge_end) - np.array(edge_start), v0, v1, v2):
            return True

    return False


def triangle_intersects_box_wrapper(triangle, box_min, box_max):
    return triangle_intersects_box(triangle[0], triangle[1], triangle[2], box_min, box_max)


def parallel_triangle_filtering(triangles, box_min, box_max, cpu_limit):
    num_processes = min(
        mp.cpu_count(), cpu_limit) if cpu_limit else mp.cpu_count()
    # Adjust chunk size for better load balancing
    chunk_size = max(1, len(triangles) // (num_processes * 10))

    with mp.Pool(num_processes) as pool:
        filtered_indices = list(tqdm(
            pool.imap(
                partial(triangle_intersects_box_wrapper,
                        box_min=box_min, box_max=box_max),
                triangles,
                chunksize=chunk_size
            ),
            total=len(triangles),
            desc="Parallel triangle filtering",
            mininterval=0.1,  # Update at most every 0.1 seconds
            smoothing=0.1  # Smooth out the updates
        ))

    return np.where(filtered_indices)[0]


def calculate_structure_shading(mesh, light_dir, point_of_interest=None, window_size=None, sample_size=1000000, cpu_limit=None):
    print("Preparing mesh data...")
    with tqdm(total=3, desc="Mesh preparation") as pbar:
        mesh.compute_normals(inplace=True)
        pbar.update(1)
        points = mesh.points
        pbar.update(1)
        faces = mesh.faces.reshape(-1, 4)[:, 1:4]
        pbar.update(1)
    triangles = points[faces]

    if point_of_interest is not None and window_size is not None:
        # Bounding box calculation
        box_min = point_of_interest - window_size / 2
        box_max = point_of_interest + window_size / 2

        print("Filtering points...")
        mask = np.all((points >= box_min) & (points <= box_max), axis=1)
        window_points = points[mask]

        if len(window_points) > sample_size:
            print(
                f"Sampling {sample_size} points from {len(window_points)} points in the window...")
            sampled_indices = np.random.choice(
                len(window_points), sample_size, replace=False)
            sampled_points = window_points[sampled_indices]
        else:
            sampled_points = window_points

        print("Filtering triangles...")
        indices = parallel_triangle_filtering(
            triangles, box_min, box_max, cpu_limit)
    else:
        # Full model calculation
        print("Processing full model...")
        if len(points) > sample_size:
            print(
                f"Sampling {sample_size} points from {len(points)} total points...")
            sampled_indices = np.random.choice(
                len(points), sample_size, replace=False)
            sampled_points = points[sampled_indices]
        else:
            sampled_points = points
        indices = np.arange(len(triangles))

    print("Building BVH...")
    build_start = time.time()
    with tqdm(total=len(indices), desc="Building BVH") as pbar:
        def bvh_progress_callback(progress):
            pbar.n = progress
            pbar.refresh()
        bvh_root = build_bvh(triangles, indices, 0, len(indices))
    print(f"BVH built in {time.time() - build_start:.2f} seconds.")

    if cpu_limit is None:
        num_processes = mp.cpu_count()
    else:
        num_processes = max(1, min(cpu_limit, mp.cpu_count()))

    chunk_size = max(1, len(sampled_points) // (num_processes * 2))
    chunks = [sampled_points[i:i + chunk_size]
              for i in range(0, len(sampled_points), chunk_size)]

    print(f"Using {num_processes} CPU cores to process {len(chunks)} chunks...")

    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, [
                      (chunk, bvh_root, triangles, indices, light_dir) for chunk in chunks]),
            total=len(chunks),
            desc="Processing chunks",
            mininterval=0.1,  # Update at most every 0.1 seconds
            smoothing=0.1  # Smooth out the updates
        ))

    print("All chunks processed. Calculating final result...")
    shadowed = np.concatenate(results)
    shaded_percentage = np.mean(shadowed) * 100
    return shaded_percentage


def write_output(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Plot Name', 'Shaded Percentage',
                        'Illuminated Percentage', 'Calculation Time (s)'])
        for result in results:
            writer.writerow([result['plot_name'],
                             f"{result['shaded_percentage']:.2f}",
                             f"{result['illuminated_percentage']:.2f}",
                             f"{result['calculation_time']:.2f}"])


def process_single_plot(plot_info, use_bounding_box, cpu_limit):
    mesh_path, plot_name = plot_info

    if not os.path.exists(mesh_path):
        print(f"3D model file not found: {mesh_path}")
        return None

    print(f"Processing plot: {plot_name}")
    print("Loading 3D mesh...")
    start_time = time.time()
    try:
        mesh = pv.read(mesh_path)
    except Exception as e:
        print(f"Failed to load 3D model: {e}")
        return None

    print(f"Mesh loaded in {time.time() - start_time:.2f} seconds")
    print(f"Number of points: {mesh.n_points}")
    print(f"Number of faces: {mesh.n_cells}")

    light_dir = np.array([0, 0, -1])  # Negative z-direction for top-down light

    if use_bounding_box:
        point_of_interest, window_size = interactive_bounding_box(mesh)
        print(f"Selected point of interest: {point_of_interest}")
        print(f"Selected window size: {window_size}")
    else:
        point_of_interest = None
        window_size = None

    print("Calculating shading percentage based on coral structure...")
    calculation_start_time = time.time()
    shaded_percentage = calculate_structure_shading(
        mesh, light_dir, point_of_interest, window_size, cpu_limit=cpu_limit)
    calculation_time = time.time() - calculation_start_time

    print(f'Shaded Percentage: {shaded_percentage:.2f}%')
    print(f'Illuminated Percentage: {100 - shaded_percentage:.2f}%')
    print(f'Calculation Time: {calculation_time:.2f} seconds')

    return {
        'plot_name': plot_name,
        'shaded_percentage': shaded_percentage,
        'illuminated_percentage': 100 - shaded_percentage,
        'calculation_time': calculation_time
    }


def main():
    parser, args = parse_args()
    assert args.input_dir, (
        'Please specify the input plot folder with the argument "--input_dir" ')

    plots = glob.glob(os.path.join(args.input_dir, "*.ply"))

    results = []
    for mesh_path in plots:
        if not os.path.exists(mesh_path):
            print(f"3D model file not found: {mesh_path}")
            continue

        print(f"\nProcessing plot: {os.path.basename(mesh_path)}")
        print("Loading 3D mesh...")
        start_time = time.time()
        try:
            mesh = pv.read(mesh_path)
        except Exception as e:
            print(f"Failed to load 3D model: {e}")
            continue

        print(f"Mesh loaded in {time.time() - start_time:.2f} seconds")
        print(f"Number of points: {mesh.n_points}")
        print(f"Number of faces: {mesh.n_cells}")

        # Negative z-direction for top-down light
        light_dir = np.array([0, 0, -1])

        point_of_interest, window_size = None, None

        print("Calculating shading percentage based on coral structure...")
        calculation_start_time = time.time()
        shaded_percentage = calculate_structure_shading(
            mesh, light_dir, point_of_interest, window_size, cpu_limit=args.cpu_limit)
        calculation_time = time.time() - calculation_start_time

        print(f'Plot name: {os.path.basename(mesh_path)}')
        print(f'Shaded Percentage: {shaded_percentage:.2f}%')
        print(f'Illuminated Percentage: {100 - shaded_percentage:.2f}%')
        print(f'Calculation Time: {calculation_time:.2f} seconds')
        print()

        results.append({
            'plot_name': os.path.basename(mesh_path),
            'shaded_percentage': shaded_percentage,
            'illuminated_percentage': 100 - shaded_percentage,
            'calculation_time': calculation_time
        })

        print("\n" + "="*50 + "\n")

    output_file = 'shading_results.csv'
    write_output(results, output_file)
    print(f"Results written to {output_file}")

    print("All plots processed successfully.")


if __name__ == "__main__":
    main()
