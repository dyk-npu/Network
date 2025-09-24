import argparse
import pathlib

import dgl
import numpy as np
import torch
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import repeat
import signal

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from config.data_config import DataConfig


def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat = []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph


def process_one_file(arguments):
    fn, args, config = arguments
    fn_stem = fn.stem
    output_path = pathlib.Path(args.output)

    try:
        solids = load_step(fn)
        if not solids:
            print(f"Warning: No solids found in {fn_stem}")
            return None

        solid = solids[0]  # Take the first solid
        graph = build_graph(
            solid, config.brep_curv_u_samples, config.brep_surf_u_samples, config.brep_surf_v_samples
        )
        output_file = output_path.joinpath(fn_stem + ".bin")
        dgl.data.utils.save_graphs(str(output_file), [graph])
        return fn_stem

    except Exception as e:
        print(f"Error processing {fn_stem}: {str(e)}")
        return None


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process(args, config=None):
    if config is None:
        config = DataConfig()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    step_files = list(input_path.glob("*.st*p"))
    # for fn in tqdm(step_files):
    #     process_one_file(fn, args, config)
    pool = Pool(processes=config.brep_num_processes, initializer=initializer)
    try:
        results = list(tqdm(pool.imap(process_one_file, zip(step_files, repeat(args), repeat(config))), total=len(step_files)))
        # Filter out None results (failed processing)
        successful_results = [r for r in results if r is not None]
        print(f"Successfully processed {len(successful_results)} out of {len(step_files)} files.")
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
        pool.terminate()
        pool.join()
    finally:
        pool.close()
        pool.join()


def main():
    parser = argparse.ArgumentParser(
        "Convert solid models to face-adjacency graphs with UV-grid features"
    )
    parser.add_argument("input", type=str, help="Input folder of STEP files")
    parser.add_argument("output", type=str, help="Output folder of DGL graph BIN files")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (optional, uses DataConfig defaults if not provided)"
    )
    args = parser.parse_args()

    # Load configuration
    config = DataConfig()
    if args.config:
        # If custom config file is provided, load it
        # TODO: Add config file loading logic here
        pass

    process(args, config)


if __name__ == "__main__":
    main()