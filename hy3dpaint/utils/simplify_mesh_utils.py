# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import time
import logging
import trimesh
import pymeshlab

logger = logging.getLogger("hunyuan3d-api")


def remesh_mesh(mesh_path, remesh_path, fast=False, target_count=40000):
    if fast:
        mesh_simplify_fast(mesh_path, remesh_path, target_count=target_count)
    else:
        mesh_simplify_trimesh(mesh_path, remesh_path, target_count=target_count)


def mesh_simplify_fast(inputpath, outputpath, target_count=40000):
    """Fast decimation using pymeshlab cleanup + pyfqmr."""
    import pyfqmr

    t = time.time()
    ms = pymeshlab.MeshSet()
    if inputpath.endswith(".glb"):
        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(inputpath)
    obj_path = outputpath.replace(".glb", ".obj")
    ms.save_current_mesh(obj_path, save_textures=False)
    logger.info("      pymeshlab load+save: %.2fs", time.time() - t)

    t = time.time()
    mesh = trimesh.load(obj_path, force="mesh")
    face_num = mesh.faces.shape[0]
    logger.info("      trimesh load (%d faces): %.2fs", face_num, time.time() - t)

    if face_num > target_count:
        t = time.time()
        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(mesh.vertices, mesh.faces)
        simplifier.simplify_mesh(target_count=target_count, aggressiveness=7, verbose=0)
        vertices, faces, _ = simplifier.getMesh()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        logger.info("      pyfqmr decimation (%d -> %d): %.2fs", face_num, faces.shape[0], time.time() - t)

    mesh.export(outputpath)


def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
    t = time.time()
    ms = pymeshlab.MeshSet()
    if inputpath.endswith(".glb"):
        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(inputpath)
    ms.save_current_mesh(outputpath.replace(".glb", ".obj"), save_textures=False)
    logger.info("      pymeshlab load+save: %.2fs", time.time() - t)

    t = time.time()
    courent = trimesh.load(outputpath.replace(".glb", ".obj"), force="mesh")
    face_num = courent.faces.shape[0]
    logger.info("      trimesh load (%d faces): %.2fs", face_num, time.time() - t)

    if face_num > target_count:
        t = time.time()
        courent = courent.simplify_quadric_decimation(face_count=target_count)
        logger.info("      quadric decimation (%d -> %d): %.2fs", face_num, target_count, time.time() - t)

    t = time.time()
    courent.export(outputpath)
    logger.info("      export: %.2fs", time.time() - t)
