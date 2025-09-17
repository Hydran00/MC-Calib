import numpy as np
import open3d as o3d
import copy
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=40)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh
# Perform Colored ICP (Iterative Closest Point) registration on two point clouds
def perform_icp(source, target, threshold=0.02):
    """
    Perform ICP registration on two point clouds.
    
    :param source: Source point cloud (o3d.geometry.PointCloud)
    :param target: Target point cloud (o3d.geometry.PointCloud)
    :param threshold: Distance threshold for ICP
    :return: Transformation matrix (numpy.ndarray)
    """
    # Apply Colored ICP with Tukey's robust kernel
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP()
    # result = o3d.pipelines.registration.registration_colored_icp(
    result = o3d.pipelines.registration.registration_colored_icp(
        source,
        target,
        threshold,
        np.eye(4),
        estimation,
    )
    print("ICP res:", result)
    return result.transformation

# Example usage
if __name__ == "__main__":
    # Load two point clouds
    pc0 = o3d.io.read_point_cloud("./build/pc0.ply")
    # statistical outlier removal
    cl, ind = pc0.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    pc0 = pc0.select_by_index(ind)
    pc1 = o3d.io.read_point_cloud("./build/pc1.ply")
    # statistical outlier removal
    cl, ind = pc1.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    pc1 = pc1.select_by_index(ind)
    pc2 = o3d.io.read_point_cloud("./build/pc2.ply")
    # statistical outlier removal
    cl, ind = pc2.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    pc2 = pc2.select_by_index(ind)
    ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Filtered Point Clouds")
    tmp = copy.deepcopy(pc0)
    tmp+= pc1
    tmp+= pc2
    vis.add_geometry(tmp)
    vis.add_geometry(ref_frame)
    vis.run()
    vis.destroy_window()
    print("")
    print(vis.get_picked_points()) #[84, 119, 69]

    # filter pavement
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-np.inf, -np.inf, -np.inf), max_bound=(np.inf, 1.0, np.inf))
    pc0 = pc0.crop(box)
    pc1 = pc1.crop(box)
    pc2 = pc2.crop(box)

    o3d.visualization.draw_geometries([pc0, pc1, pc2, ref_frame], window_name="Filtered Point Clouds")


    # Preprocess point clouds
    voxel_size = 0.01  # Adjust voxel size as needed
    pc0_down, _ = preprocess_point_cloud(pc0, voxel_size)
    pc1_down, _ = preprocess_point_cloud(pc1, voxel_size)
    pc2_down, _ = preprocess_point_cloud(pc2, voxel_size)
    # Perform ICP
    transformation1 = perform_icp(pc1_down, pc0_down, voxel_size * 2)
    transformation2 = perform_icp(pc2_down, pc0_down, voxel_size * 2)



    # Optionally visualize the result
    pc1.transform(transformation1)
    pc2.transform(transformation2)
    # store the transformed source point cloud
    pc0 += pc1
    pc0 += pc2
    o3d.visualization.draw_geometries([pc0], window_name="Point Clouds")
    path = "./build/icp_result.ply"
    o3d.io.write_point_cloud(path, pc0, write_ascii=True)
    print("Transformed point cloud saved to ", path)
