import numpy as np
import open3d as o3d

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
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
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
                loss
            )
    # result = o3d.pipelines.registration.registration_colored_icp(
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        0.05,
        np.eye(4),
        estimation,
    )
    print("ICP res:", result)
    return result.transformation

# Example usage
if __name__ == "__main__":
    # Load two point clouds
    source = o3d.io.read_point_cloud("./build/pc1.ply")
    target = o3d.io.read_point_cloud("./build/pc2.ply")

    o3d.visualization.draw_geometries([source, target], window_name="Source and Target Point Clouds")
    # Preprocess point clouds
    voxel_size = 0.01  # Adjust voxel size as needed
    source_down, _ = preprocess_point_cloud(source, voxel_size)
    target_down, _ = preprocess_point_cloud(target, voxel_size)
    # Perform ICP
    transformation = perform_icp(source_down, target_down)

    # Print the transformation matrix
    print("Transformation Matrix:")
    print(transformation)

    # Optionally visualize the result
    source.transform(transformation)
    o3d.visualization.draw_geometries([source, target])
    # store the transformed source point cloud
    o3d.io.write_point_cloud("./build/transformed_source.ply", source)
