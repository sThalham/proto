#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>
//#include <pcl/filters/bilateral.h>
#include <pcl/filters/voxel_grid.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
//#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;

bool
enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
    return (true);
  else
    return (false);
}

bool
enforceCurvatureOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  //if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
    //return (true);
  std::cerr << fabs(point_a_normal.dot(point_b_normal)) << std::endl;
  if (fabs (point_a_normal.dot (point_b_normal)) < 0.25)
    return (true);
  return (false);
}

bool
customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  if (squared_distance < 100)
  {
    //if (fabs (point_a.intensity - point_b.intensity) < 8.0f)
      //return (true);
    if (fabs (point_a_normal.dot (point_b_normal)) > 0.98)
      return (true);
  }
  else
  //{
  //  if (fabs (point_a.intensity - point_b.intensity) < 3.0f)
  //    return (true);
  //}
  return (false);
}

int
main (int argc, char** argv)
{
  // Data containers used
  pcl::PointCloud<PointTypeIO>::Ptr cloud_in (new pcl::PointCloud<PointTypeIO>), cloud_out (new pcl::PointCloud<PointTypeIO>);
  pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>);
  pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
  pcl::search::KdTree<PointTypeIO>::Ptr search_tree (new pcl::search::KdTree<PointTypeIO>);
  pcl::console::TicToc tt;

  // Load the input point cloud
  //std::cerr << "Loading...\n", tt.tic ();
  pcl::io::loadPCDFile (argv[1], *cloud_in);
  //std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_in->points.size () << " points\n";

  // Downsample the cloud using a Voxel Grid class
  //std::cerr << "Downsampling...\n", tt.tic ();
  //pcl::VoxelGrid<PointTypeIO> vg;
  //vg.setInputCloud (cloud_in);
  //vg.setLeafSize (5.0, 5.0, 5.0);
  //vg.setDownsampleAllData (true);
  //vg.filter (*cloud_out);
  //std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_out->points.size () << " points\n";

  // Set up a Normal Estimation class and merge data in cloud_with_normals
  //std::cerr << "Computing normals...\n", tt.tic ();
  pcl::copyPointCloud (*cloud_in, *cloud_with_normals);
  pcl::NormalEstimationOMP<PointTypeIO, PointTypeFull> ne;
  ne.setInputCloud (cloud_in);
  ne.setSearchMethod (search_tree);
  //ne.setRadiusSearch (5.0); // deprecated
  ne.setKSearch(9); // looking good - Duke Nukem
  ne.compute (*cloud_with_normals);
  
  //pcl::IntegralImageNormalEstimation<PointTypeIO, PointTypeFull> ne;
  //ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
  //ne.setMaxDepthChangeFactor(2.0f);
  //ne.setNormalSmoothingSize(10.0f);
  //ne.setInputCloud(cloud_in);
  //ne.compute(*cloud_with_normals);

  //std::cerr << ">> Done: " << tt.toc () << " ms\n";

  // Set up a Conditional Euclidean Clustering class
  //std::cerr << "Segmenting to clusters...\n", tt.tic ();
  pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
  cec.setInputCloud (cloud_with_normals);
  cec.setConditionFunction (&customRegionGrowing);
  cec.setClusterTolerance (25.0);   //minor influence
  //cec.setMinClusterSize (cloud_with_normals->points.size () / 1000);
  cec.setMinClusterSize(10);
  //cec.setMaxClusterSize (cloud_with_normals->points.size () / 5);
  cec.setMaxClusterSize(cloud_with_normals->points.size());
  cec.segment (*clusters);
  cec.getRemovedClusters (small_clusters, large_clusters);
  //std::cerr << ">> Done: " << tt.toc () << " ms\n";

  // Using the intensity channel for lazy visualization of the output
  double intenSmall = 127.0;
  double intenBig = 128.0;
  for (int i = 0; i < small_clusters->size (); ++i)
    for (int j = 0; j < (*small_clusters)[i].indices.size (); ++j)
      cloud_in->points[(*small_clusters)[i].indices[j]].intensity = intenSmall;
      intenSmall -= 1.0;
  for (int i = 0; i < large_clusters->size (); ++i)
    for (int j = 0; j < (*large_clusters)[i].indices.size (); ++j)
      cloud_in->points[(*large_clusters)[i].indices[j]].intensity = intenBig;
      intenBig += 1.0;
  //std::cerr << "clusters: " << clusters->size() << std::endl;
  for (int i = 0; i < clusters->size (); ++i)
  {
    int label = rand ();
    //std::cerr << "label: " << label << std::endl;
    for (int j = 0; j < (*clusters)[i].indices.size (); ++j)
      cloud_in->points[(*clusters)[i].indices[j]].intensity = label;
  }

  //pcl::PointCloud<PointTypeIO>::Ptr cloud_upscaled (new pcl::PointCloud<PointTypeIO>);

  //pcl::BilateralFilter<pcl::PointXYZI>::Ptr fbFilter;
  //fbFilter.setInputCloud(cloud_out);
  //fbFilter.setHalfSize();
  //fbFilter.setStdDev();
  //fbFilter.setSearchMethod(search_tree);
  //fbFilter.applyFilter(cloud_upscaled)

  // Save the output point cloud
  //std::cerr << "Saving...\n", tt.tic ();
  pcl::io::savePCDFile (argv[2], *cloud_in);
  //std::cerr << ">> Done: " << tt.toc () << " ms\n";

  return (0);
}
