import torch
import faiss


def cluster_traces_kmeans(vectors: torch.Tensor, n_clusters: int = 3):
    kmeans = faiss.Kmeans(
        vectors.shape[1], 
        min(n_clusters, vectors.shape[0]), 
        niter=50, 
        verbose=False,
        min_points_per_centroid=1,
        max_points_per_centroid=10000000,
    )
    kmeans.train(vectors.cpu().numpy())
    distances, cluster_ids_x_np = kmeans.index.search(vectors.cpu().numpy(), 1)
    distances = distances.squeeze(1)
    cluster_ids_x_np = cluster_ids_x_np.squeeze(1)

    cluster_ids_x = torch.from_numpy(cluster_ids_x_np).to(vectors.device)

    sampled_ids = cluster_ids_x.new_zeros(cluster_ids_x.size(0)).to(vectors.device)
    for cluster_id in range(min(n_clusters, vectors.shape[0])):
        cluster_idx = (cluster_ids_x == cluster_id).nonzero()
        if cluster_idx.size(0) > 0:
            # Get distances for points in this cluster
            cluster_distances = distances[cluster_idx]
            # Find the point with minimum distance to the cluster center
            min_distance_idx = cluster_distances.argmin()
            # Mark the point with minimum distance for sampling
            sampled_ids[cluster_idx[min_distance_idx]] = 1

    return sampled_ids

