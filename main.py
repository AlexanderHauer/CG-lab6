import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# --- Task 1 ---
points_task1 = np.array([[3, -5], [-6, 6], [6, -4], [5, -5], [9, 10]])

vor = Voronoi(points_task1)

del_tri = Delaunay(points_task1)

# Plot
plt.subplot(1, 2, 1)
plt.title("Voronoi Diagram")
plt.scatter(points_task1[:, 0], points_task1[:, 1], color='red', label="Points")
plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o', color='blue', label="Vertices")

for point_idx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')
    elif np.any(simplex < 0):
        i = simplex[simplex >= 0][0]
        t = vor.points[point_idx[1]] - vor.points[point_idx[0]]
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])

        midpoint = vor.points[point_idx].mean(axis=0)
        far_point = vor.vertices[i] + n * 100

        plt.plot([vor.vertices[i, 0], far_point[0]],
                 [vor.vertices[i, 1], far_point[1]], 'k--')

plt.legend()
plt.grid()


plt.subplot(1, 2, 2)
plt.title("Delaunay Triangulation")
plt.triplot(points_task1[:, 0], points_task1[:, 1], del_tri.simplices)
plt.scatter(points_task1[:, 0], points_task1[:, 1], color='red')
plt.grid()
plt.show()

# --- Task 2 ---
def find_two_points(points):
    # Heuristic: Add points to create exactly 4 infinite edges
    # Two points are added far from the others to ensure the desired Voronoi diagram structure.
    points = np.vstack([points, [10, -10], [-10, 10]])
    vor = Voronoi(points)
    return points, vor

points_task2 = np.array([[5, -1], [7, -1], [9, -1], [7, -3], [11, -1], [-9, 3]])
new_points, vor_task2 = find_two_points(points_task2)

plt.figure(figsize=(6, 6))
plt.title("Voronoi Diagram with Added Points")
plt.scatter(new_points[:, 0], new_points[:, 1], color='red')
for region in vor_task2.regions:
    if not -1 in region and len(region) > 0:
        polygon = [vor_task2.vertices[i] for i in region]
        plt.fill(*zip(*polygon), alpha=0.4)
plt.plot(vor_task2.vertices[:, 0], vor_task2.vertices[:, 1], 'o', color='blue')
plt.grid()
plt.show()

# --- Task 3 ---
points_task3 = np.array([[-1, 6], [-1, -1], [4, 7], [6, 7], [1, -1], [-5, 3], [-2, 3]])
def minimal_spanning_tree(points, lambdas):
    # For each lambda value, the point Q is adjusted and added to the set.
    # The distance matrix of all points is computed, and the MST algorithm determines the total edge length.
    results = []
    for lam in lambdas:
        q = [2 - lam, 3]
        all_points = np.vstack([points, q])
        dist_matrix = distance_matrix(all_points, all_points)
        mst = minimum_spanning_tree(dist_matrix)
        results.append((lam, mst.sum()))
    return results

lambdas = np.linspace(-10, 10, 100)
mst_results = minimal_spanning_tree(points_task3, lambdas)
best_lambda = min(mst_results, key=lambda x: x[1])

print(f"Best Lambda: {best_lambda[0]}, Minimal MST Length: {best_lambda[1]}")
print("\n")
# --- Task 4: Voronoi Diagram Half-Line Count ---
def count_half_lines(vor):
    return sum(1 for ridge in vor.ridge_vertices if -1 in ridge)

points_task4 = np.vstack([
    np.array([[1 - i, i - 1] for i in range(6)]),
    np.array([[i, -i] for i in range(6)]),
    np.array([[0, i] for i in range(6)])
])
vor_task4 = Voronoi(points_task4)

half_line_count = count_half_lines(vor_task4)
print(f"Number of Half-Lines in Voronoi Diagram: {half_line_count}")
print("\n")
# Plot Voronoi Diagram
plt.figure(figsize=(8, 8))
plt.title("Voronoi Diagram for Task 4")
plt.scatter(points_task4[:, 0], points_task4[:, 1], color='red')
for region in vor_task4.regions:
    if not -1 in region and len(region) > 0:
        polygon = [vor_task4.vertices[i] for i in region]
        plt.fill(*zip(*polygon), alpha=0.4)
plt.plot(vor_task4.vertices[:, 0], vor_task4.vertices[:, 1], 'o', color='blue')
plt.grid()
plt.show()

# --- Task 5 ---
def example_two_sets():
    # Set 1: A triangle
    points_set1 = np.array([[0, 0], [1, 0], [0, 1]])
    delaunay1 = Delaunay(points_set1)

    # Set 2: A square
    points_set2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    delaunay2 = Delaunay(points_set2)

    vor1 = Voronoi(points_set1)
    vor2 = Voronoi(points_set2)

    print("Set 1: 3 points, Triangles =", len(delaunay1.simplices), "Edges =", len(delaunay1.convex_hull))
    print("Set 2: 4 points, Triangles =", len(delaunay2.simplices), "Edges =", len(delaunay2.convex_hull))
    print("Half-line edges in Voronoi Diagram for Set 1:", count_half_lines(vor1))
    print("Half-line edges in Voronoi Diagram for Set 2:", count_half_lines(vor2))
    print("\n")
example_two_sets()

# --- Task 6 ---
def triangles_and_edges(lambda_values):
    points = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [0, -2]])
    results = []
    for lam in lambda_values:
        m = np.array([[0, lam]])
        all_points = np.vstack([points, m])
        delaunay = Delaunay(all_points)
        num_triangles = len(delaunay.simplices)
        num_edges = len(delaunay.convex_hull)
        results.append((lam, num_triangles, num_edges))
    return results

lambda_values = np.linspace(-10, 10, 50)
task6_results = triangles_and_edges(lambda_values)
for lam, num_triangles, num_edges in task6_results:
    print(f"Lambda: {lam}, Triangles: {num_triangles}, Edges: {num_edges}")
