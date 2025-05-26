import sympy
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, Point
import random
import threading
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED
import multiprocessing
import queue

class Generator:
    def __init__(
        self,
        boundary_size: list[float], # [width, height], if width<0 or height<0, then it will be set to random
        view_point_num: int,  # number of obstacles, if <0, then it will be set to random
        sample_rate: float = 0.01, # sample rate of config space
        generate_scene_num: int = 1000, # number of scenes to generate
    ):
        self.boundary_size: list[float] = boundary_size
        self.view_point_num: int = view_point_num
        self.sample_rate: float = sample_rate
        self.sample_num: int =  int(1.0 / sample_rate)**2
        self.sqrt_sample_num: int = int(1.0 / sample_rate)
        self.generate_scene_num: int = generate_scene_num
        
        self.res_buffer = []
        self.buffer_lock = threading.Lock()
        self.threads = queue.Queue()
        
    def generate(self, thread_num:int = 8) -> list[
        tuple[
            list[list[float]], # list of triangles, each triangle is a list of points
            list[list[float]], # list of test points
            list[bool] # list of test points, True if in the config space, False if not
        ]
    ]:
        res = []
        def _print_exception(future):
            if future.exception() is not None:
                print(f"Thread {future} generated an exception: {future.exception()}")
        q = multiprocessing.Manager().Queue(-1)
        with ProcessPoolExecutor(max_workers=thread_num) as executor:
            all_futures = [executor.submit(
                generate_thread,
                i,
                self.boundary_size,
                self.view_point_num,
                self.sample_rate,
                self.sample_num,
                self.sqrt_sample_num,
                q
            ) for i in range(self.generate_scene_num)]
            for future in all_futures:
                future.add_done_callback(_print_exception)
            wait(all_futures, return_when=ALL_COMPLETED)
        while not q.empty():
            res.append(q.get())
        return res
    
def generate_thread(
        id:int,
        boundary_size: list[float],
        view_point_num: int,
        sample_rate: float,
        sample_num: int,
        sqrt_sample_num: int,
        q: queue.Queue
    ):
    points: list[list[float]] = [
            [random.uniform(-0.5, 0.5) * boundary_size[0], random.uniform(-0.5, 0.5) * boundary_size[1]]
            for _ in range(view_point_num)
        ]
    pts = np.array(points)
    tri = Delaunay(points)
    # select random triangles
    # @Output
    selected_triangles = pts[tri.simplices[random.sample(range(len(tri.simplices)), k=int(len(tri.simplices) / 2))]]
    selected_triangles_polygon = [Polygon(triangle) for triangle in selected_triangles]
    obstacles = MultiPolygon(selected_triangles_polygon)
    # @Output
    test_points = [
        [i % sqrt_sample_num * sample_rate * boundary_size[0] - 0.5 * boundary_size[0], 
            i // sqrt_sample_num * sample_rate * boundary_size[1] - 0.5 * boundary_size[1]]
        for i in range(sample_num)
    ]
    # @Output
    test_rst_temp = [
        obstacles.contains(Point(test_point[0], test_point[1])) for test_point in test_points
    ]
    test_points_free = [
        test_points[i] for i in range(len(test_points)) if test_rst_temp[i]
    ]
    test_points_obstacles = [
        test_points[i] for i in range(len(test_points)) if not test_rst_temp[i]
    ]
    half_num = 0
    if len(test_points_free) > len(test_points_obstacles):
        half_num = len(test_points_obstacles)
        test_points_free = test_points_free[random.sample(range(len(test_points_free)), k=half_num)]
    elif len(test_points_free) < len(test_points_obstacles):
        half_num = len(test_points_free)
        test_points_obstacles = test_points_obstacles[random.sample(range(len(test_points_obstacles)), k=half_num)]
    else:
        half_num = len(test_points_free)
    
    test_points = []
    test_points.extend(test_points_free)
    test_points.extend(test_points_obstacles)
    test_rst = []
    test_rst.extend([True] * half_num)
    test_rst.extend([False] * half_num)
    
    # feature of single triangle
    # C_sx, C_sy, S_cx, S_cy, V_0x, V_1x, V_2x, V_0y, V_1y, V_2y, S_vx, S_vy
    # C_sx, C_sy: center of triangle (scaled, relative to origin)
    # S_cx, S_cy: scalar according to scene
    # V_0x, V_0y, V_1x, V_1y, V_2x, V_2y: vertices of triangle (scaled, relative to center)
    # S_vx, S_vy: scalar according to boundary of triangle
    # feature of single test point
    # P_x, P_y, S_x, S_y
    # P_x, P_y: test point (scaled, relative to origin)
    # S_x, S_y: scalar according to scene
    
    feature_triangles = []
    feature_test_points = []
    
    boundary = obstacles.bounds
    origin = [(boundary[0] + boundary[2]) / 2, (boundary[1] + boundary[3]) / 2]
    S_cx = (boundary[2] - boundary[0]) / 2
    S_cy = (boundary[3] - boundary[1]) / 2
    for polygon in selected_triangles_polygon:
        vertices = polygon.exterior.coords
        polygon_bounds = polygon.bounds
        center = [(polygon_bounds[0] + polygon_bounds[2]) / 2, (polygon_bounds[1] + polygon_bounds[3]) / 2]
        S_vx = (polygon_bounds[2] - polygon_bounds[0]) / 2
        S_vy = (polygon_bounds[3] - polygon_bounds[1]) / 2
        feature_triangles.append([
            (center[0] - origin[0]) / S_cx, (center[1] - origin[1]) / S_cy,
            S_cx, S_cy,
            *[(vertices[i][0] - center[0]) / S_vx for i in range(3)],
            *[(vertices[i][1] - center[1]) / S_vy for i in range(3)],
            S_vx, S_vy
        ])
    for test_point in test_points:
        feature_test_points.append([
            (test_point[0] - origin[0]) / S_cx, (test_point[1] - origin[1]) / S_cy,
            S_cx, S_cy
        ])    
    
    res = (
        feature_triangles,
        feature_test_points,
        test_rst
    )
    q.put(res)
    
    
            
if __name__ == "__main__":
    generator = Generator(
        boundary_size=[100, 100],
        view_point_num=100,
        sample_rate=0.005,
        generate_scene_num=1
    )
    res = generator.generate()
    print(len(res))