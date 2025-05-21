from sympy import Point2D, Polygon, core, Point3D
from matplotlib import pyplot as plt
import numpy as np

class Scene:
    def __init__(
        self,
        boundary: Polygon,
        obstacles: list,
        padding_rate: float = 0.1
        ):
        self.raw_boundary = boundary
        self.raw_obstacles = obstacles
        self.constructed_obstacles = self.raw_obstacles[:]        
        self.decomposed_obstacles:list = []
        self.decomposed_obstacles_feature = []
        self.padding_rate = padding_rate
        
        self.center = self.raw_boundary.centroid
        self.max_bounds = self.raw_boundary.bounds
        
        self.scale = (
            self.max_bounds[2] - self.max_bounds[0],
            self.max_bounds[3] - self.max_bounds[1]
        )
        
        for i in range(len(self.raw_boundary.vertices)):
            p1 = self.raw_boundary.vertices[i]
            p2 = self.raw_boundary.vertices[(i + 1) % len(self.raw_boundary.vertices)]
            direction = Point2D(-(p2.y - p1.y), p2.x - p1.x).unit
            ci = p1 - self.center
            if ci.dot(direction) < 0:
                direction = -direction
            self.constructed_obstacles.append(
                Polygon(
                    p1, p2,
                    (p1 + p2) / 2 + direction * self.padding_rate * self.scale[0]
                )
            )
            
        for obs in self.constructed_obstacles:
            if len(obs.vertices) == 3:
                self.decomposed_obstacles.append(obs)
            elif len(obs.vertices) > 3:
                c = obs.centroid
                for i in range(len(obs.vertices)):
                    p1 = obs.vertices[i]
                    p2 = obs.vertices[(i + 1) % len(obs.vertices)]
                    self.decomposed_obstacles.append(
                        Polygon(p1, p2, c)
                    )
                    
        for obs in self.decomposed_obstacles:
            cx0 = (obs.centroid.x - self.center.x) / self.scale[0]
            cy0 = (obs.centroid.y - self.center.y) / self.scale[1]
            sx = self.scale[0]
            sy = self.scale[1]
            b0 = obs.bounds
            svx = b0[2] - b0[0]
            svy = b0[3] - b0[1]
            xs0 = (obs.vertices[0].x - obs.centroid.x) / svx
            xs1 = (obs.vertices[1].x - obs.centroid.x) / svx
            xs2 = (obs.vertices[2].x - obs.centroid.x) / svx
            ys0 = (obs.vertices[0].y - obs.centroid.y) / svy
            ys1 = (obs.vertices[1].y - obs.centroid.y) / svy
            ys2 = (obs.vertices[2].y - obs.centroid.y) / svy
            self.decomposed_obstacles_feature.append(
                [sx, sy, svx, svy, cx0, cy0, xs0, xs1, xs2, ys0, ys1, ys2]
            )
    
    def plot(self, ax:plt.Axes):
        # for features in self.decomposed_obstacles_feature:
        #     sx, sy, svx, svy, cx0, cy0, xs0, xs1, xs2, ys0, ys1, ys2 = features
        #     plist = [
        #             (cx0 * sx + self.center.x + xs0 * svx, cy0 * sy + self.center.y + ys0 * svy),
        #             (cx0 * sx + self.center.x + xs1 * svx, cy0 * sy + self.center.y + ys1 * svy),
        #             (cx0 * sx + self.center.x + xs2 * svx, cy0 * sy + self.center.y + ys2 * svy),
        #         ]
        #     print(plist)
        #     ax.fill(
        #         [p[0] for p in plist],
        #         [p[1] for p in plist],
        #         alpha=0.5,
        #         edgecolor='black',
        #         facecolor='blue'
        #     )
        for obs in self.constructed_obstacles:
            x = [p.x for p in obs.vertices]
            y = [p.y for p in obs.vertices]
            ax.fill(x, y, alpha=0.5, edgecolor='black', facecolor='blue')
            
    def get_features(self):
        return self.decomposed_obstacles_feature
    
    def check_collision(self, state):
        if isinstance(state, Point3D):
            state = Point2D(state.x, state.y)
            if not self.raw_boundary.encloses_point(state):
                return True
            for obs in self.raw_obstacles:
                if obs.encloses_point(state):
                    return True
            return False
    
class Robot:
    def __init__(
        self,
        shape: Polygon,
        limits: list,
    ):
        self.limits = limits
        self.state = None
        self.state_feature = None
        self.last_action = None
        self.shape = shape
        self.shape_feature = []
        
        if self.shape is not None:
            boundary = self.shape.bounds
            svx = boundary[2] - boundary[0]
            svy = boundary[3] - boundary[1]
            c0 = self.shape.centroid
            for i in range(len(self.shape.vertices)):
                p1 = self.shape.vertices[i]
                p2 = self.shape.vertices[(i + 1) % len(self.shape.vertices)]
                xs0 = (p1.x - c0.x) / svx
                xs1 = (p2.x - c0.x) / svx
                ys0 = (p1.y - c0.y) / svy
                ys1 = (p2.y - c0.y) / svy
                self.shape_feature.append(
                    [svx, svy, xs0, xs1, ys0, ys1]
                )
        
    def reset(self, state:Point3D):
        self.state = state
        self.last_action = None
        
    def get_shape_features(self):
        return self.shape_feature
    
    def step(self, action, sx, sy, cx, cy):
        # last_state = self.state
        dt = self.limits[-1]
        v = action[0] * self.limits[0]
        w = action[1] * self.limits[1]
        if w == 0:
            dx = v * np.cos(float(self.state.z)) * dt
            dy = v * np.sin(float(self.state.z)) * dt
            dtheta = 0
        else:
            R = v / w
            L = np.sqrt(2*R**2*(1-np.cos(w*dt)))
            dtheta = w * dt
            dx = L * np.cos(float(self.state.z) + dtheta / 2)
            dy = L * np.sin(float(self.state.z) + dtheta / 2)
        self.state = Point3D(
            self.state.x + dx,
            self.state.y + dy,
            (self.state.z + dtheta + np.pi) % (2 * np.pi) - np.pi
        )
        self.last_action = action
        self.state_feature = [float((self.state.x - cx) / sx), 
               float((self.state.y - cy) / sy),
               float(self.state.z / np.pi)]
        return self.state_feature
    
    def get_state_features(self):
        return self.state_feature
    
    def plot(self, ax:plt.Axes):
        if self.state is not None:
            ax.plot(self.state.x, self.state.y, 'o', color='green', markersize=1)
            ax.arrow(self.state.x, self.state.y, 0.1 * np.cos(float(self.state.z)), 0.1 * np.sin(float(self.state.z)),
                     head_width=0.1, head_length=0.2, fc='red', ec='red')
        if self.shape is not None:
            #TODO:
            pass
    
class Environment:
    def __init__(
        self,
        boundary,
        obstacles,
        robot_shape,
        robot_limit,
        start:Point3D,
        goal:Point3D,
    ):
        self.scene = Scene(boundary, obstacles)
        self.robot = Robot(robot_shape, robot_limit)
        self.start = start
        self.goal = goal
        self.start_feature = [
            (start.x - self.scene.center.x) / self.scene.scale[0],
            (start.y - self.scene.center.y) / self.scene.scale[1],
            ((start.z + np.pi) % (2 * np.pi) - np.pi) / np.pi
        ]
        self.goal_feature = [
            (goal.x - self.scene.center.x) / self.scene.scale[0],
            (goal.y - self.scene.center.y) / self.scene.scale[1],
            ((goal.z + np.pi) % (2 * np.pi) - np.pi) / np.pi
        ]
        self.steps = 0
        
        self.robot.reset(self.start)
        
    def reset(self):
        self.robot.reset(self.start)
        self.robot.step([0, 0], self.scene.scale[0], self.scene.scale[1], self.scene.center.x, self.scene.center.y)
        self.steps = 0
        
    def plot(self, ax:plt.Axes):
        self.scene.plot(ax)
        self.robot.plot(ax)
        ax.plot(self.start.x, self.start.y, 'o', color='green', markersize=1)
        ax.arrow(self.start.x, self.start.y, 0.1 * np.cos(float(self.start.z)), 0.1 * np.sin(float(self.start.z)),
                    head_width=0.1, head_length=0.2, fc='red', ec='red')
        ax.plot(self.goal.x, self.goal.y, 'o', color='green', markersize=1)
        ax.arrow(self.goal.x, self.goal.y, 0.1 * np.cos(float(self.goal.z)), 0.1 * np.sin(float(self.goal.z)),
                    head_width=0.1, head_length=0.2, fc='red', ec='red')
        
    def step(self, action):
        """
        Takes an action and updates the robot's state.
        Returns the new state, reward, done flag, and additional info.
        """
        sx = self.scene.scale[0]
        sy = self.scene.scale[1]
        cx = self.scene.center.x
        cy = self.scene.center.y
        
        last_state_np = np.array(self.robot.state_feature[0:2])
        new_state = self.robot.step(action, sx, sy, cx, cy)
        
        collision = self.scene.check_collision(self.robot.state)
        
        if collision:
            return new_state, -1000, True, None
        
        if self.robot.state.distance(self.goal) < 0.1:
            return new_state, 1000, True, None
        
        self.steps += 1
        if self.steps > 200:
            return new_state, -1000, True, None
        
        new_state_np = np.array(new_state[0:2])
        goal_feature_np = np.array([float(self.goal_feature[0]), float(self.goal_feature[1])])
        err_last = np.linalg.norm(last_state_np - goal_feature_np)
        err_new = np.linalg.norm(new_state_np - goal_feature_np)
        return new_state, (err_last - err_new) * 10, False, None
    
            
if __name__ == "__main__":
    env = Environment(
        Polygon(Point2D(0,0), Point2D(0, 10), Point2D(10, 10) ,Point2D(10, 0)),
        [
            Polygon(Point2D(1,1), Point2D(1, 2), Point2D(2, 2) ,Point2D(2, 1))
        ],
        None,
        [0.5, 0.5, 0.2],
        Point3D(0.5, 0.5, 0),
        Point3D(9.5, 9.5, 0)
    )
    
    fig = plt.figure()
    ax = plt.gca()
        
    for i in range(100):
        plt.cla()
        print(env.step([0.5, 0.5]))
        env.plot(ax)
        plt.pause(0.1)
            
            
            
            