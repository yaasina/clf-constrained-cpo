import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class KinematicBicycleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_deviation=5.0):
        super().__init__()
        
        # Vehicle parameters
        self.lw = 2.7  # Wheelbase length
        self.lfo = 0.9
        self.lro = 0.9
        self.l = self.lw + self.lfo + self.lro  # Total vehicle length
        self.dt = 0.05  # Time step
        self.max_steer = 0.52 # Max steering angle
        self.max_accel = 4.5  # Max acceleration (m/s^2)
        self.max_speed = 40.0  # Max speed (m/s)

        self.max_deviation = max_deviation
        self.max_steps = 1000  # Max steps per episode
        self.current_step = 0
        
        # State: [x, y, theta, velocity]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, 0.0]),
            high=np.array([np.inf, np.inf, np.pi, self.max_speed]),
            dtype=np.float32
        )
        
        # Action: [acceleration, steering angle]
        self.action_space = spaces.Box(
            low=np.array([-self.max_accel, -self.max_steer]),
            high=np.array([self.max_accel, self.max_steer]),
            dtype=np.float32
        )
        
        self.state = None
        self.render_mode = render_mode
        
        # Visualization setup
        self.fig, self.ax = None, None
        self.xd, self.yd, self.thetad, self.vd = self.calculate_trajectory() 
        self.trajectory = []  # Store trajectory for rendering 

    def calculate_trajectory(self):
        # function is sin with period of 50, episode will end at the end of one period
        # dt of 0.05 and max_steps of 1000 means 50 seconds of simulation time
        # max velocity of 40 means the vehicle can travel a max of 2000 meters in 50 seconds
        # estimate arc length of sin by boxing it in rectangles lol
        # amplitude of 40 and period of 100 (40 + 100 + 80 + 40 = 260)
        # Vehicle has ample time to complete the trajectory 260 < 2000
        x_length = 100
        target_traj = lambda x: 40 * np.sin(2 * np.pi * x / x_length)  # Example trajectory function
        x = np.linspace(0, x_length, self.max_steps)
        y = target_traj(x)

        dx = [x[i+1] - x[i] for i in range(len(x) - 1)]
        dx.append(x[-1] - x[-2])

        dy = [y[i+1] - y[i] for i in range(len(y) - 1)]
        y_p1 = target_traj(x[-1] + dx[-1])
        dy.append(y_p1 - y[-1])

        theta = np.array([np.arctan2(dy[i], dx[i]) for i in range(len(dx))])

        vel = np.array([np.sqrt(dx[i]**2 + dy[i]**2) / self.dt for i in range(len(dx))])

        return x, y, theta, vel


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([self.xd[0], self.yd[0], self.thetad[0], self.vd[0]], dtype=np.float32)  # Start at origin

        x_error = self.xd[self.current_step] - self.state[0]
        y_error = self.yd[self.current_step] - self.state[1]
        theta_error = self.thetad[self.current_step] - self.state[2]
        v_error = self.vd[self.current_step] - self.state[3]
        error = np.array([x_error, y_error, theta_error, v_error], dtype=np.float32)

        self.trajectory = [self.state[:2]]

        info = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0,
            'v': 0.0,
        }
        return error, {}

    def step(self, action):
        accel, delta = np.clip(action, self.action_space.low, self.action_space.high)
        action = np.array([accel, delta], dtype=np.float32)
        x, y, theta, v = self.state
        
        # Update state using kinematic bicycle model
        dx, dy, dtheta, dv = self.compute_derivative(self.state, action)
        x += dx * self.dt
        y += dy * self.dt
        theta += dtheta * self.dt
        v += dv * self.dt
        
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # Normalize angle

        self.state = np.array([x, y, theta, v], dtype=np.float32)
        self.trajectory.append([x, y, theta])

        self.current_step += 1
        
        # # Compute reward (tracking error)
        # y_target = self.target_traj(x)
        # tracking_error = abs(y - y_target)
        # rew_alive = 1e-1
        # reward = -tracking_error*1e-1 + rew_alive  # Penalize deviation from trajectory

        y_error = self.yd[self.current_step] - y
        x_error = self.xd[self.current_step] - x
        theta_error = self.thetad[self.current_step] - theta
        v_error = self.vd[self.current_step] - v

        pos_error = np.sqrt(x_error**2 + y_error**2)

        reward_alive = 0.5
        reward = - (pos_error * 1e-1 + 1e-4 * np.sum(np.square(action))) + reward_alive

        # Termination condition
        terminated = np.linalg.norm([y_error, x_error]) > self.max_deviation
        truncated = self.current_step >= self.max_steps - 1

        # if terminated:
        #     reward = -1000.0  # Large penalty for exceeding max deviation

        info = {
            'x': x,
            'y': y,
            'theta': theta,
            'v': v,
        }
        
        error = np.array([x_error, y_error, theta_error, v_error], dtype=np.float32)
        
        return error, reward, terminated, truncated, info
    
    def compute_derivative(self, state, action):
        x, y, theta, v = state
        accel, delta = action
        
        beta = np.arctan((self.l/2 - self.lro)/self.lw * np.tan(delta))
        # Compute derivatives using kinematic bicycle model
        dx = v * np.cos(theta + beta)
        dy = v * np.sin(theta + beta)
        dtheta = (v / self.lw) * np.sin(delta)
        dv = accel
        
        return np.array([dx, dy, dtheta, dv], dtype=np.float32)


    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-10, 100)
            self.ax.set_ylim(-50, 50)
            self.ax.set_xlabel("X position")
            self.ax.set_ylabel("Y position")
            self.ax.set_title("Kinematic Bicycle Model Trajectory")
            
        self.ax.clear()
        
        # Plot trajectory
        self.ax.plot(*zip(*[(t[0], t[1]) for t in self.trajectory]), marker="o", markersize=3, linestyle="-", label="Vehicle Path")
        x_vals = np.linspace(-10, 50, 100)
        y_vals = [self.target_traj(x) for x in x_vals]
        self.ax.plot(x_vals, y_vals, "r--", label="Target Trajectory")
        
        # Draw vehicle representation
        if self.trajectory:
            x, y, theta = self.trajectory[-1]
            car_length = self.l * 2.0
            car_width = self.l * 0.75
            
            rear_x = x - (car_length / 2) * np.cos(theta)
            rear_y = y - (car_length / 2) * np.sin(theta)
            front_x = x + (car_length / 2) * np.cos(theta)
            front_y = y + (car_length / 2) * np.sin(theta)
            
            car_rect = plt.Rectangle((rear_x - car_width / 2, rear_y - car_width / 2), car_length, car_width, angle=np.degrees(theta), color="blue", alpha=0.6)
            self.ax.add_patch(car_rect)
        
        self.ax.legend()
        plt.pause(0.01)

    def close(self):
        if self.fig:
            plt.close(self.fig)

