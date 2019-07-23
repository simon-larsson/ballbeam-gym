""" 
Ball & Beam Simulation

Simple ball and beam simulation built to interface easily with OpenAI's
gym environments.

"""

from math import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

class BallBeam():
    """ Ball & Beam

    System dynamics
    ---------------
    dx/dt = v(t)
    dv/dt = -m*g*sin(theta(t))/((I + 1)*m)

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    init_velocity : initial speed of ball, float (units/s)

    rot_speed : speed the angle can be changed with (rad/s)
    """

    def __init__(self, timestep=0.05, beam_length=1.0, max_angle=0.2, init_velocity=0.0):
        self.dt = timestep                 # time step
        self.g = 9.82                       # gravity
        self.r = 0.05                       # ball radius
        self.L = beam_length                # beam length
        self.I = 2/5*self.r**2              # solid ball inertia (omits mass)
        self.init_velocity = init_velocity  # initial velocity
        self.max_angle = max_angle          # max beam angle (rad)
        self.reset()
        self.rendered = False

    def reset(self):
        radius = self.L/2
        self.t = 0.0                            # time
        self.x = 0.0                            # ball position x
        self.y = self.r                         # ball position y
        self.v = self.init_velocity             # ball velocity
        self.theta = 0.0                        # beam angle (rad)
        self.v_x = 0.0                          # velocity x component
        self.v_y = 0.0                          # velocity y component
        self.lim_x = [-cos(self.theta)*radius,  # geam limits x
                       cos(self.theta)*radius] 
        self.lim_y = [-sin(self.theta)*radius,  # beam limits y
                       sin(self.theta)*radius] 

    def update(self, angle):
        """ Update simulation with one time step

        Parameters
        ----------
        angle : angle the beam should be set to, float (rad)
        """
        radius = self.L/2

        # simulation could be improved further by using voltage as input and a
        # motor simulation deciding theta     
        self.theta = max(-self.max_angle, min(self.max_angle, angle))

        if self.on_beam:
            # dynamics on beam
            self.v += -self.g/(1 + self.I)*sin(self.theta)*self.dt
            self.v_x = self.v*cos(self.theta)
            self.v_y = self.v*sin(self.theta)
            self.x += self.v*self.dt
            self.y = self.r/cos(self.theta) + self.x*sin(self.theta)
        else:
            # free fall dynamics off beam
            self.v_y -= self.g*self.dt
            self.v = (self.v_x**2 + self.v_y**2)**0.5
            self.x += self.v_x*self.dt
            self.y += self.v_y*self.dt
        
        # edge of beam
        self.lim_x = [-cos(self.theta)*radius, cos(self.theta)*radius]
        self.lim_y = [-sin(self.theta)*radius, sin(self.theta)*radius]
        
        # update time
        self.t += self.dt

    def _init_render(self, setpoint):
        """ Initialize rendering """
        radius = self.L/2

        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        fig.canvas.set_window_title('Beam & Ball')
        ax.set(xlim = (-2*radius, 2*radius), ylim = (-self.L/2, self.L/2))
        
        # draw ball
        self.ball_plot = Circle((self.x, self.y), self.r)
        ax.add_patch(self.ball_plot)
        ax.patches[0].set_color('red')
        # draw beam
        ax.plot([-cos(self.theta)*radius, cos(self.theta)*radius], 
                [-sin(self.theta)*radius, sin(self.theta)*radius], lw=4, color='black')
        ax.plot(0.0, 0.0, '.', ms=20)

        if setpoint is not None:
            ax.add_patch(Polygon( \
                [[setpoint*cos(self.theta), -0.01*self.L + setpoint*sin(self.theta)],
                 [setpoint*cos(self.theta) - 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)],
                 [setpoint*cos(self.theta) + 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)]]))
            ax.patches[1].set_color('red')

        self.fig = fig
        self.ax = ax
        self.rendered = True

    def render(self, setpoint=None):
        """ Render simulation at it's current state

        Parameters
        ----------
        setpoint : optional marking of the ball setpoint, float (units)
        """
        if not self.rendered:
            self._init_render(setpoint)

        # update ball
        self.ball_plot.set_center((self.x, self.y))
        # update beam
        self.ax.lines[0].set(xdata=self.lim_x, ydata=self.lim_y)

        if setpoint is not None:
            self.ax.patches[1].set_xy( \
                [[setpoint*cos(self.theta), -0.01*self.L + setpoint*sin(self.theta)],
                 [setpoint*cos(self.theta) - 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)],
                 [setpoint*cos(self.theta) + 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)]])

        # update figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @property
    def on_beam(self):
        """ Check if ball is still on the beam """
        return self.lim_x[0] < self.x and self.lim_x[1] > self.x
