""" 
Ball & Beam Simulation

Simple ball and beam simulation built to interface easily with OpenAI's
gym environments.
"""

from math import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

import numpy as np
import time

class BallBeam():
    """ BallBeam

    Simple ball and beam simulation built to interface easily with OpenAI's
    gym environments.

    System dynamics
    ---------------
    dx/dt = v(t)
    dv/dt = -m*g*sin(theta(t))/((I + 1)*m)

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)
    """

    def __init__(self, timestep=0.05, beam_length=1.0, max_angle=0.2, init_velocity=0.0):
        self.dt = timestep                  # time step
        self.g = 9.82                       # gravity
        self.r = 0.05                       # ball radius
        self.L = beam_length                # beam length
        self.I = 2/5*self.r**2              # solid ball inertia (omits mass)
        self.init_velocity = init_velocity  # initial velocity
        self.max_angle = max_angle          # max beam angle (rad)
        self.reset()
        self.human_rendered = False
        self.machine_rendered = False

    def reset(self):
        radius = self.L/2                       # beam radius
        self.t = 0.0                            # time
        self.x = 0.0                            # ball position x
        self.y = self.r                         # ball position y
        self.v = self.init_velocity             # ball velocity
        self.theta = 0.0                        # beam angle (rad)
        self.dtheta = 0.0                       # beam angle change (rad)
        self.v_x = 0.0                          # velocity x component
        self.v_y = 0.0                          # velocity y component
        self.lim_x = [-cos(self.theta)*radius,  # geam limits x
                       cos(self.theta)*radius] 
        self.lim_y = [-sin(self.theta)*radius,  # beam limits y
                       sin(self.theta)*radius] 

    def update(self, angle):
        """ 
        Update simulation with one time step

        Parameters
        ----------
        angle : angle the beam should be set to, float (rad)
        """
        radius = self.L/2

        # simulation could be improved further by using voltage as input and a
        # motor simulation deciding theta     
        theta = max(-self.max_angle, min(self.max_angle, angle))
        
        # store angle change for angular velocity
        self.dtheta = theta - self.theta
        self.theta = theta 

        if self.on_beam:
            x = self.x
            y = self.y

            # dynamics on beam
            self.v += -self.g/(1 + self.I)*sin(self.theta)*self.dt            
            self.x += self.v*self.dt
            self.y = self.r/cos(self.theta) + self.x*sin(self.theta)

            # keep track of velocities for x and y
            self.v_x = (self.x - x)/self.dt
            self.v_y = (self.y - y)/self.dt
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

    def _init_render(self, setpoint, mode):
        """ Initialize rendering """
        radius = self.L/2

        if mode == 'human':
            self.human_rendered = True
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
        else:
            self.machine_rendered = True
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

            # avoid drawing plot but still initialize
            _ = fig.canvas.set_window_title('Beam & Ball')
            _ = ax.set(xlim = (-2*radius, 2*radius), ylim = (-self.L/2, self.L/2))
            
            # draw ball
            self.ball_plot = Circle((self.x, self.y), self.r)
            _ = ax.add_patch(self.ball_plot)
            _ = ax.patches[0].set_color('red')
            # draw beam
            _ = ax.plot([-cos(self.theta)*radius, cos(self.theta)*radius], 
                    [-sin(self.theta)*radius, sin(self.theta)*radius], lw=4, color='black')
            _ = ax.plot(0.0, 0.0, '.', ms=20)

            if setpoint is not None:
                _ = ax.add_patch(Polygon( \
                    [[setpoint*cos(self.theta), -0.01*self.L + setpoint*sin(self.theta)],
                    [setpoint*cos(self.theta) - 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)],
                    [setpoint*cos(self.theta) + 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)]]))
                _ = ax.patches[1].set_color('red')

            self.machine_fig = fig
            self.machine_ax = ax

    def render(self, setpoint=None, mode='human'):
        """ 
        Render simulation at it's current state

        Parameters
        ----------
        setpoint : optional marking of the ball setpoint, float (units)

        mode : rendering mode, str [human, machine]
        """
        if (not self.human_rendered and mode == 'human') or \
           (not self.machine_rendered and mode == 'machine'):
            self._init_render(setpoint, mode)

        if mode == 'human':
            # update ball
            self.ball_plot.set_center((self.x, self.y))
            # update beam
            self.ax.lines[0].set(xdata=self.lim_x, ydata=self.lim_y)

            # mark setpoint
            if setpoint is not None:
                self.ax.patches[1].set_xy( \
                    [[setpoint*cos(self.theta), -0.01*self.L + setpoint*sin(self.theta)],
                    [setpoint*cos(self.theta) - 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)],
                    [setpoint*cos(self.theta) + 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)]])

            # update figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        else:
            # update ball
            _ = self.ball_plot.set_center((self.x, self.y))
            # update beam
            _ = self.machine_ax.lines[0].set(xdata=self.lim_x, ydata=self.lim_y)

            # mark setpoint
            if setpoint is not None:
                _ = self.machine_ax.patches[1].set_xy( \
                    [[setpoint*cos(self.theta), -0.01*self.L + setpoint*sin(self.theta)],
                    [setpoint*cos(self.theta) - 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)],
                    [setpoint*cos(self.theta) + 0.015*self.L, -0.03*self.L + setpoint*sin(self.theta)]])

            # update figure
            _ = self.machine_fig.canvas.draw()
            _ = self.machine_fig.canvas.flush_events()


    @property
    def on_beam(self):
        """ 
        Check if ball is still on the beam 
        
        Returns
        -------
        on_beam : if ball is still located on the beam, bool
        
        """
        return self.lim_x[0] < self.x and self.lim_x[1] > self.x
