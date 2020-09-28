import matlab.engine
import matplotlib.pyplot as plt

import random
import time
import numpy as np
import math

ans = []


class SimulinkPlant:
    def __init__(self, modelName='plant'):
        self.modelName = modelName  # The name of the Simulink Model (To be placed in the same directory as the Python Code)
        # Logging the variables
        self.q = 0
        self.q1 = 0
        self.w = 0
        self.w1 = 0

    def setControlAction(self, torque):
        # Helper Function to set value of control action
        self.eng.set_param('{}/u'.format(self.modelName), 'value', str(torque),
                           nargout=0)  # u블럭의 value 파라미터는 str(u) 이다. 제어값은 여기에 넣는다.
    # def getworkspace(self):
    #     return self.eng.workspace['out.q'], self.eng.workspace['out.q1'], self.eng.workspace['out.w'], self.eng.workspace['out.w1']
    def getHistory(self):
        # Helper Function to get Plant Output and Time History
        self.eng.eval('q = out.q(end);, w = out.w(end);, q1 = out.q1(end);, w1 = out.w1(end);', nargout = 0)
        #self.eng.eval("w = out.w(end)", nargout=0)
        #self.eng.eval("q1 = out.q1(end)", nargout=0)
        #self.eng.eval("w1 = out.w1(end)", nargout=0)
        return self.eng.workspace['q'], self.eng.workspace['q1'], self.eng.workspace['w'], self.eng.workspace['w1']

    def connectToMatlab(self):
        print("Starting matlab")
        self.eng = matlab.engine.start_matlab()

        print("Connected to Matlab")

        # Load the model
        self.eng.eval("model = '{}'".format(self.modelName), nargout=0)  # eval : 텍스트로 된 MATLAB 표현식 실행
        self.eng.eval("load_system(model)",
                      nargout=0)  # load_system : loads the model sys into memory without opening the model in the Simulink® Editor.

        simulation_time = self.eng.get_param(self.modelName, 'SimulationTime')
        print("simulation time: {0}".format(simulation_time))

        # Initialize Control Action to 0
        self.setControlAction(0)
        # print("Initialized Model")
        self.eng.set_param( self.modelName, 'SimMechanicsOpenEditorOnUpdate', 'off', nargout=0) #No Visualization
        # Start Simulation and then Instantly pause
        self.eng.set_param(self.modelName, 'SimulationCommand', 'start', 'SimulationCommand', 'pause',
                           nargout=0)
        # print(self.eng.get_param(self.modelName, 'SimulationStatus'))
        self.q, self.q1, self.w, self.w1 = self.getHistory()

    def connectStart(self):
        self.eng.set_param(self.modelName, 'SimulationCommand', 'start', 'SimulationCommand', 'pause',
                           nargout=0)

    def connectStop(self):
        self.eng.set_param(self.modelName, 'SimulationCommand', 'stop',
                           nargout=0)
    def conncectpause(self):
        self.eng.set_param(self.modelName, 'SimulationCommand', 'pause',
                           nargout=0)

    def connectController(self, controller):
        self.controller = controller
        self.controller.initialize()

    def simulate(self, action):
        # Control Loop

        # Generate the Control action based on the past outputs

        if action == 0:
            torque = -0.001
        elif action == 1:
            torque = 0.001
        else:
            torque = 0


        simulation_time = self.eng.get_param(self.modelName, 'SimulationTime')

        # self.controller.getControlEffort(self.yHist, self.tHist)
        # Set that Control Action
        self.setControlAction(torque)
        self.eng.set_param(self.modelName, 'SimulationCommand', 'continue', 'SimulationCommand', 'pause', nargout=0)
        # self.setControlAction(u)
        # self.eng.set_param(self.modelName, 'SimulationCommand', 'pause', nargout=0)
        # self.q, self.q1, self.w, self.w1 = self.getHistory()

        # print("simulation time: {0:5.3f}".format(simulation_time))

    def disconnect(self):
        self.eng.set_param(self.modelName, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()


class PIController:
    def __init__(self):

        # Maintain a History of Variables
        self.yHist = []
        self.tHist = []
        self.uHist = []
        self.eSum = 0

    def initialize(self):

        # Initialize the graph

        self.fig, = plt.plot(self.tHist, self.yHist)
        plt.xlim(0, 10)
        plt.ylim(0, 20)
        plt.ylabel("Plant Output")
        plt.xlabel("Time(s)")
        plt.title("Plant Response")

    def updateGraph(self):
        # Update the Graph
        self.fig.set_xdata(self.tHist)
        self.fig.set_ydata(self.yHist)
        self.fig, = plt.plot(self.tHist, self.yHist)
        plt.pause(0.1)
        plt.show()

    def getControlEffort(self, yHist, tHist):

        # Returns control action based on past outputs

        self.yHist = yHist

        self.tHist = tHist
        print("yhist : ", self.yHist)
        self.updateGraph()

        if (type(self.yHist) == float):
            y = self.yHist
        else:
            y = self.yHist[-1][0]