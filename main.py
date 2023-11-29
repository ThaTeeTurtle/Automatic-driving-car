import numpy as np
from layer import Layer

class CLayer(Layer):
    def __init__(self, input_shape=None, return_shape=None, input_size=4, output_size=2):
        self.input_shape = input_shape
        self.return_shape = return_shape
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5


class CarAgentSensorData:
    def __init__(self, sensor_front=[0], sensor_left=[0], sensor_right=[0], sensor_back=[0]):
        self.s_front_shape = np.shape(sensor_front)
        self.s_left_shape = np.shape(sensor_left)
        self.s_right_shape = np.shape(sensor_right)
        self.s_back_shape = np.shape(sensor_back)

    def get_data(self):
        return ([self.s_front_shape,
                 self.s_left_shape,
                 self.s_right_shape,
                 self.s_back_shape])


class AgentInputLayer(CLayer):
    def __init__(self, input_shape, return_shape, input_size, output_size):
        super().__init__(input_shape, return_shape, input_size, output_size)


class HiddenLayer(CLayer):
    def __init__(self, input_shape, return_shape, input_size, output_size):
        super().__init__(input_shape, return_shape, input_size, output_size)


class AgentOutputLayer(CLayer):
    def __init__(self, input_shape, return_shape, input_size, output_size):
        super().__init__(input_shape, return_shape, input_size, output_size)


class CarAgent:
    def __init__(self, input_layer="AgentInputLayer", output_layer="AgentOutputLayer", hidden_layers=0,
                 sensors=CarAgentSensorData()):
        self.i_layer = input_layer
        self.o_layer = output_layer
        self.h_layers = hidden_layers
        self.sensor_data = sensors.get_data()

    def create_input_layer(self):
        match self.i_layer:
            case "AgentInputLayer":
                self.i_layer = AgentInputLayer()
            case _:
                pass

    def create_hidden_layers(self):
        hidden_layers = []
        input_shape = None  # Somewhat the input data of the return shape of the layer before
        return_shape = None  # For the last one the input shape of the output layer
        for i in range(0, self.h_layers):
            hidden_layers.append(HiddenLayer(input_shape, return_shape))

    def create_output_layer(self):
        match self.o_layer:
            case "AgentOutputLayer":
                self.o_layer = AgentOutputLayer()
            case _:
                pass
