"""
This module defines Self-Organizing Maps, also known as Kohonen Maps.
This is inspired by the MiniSom and the Sompy implementations.
In particular, the training must be an online process in this version,
because the input data is discovered as the Agent interacts with the
Environment.

Sources:
https://github.com/JustGlowing/minisom/blob/master/minisom.py
https://github.com/ttlg/sompy/blob/master/sompy/sompy.py
"""

import numpy as np


def fast_norm(vector):
    return np.sqrt(np.dot(vector, vector.T))


class SOM(object):

    def __init__(self,
                 dimx, dimy,
                 unit_len,
                 sigma=1.0,
                 learning_rate=0.5,
                 init='random'):
        """
        Create a new Self-Organizing Map, with a rectangular shape.

        :param dimx: The X-dimension of the map.
        :param dimy: The Y-dimension of the map.
        :param unit_len: Size of the vector associated to each unit.
        :param sigma: Size of the neighborhood.
        :param learning_rate: Initial learning rate.
        :param init: Method to initialize the neurons' units (vectors).
            Either 'random' (uniform distribution in `[0,1)`), or `zero`
            (all values are set to 0s).
        """

        # Shape
        self.dimx = dimx
        self.dimy = dimy
        self.unit_len = unit_len
        self.shape = (dimx, dimy)

        # Accelerates the computation of coordinates later
        self.nb_units = dimx * dimy
        self.neigx = np.arange(dimx)
        self.neigy = np.arange(dimy)
        # Each neuron is given a unique identifier and its (x,y) coordinates.
        # `coords_map` makes the mapping between identifiers and coordinates,
        # e.g., neuron #0 has coordinates `coords_map[0] = (0,0)`, neuron #1
        # has coordinates (0,1), etc.
        self.coords_map = [(i // dimy, i % dimy) for i in range(dimx * dimy)]

        # Parameters
        self.initial_sigma = sigma
        self.initial_lr = learning_rate

        # Weights of the map (dimx * dimy vectors of unit_len values)
        if init == 'zero':
            self.units = np.zeros((dimx, dimy, unit_len))
        elif init == 'random':
            self.units = np.random.rand(dimx, dimy, unit_len)
        else:
            raise Exception(f'Unrecognized `init` argument: {init}')

        # Iteration step, used to compute the decay
        self.step = 0

        # Keep track of the quantization error
        self.error = []
        # Keep track of how many steps each unit is the Best Matching Unit (BMU)
        self.bmu_count = np.zeros(self.shape).astype(int)

    def compute_winner_node(self, data):
        """
        Compute the Best Matching Unit to the input vector `data`.

        The Best Matching Unit is the closest neuron to data, i.e., the
        one with the lower activation (a bit misleading).

        :param data: The input vector, a NumPy array of same size as #input_len.
        :return: The closest neuron's identifier (
        """
        activation_map = self._compute_activation_map(data)
        bmu = np.argmin(activation_map)
        self.bmu_count[self.coords_map[bmu]] += 1
        return bmu

    def update(self, data, winner):
        """
        Update the SOM towards a pattern to learn, given a winner node.

        Every node in the map is updated, based on the neighborhood
        function (relative distance to the winner node).

        :param data: The pattern to learn, i.e., a vector of weights (of same
            shape as #input_len).
        :param winner: The index of the winner node, i.e., the closest node.
        """
        # The update formula is:
        # u_m = u_m + λ*φ(k,m,N)(u_k' - u_m)
        # Where:
        # - u_m is each action unit (indexed by m)
        # - λ is the learning rate
        # - φ is the neighborhood function
        #   - k is the index of the winner node (center of neighborhood)
        #   - N is the size of the neighborhood
        # - u_k' is the data to learn
        lr = self.learning_rate
        neighborhood = self.neighborhood(winner) * lr  # <=> λ*φ(k,...)
        unit_winner = self.get_unit(winner)
        for x, y in np.ndindex(self.shape):
            psi = neighborhood[x, y]  # <=> λ*φ(k,m,N)  with m=[x,y], k=winner
            self.units[x, y] += (psi * (data - self.units[x, y]))
        self.error.append(fast_norm(data - unit_winner))
        self.step += 1

    def get_unit(self, idx):
        """Return the weights of a neuron, identified by its index `idx`."""
        return self.units[self.coords_map[idx]]

    def neighborhood(self, center):
        """
        Compute the neighborhood matrix for all nodes.

        :param center: The index of the center neuron in this neighborhood.
        :return: A matrix in which an element, indexed by (i,j), is the
            distance of the neuron with coordinates (i,j) to the center neuron
            (weighted by the size of the neighborhood).
        """
        center = self.coords_map[center]
        return self._gaussian(center, self.sigma)

    def _compute_activation_map(self, data):
        """
        Compute the matrix in which an element (i,j) is the response of the
        neuron with coordinates (i,j) to the vector data.
        The lower the activation, the closest the neuron is to data.
        """
        data = np.asarray(data)
        sub = np.subtract(data, self.units)  # x - w
        # https://docs.scipy.org/doc/numpy/reference/arrays.nditer.html
        activation_map = np.zeros(self.shape)
        it = np.nditer(activation_map, flags=['multi_index'])
        while not it.finished:
            # || x - w ||
            activation_map[it.multi_index] = fast_norm(sub[it.multi_index])
            it.iternext()
        return activation_map

    def _decay(self, value, step):
        """
        Decay a value based on the time step.

        To allow for long-term adaptation, we simply return the value without
        decaying.
        """
        return value
        # return value / 2 ** (step // 1000)

    def _gaussian(self, center, sigma):
        """Return a matrix of gaussian distances."""
        d = 2 * np.pi * sigma * sigma
        ax = np.exp(-np.power(self.neigx - center[0], 2) / d)
        ay = np.exp(-np.power(self.neigy - center[1], 2) / d)
        return np.outer(ax, ay)  # the external product gives a matrix

    def quantization_error(self):
        """Return the mean error of the map."""
        return sum(self.error) / (len(self.error) + 10E-300)

    @property
    def sigma(self):
        return self._decay(self.initial_sigma, self.step)

    @property
    def learning_rate(self):
        return self._decay(self.initial_lr, self.step)
