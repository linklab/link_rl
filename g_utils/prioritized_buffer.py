import math

import numpy as np
import random

from gym.spaces import Discrete

from g_utils.buffers import Buffer
from g_utils.types import Transition


class Node:
    """
    Simple node for a SumTree. Holds:
     - value
     - index (only for leaf nodes)
     - parent
     - child1
     - child2
    The index, only present for leaf nodes, corresponds to a specific transition in replay memory.
    """

    def __init__(self, value, index, parent, child1, child2):
        """
        :param value: float, value of the node
        :param index: int, index of the corresponding experience, else None
        :param parent: parent Node, only None for root node
        :param child1: left child Node, only None for leaf nodes
        :param child2: right child Node, only None for leaf nodes
        """
        self.value = value
        self.index = index
        self.parent = parent
        self.child1 = child1
        self.child2 = child2

    def __str__(self):
        return f"[Value: {self.value}, Index: {self.index}, " \
               f"Parent: {self.parent.value if self.parent is not None else None}, " \
               f"Child1: {self.child1.value if self.child1 is not None else None}, " \
               f"Child2: {self.child2.value if self.child2 is not None else None}]"

    def is_leaf_node(self):
        """
        :returns: True if node is a leaf node, i.e. has no children
        """
        return self.child1 is None and self.child2 is None

    def set_value_and_update_parent(self, new_value):
        """
        Gives the node a new value (can only be done for leaf nodes) and then
        updates the value of all related nodes, i.e. its parent. This is called
        recursively up until the root node.
        :param new_value: float, new value for the leaf node
        :raises ValueError: if the node is not a leaf node
        """
        if not self.is_leaf_node():
            raise ValueError("Can only update values of leaf nodes")
        self.value = new_value
        self.parent.recalculate_value()

    def recalculate_value(self):
        """
        Recalculates the value based on the value of its children. Then,
        tells its parent to also recalculate its value.
        """
        self.value = self.child1.value + self.child2.value
        if self.parent:
            self.parent.recalculate_value()


class SumTree:
    """
    Basic implementation of a SumTree. It has a structure where the parent is the sum of its children.
    Hence, something like:
            11
           /  \
          6    5
         / \
        4  2
    Sampling can be done by picking a number between 0 and 11 and traversing it
    down until a leaf node is reached. The number of leaf nodes should equal
    the maximum number of experiences that can be stored in memory. Then, every
    leaf node has an index corresponding to a specific experience in replay
    memory.
    """

    def __init__(self, config):
        """
        :param n_leaf_nodes: int, number of leaf nodes in the tree
        :param alpha: float, how much to prioritize
        """
        self.config = config
        self.construct_empty_tree(n_leaf_nodes=self.config.BUFFER_CAPACITY)

    def set_priority(self, i, error):
        """
        Updates the priority of the ith leaf node, based on the given error.
        :param i: int, index of the leaf node to update
        :param error: float, error of the ith experience
        """
        priority = np.power(error + self.config.PER_EPSILON, self.config.PER_ALPHA)
        self.leaf_nodes[i].set_value_and_update_parent(priority)

    def get_index_from_value(self, x):
        """
        Performs the traversing given a value x. The algorithm works by
        repeatedly performing the following:
            1) Start at the root node
            2) If x is smaller than the value of child 1, move to child 1,
               else, subtract the value of child 1 and move to child 2.
            3) Check if this is a leaf node. If so return its index and value.
            4) Repeat until done.
        :param x: float, value to use while traversing
        :returns: index of the leaf node and the corresponding value
        """
        node = self.root
        while not node.is_leaf_node():
            if x <= node.child1.value:
                node = node.child1
            else:
                x -= node.child1.value
                node = node.child2
        return node.index, node.value

    def construct_empty_tree(self, n_leaf_nodes):
        """
        Constructs an empty SumTree with a specified number of leaf
        nodes. All values will be initialized with 0. Only the leaf
        nodes are remembered as list, and the root node. They
        are set as attributes.
        :param n_leaf_nodes: int, number of leaf nodes
        """
        self.leaf_nodes = [
            Node(value=0, index=i, parent=None, child1=None, child2=None) for i in range(n_leaf_nodes)
        ]
        layer_nodes = self.leaf_nodes
        while len(layer_nodes) != 1:
            layer_nodes = self._construct_parents(layer_nodes)
        self.root = layer_nodes[0]

    def _construct_parents(self, layer_nodes):
        """
        Given a 'layer' with nodes, constructs parents for these nodes.
        A parent will be a new node with as value the sum of two children.
        If the layer has an uneven number of nodes, the last node is moved
        up to the parent layer.
        :param layer_nodes: list with Nodes
        :returns: list with parents for the Nodes
        """

        parent_nodes = []
        n_nodes = len(layer_nodes)
        for i in range(0, n_nodes, 2):
            # If uneven, the last node will be added to the parent node layer
            if i + 1 == n_nodes:
                parent_nodes.append(layer_nodes[i])
            else:
                # Determine the childeren
                child1 = layer_nodes[i]
                child2 = layer_nodes[i + 1]
                # Construct a new parent
                parent_node = Node(
                    value=child1.value + child2.value,
                    index=None,
                    parent=None,
                    child1=child1,
                    child2=child2,
                )
                # Add the parent to the children and to the list
                child1.parent = child2.parent = parent_node
                parent_nodes.append(parent_node)

        return parent_nodes

    def print_tree(self):
        print("* SUN TREE *")
        print(f'Capacity: {self.config.BUFFER_CAPACITY}')
        current_node = self.root
        print("ROOT: ", self.root)


class PrioritizedBuffer(Buffer):
    def __init__(self, action_space, config):
        super(PrioritizedBuffer, self).__init__(action_space, config)

        self.sum_tree = SumTree(config=self.config)
        # self.priorities = [None] * self.config.BUFFER_CAPACITY
        self.default_error = 100_000

        self.batch_indices = None
        self.num_transitions = 0

    def clear(self):
        super(PrioritizedBuffer, self).clear()
        self.sum_tree = SumTree(config=self.config)

    def append(self, transition):
        super(PrioritizedBuffer, self).append(transition)
        # Update the corresponding leaf node and give it the initial error
        self.sum_tree.set_priority(i=self.head, error=self.default_error)
        self.num_transitions = min(self.num_transitions + 1, self.config.BUFFER_CAPACITY)

    def update_priorities(self, batch_errors):
        """
        Given the predictions errors of the sampled batch, updates their
        errors accordingly by updating their values in the SumTree.
        :param batch_errors: np.ndarray with errors for every experience
        """
        for idx, error in zip(self.batch_indices, batch_errors):
            self.sum_tree.set_priority(i=idx, error=error)

    def sample_indices(self, batch_size):
        """
        Retrieve a batch of experiences from memory. Samples indices of
        experiences in memory using the SumTree.
        The batch indices are remembered such that the priorities of the
        corresponding experiences can be updated later on.
        :param batch_size: int, number of experiences to sample from memory
        :returns: dict with a batch of experiences
        """
        self.batch_indices, priorities = self.sample_batch_indices(batch_size)

        priorities = np.asarray(priorities).astype(float)
        sampling_probabilities = priorities / self.sum_tree.root.value
        important_sampling_weights = np.power(
            self.num_transitions * sampling_probabilities, -1.0 * self.config.PER_BETA
        )
        important_sampling_weights /= important_sampling_weights.max()

        return self.batch_indices, important_sampling_weights

    def sample_batch_indices(self, batch_size):
        """
        Samples a number of indices from the sum tree.
        :param batch_size: int, number of indices to sample
        :returns: list of indices
        """
        indices = []
        priorities = []
        for _ in range(batch_size):
            x = random.uniform(0, self.sum_tree.root.value)
            idx, priority = self.sum_tree.get_index_from_value(x)
            indices.append(idx)
            priorities.append(priority)

        return indices, priorities


if __name__ == "__main__":
    class Config:
        BUFFER_CAPACITY = 4
        PER_ALPHA = 0.6
        PER_EPSILON = 0.01
        PER_BETA = 0.4
        MODEL_PARAMETER = None
        BATCH_SIZE = 2

    config = Config()

    prioritized_buffer = PrioritizedBuffer(action_space=Discrete, config=config)
    print("#" * 100)
    prioritized_buffer.print_buffer()
    prioritized_buffer.sum_tree.print_tree()
    print("#" * 100)

    for idx in range(config.BUFFER_CAPACITY + 4):
        prioritized_buffer.append(Transition(
            observation=np.full((4,), idx),
            action=0,
            next_observation=np.full((4,), idx + 1),
            reward=1.0,
            done=False,
            info=None
        ))
        print("#" * 100)
        prioritized_buffer.print_buffer()
        prioritized_buffer.sum_tree.print_tree()
        print("#" * 100)

    print()

    print("SAMPLE & UPDATE #1")
    transition_indices, important_sampling_weights = prioritized_buffer.sample_indices(batch_size=config.BATCH_SIZE)
    print(transition_indices, important_sampling_weights)

    errors = np.ones_like(transition_indices)
    prioritized_buffer.update_priorities(errors)
    print()
    #
    # print("SAMPLE & UPDATE #2")
    # transition_indices, important_sampling_weights = prioritized_buffer.sample_indices(batch_size=config.BATCH_SIZE)
    # print(transition_indices, important_sampling_weights)
    #
    # errors = np.ones_like(transition_indices)
    # prioritized_buffer.update_priorities(errors)
    # print("#" * 100)
    # prioritized_buffer.sum_tree.print_tree()
    # print("#" * 100);print()
    # print()
    #
    # print("SAMPLE & UPDATE #3")
    # transition_indices, important_sampling_weights = prioritized_buffer.sample_indices(batch_size=config.BATCH_SIZE)
    # print(transition_indices, important_sampling_weights)
    #
    # errors = np.full(transition_indices.shape, 100.0)
    # prioritized_buffer.update_priorities(errors)
    # print("#" * 100)
    # prioritized_buffer.sum_tree.print_tree()
    # print("#" * 100);print()
    # print()
    #
    # print("SAMPLE & UPDATE #4")
    # transition_indices, important_sampling_weights = prioritized_buffer.sample_indices(batch_size=config.BATCH_SIZE)
    # print(transition_indices, important_sampling_weights)
    #
    # errors = np.full(transition_indices.shape, 10.0)
    # prioritized_buffer.update_priorities(errors)
    # print("#" * 100)
    # prioritized_buffer.sum_tree.print_tree()
    # print("#" * 100);print()
