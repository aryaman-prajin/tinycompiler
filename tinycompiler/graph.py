from typing import Any # cool feature to remember for complete type hinting 


class Node:
    def __init__(
        self, 
        # op ∈ {"placeholder", "call function", "call module", "output", "fused"}
        op: str, 
        name: str, # irrelevant to the compiler logic
        target: Any = None, #specific op from the branch specified by op
        inputs: list['Node'] | None = None, #moreso for the graph picture
        users: list['Node'] | None = None, # who reads the result (obv needed for DCE and fusion)
        args: list[Any] | None = None, # maintains constants that input wouldn't, and non-commutative expressions
    ):
        self.op = op
        self.name = name
        self.target = target
        '''Easy enough way to assign different empty lists to each node.
        And in case inputs have been assigned already to an operation, then
        all is well. Strange pythonic behaviour to not let this happen in the
        method signature. The more you know...'''
        self.inputs: list['Node'] = inputs if inputs is not None else [] 
        self.users: list['Node'] = users if users is not None else []
        '''eg [relu, 2] for relu * 2. Extremely useful for non-commutative expressions,
         and those with constants as mentioned earlier.
        Hand-built nodes with no constants and such can skip it cause it just copies inputs'''
        self.args: list[Any] = args if args is not None else list(self.inputs)

    def replace_input(self, old: 'Node', new: 'Node'):
        # keeps inputs and args in sync, when it is all convoluted by fusion
        self.inputs = [new if each_node is old else each_node for each_node in self.inputs]
        self.args = [new if arg is old else arg for arg in self.args]

    def __repr__(self):
        return f"Node(op='{self.op}', name='{self.name}')"


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []
    
    def add_node(self, node: Node, index: int | None = None):
        '''Append by default. A pass that replaces nodes can pass an index and
           so the new node sits at the right point in execution order'''
        if index is None:
            self.nodes.append(node) # to the end
        else:
            self.nodes.insert(index, node) # or at the specified index
        for each_node in node.inputs:
            each_node.users.append(node) # update users as a result

    def remove_node(self, node: Node):
        self.nodes.remove(node)
        for each_node in node.inputs:
            each_node.users.remove(node)