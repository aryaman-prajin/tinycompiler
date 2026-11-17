import operator
import torch
from tinycompiler.graph import Node, Graph


'''F'ns we know are elementwise. Anything not in here never gets fused'''
ELEMENTWISE = {
    operator.add, operator.sub, operator.mul,  # x + c, x - c and x * c trace to these
    torch.add, torch.sub, torch.mul,
    torch.relu, torch.sigmoid, torch.tanh, torch.exp, torch.neg,
}


def is_fusable(node: Node):
    if node.op == "fused":
        return True
    return node.op == "call_function" and node.target in ELEMENTWISE


def get_absorbed(node: Node) -> list[Node]:
    '''Fused nodes go into target in order'''
    if node.op == "fused":
        return node.target
    return [node]


def merge(graph: Graph, producer: Node, consumer: Node) -> Node:

    absorbed = get_absorbed(producer) + get_absorbed(consumer) # add = producer, relu = consumer
    inputs = []
    # the following block would remove Relu's inputs (add) while making AddReLU
    for each_node in absorbed:
        for input_node in each_node.inputs:
            if input_node in absorbed or input_node is producer:
                continue
            if input_node not in inputs:
                inputs.append(input_node) 

    fused = Node(
        "fused",
        "fused_" + "_".join(each_node.name for each_node in absorbed),
        target=absorbed,
        inputs=inputs,
    )

    graph.add_node(fused, index=graph.nodes.index(consumer))

    # fused's user = output
    for each_node in consumer.users:
        each_node.replace_input(consumer, fused)
        fused.users.append(each_node)
    consumer.users = []

    graph.remove_node(consumer)
    graph.remove_node(producer)

    ''' If the producer was an older fused node, the consumer's nodes still point
      at it. Point at its last node instead'''
    producer_result = get_absorbed(producer)[-1]
    for each_node in get_absorbed(consumer):
        each_node.replace_input(producer, producer_result)

    return fused


def fuse_elementwise(graph: Graph) -> Graph:
    while True:
        merged = False
        for producer in graph.nodes:
            if not is_fusable(producer):
                continue

            '''A consumer can use the same value twice, like t * t, which puts it
            in the users list twice. we wnat to count each consumer once'''
            distinct_users = []
            for each_node in producer.users:
                if each_node not in distinct_users:
                    distinct_users.append(each_node)

            if len(distinct_users) == 1 and is_fusable(distinct_users[0]):
                merge(graph, producer, distinct_users[0])
                merged = True
                break  

        if not merged:  
            break
    return graph