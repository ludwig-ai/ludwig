#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging

from ludwig.constants import TIED

logger = logging.getLogger(__name__)


def topological_sort(graph_unsorted):
    """Repeatedly go through all of the nodes in the graph, moving each of the nodes that has all its edges
    resolved, onto a sequence that forms our sorted graph.

    A node has all of its edges resolved and can be moved once all the nodes its edges point to, have been moved from
    the unsorted graph onto the sorted one.
    """

    # This is the list we'll return, that stores each node/edges pair
    # in topological order.
    graph_sorted = []

    # Convert the unsorted graph into a hash table. This gives us
    # constant-time lookup for checking if edges are unresolved, and
    # for removing nodes from the unsorted graph.
    graph_unsorted = dict(graph_unsorted)

    # Run until the unsorted graph is empty.
    while graph_unsorted:

        # Go through each of the node/edges pairs in the unsorted
        # graph. If a set of edges does not contain any nodes that
        # haven't been resolved, that is, that are still in the
        # unsorted graph, remove the pair from the unsorted graph,
        # and append it to the sorted graph. Note here that by using
        # using the items() method for iterating, a copy of the
        # unsorted graph is used, allowing us to modify the unsorted
        # graph as we move through it. We also keep a flag for
        # checking that that graph is acyclic, which is true if any
        # nodes are resolved during each pass through the graph. If
        # not, we need to bail out as the graph therefore can't be
        # sorted.
        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            if edges is None:
                edges = []
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            # Uh oh, we've passed through all the unsorted nodes and
            # weren't able to resolve any of them, which means there
            # are nodes with cyclic edges that will never be resolved,
            # so we bail out with an error.
            raise RuntimeError("A cyclic dependency occurred")

    return graph_sorted


def topological_sort_feature_dependencies(features):
    # topological sorting of output features for resolving dependencies
    dependencies_graph = {}
    output_features_dict = {}
    for feature in features:
        dependencies = []
        if "dependencies" in feature:
            dependencies.extend(feature["dependencies"])
        if TIED in feature:
            dependencies.append(feature[TIED])
        dependencies_graph[feature["name"]] = dependencies
        output_features_dict[feature["name"]] = feature
    return [output_features_dict[node[0]] for node in topological_sort(dependencies_graph)]


if __name__ == "__main__":
    graph_unsorted = [(2, []), (5, [11]), (11, [2, 9, 10]), (7, [11, 8]), (9, []), (10, []), (8, [9]), (3, [10, 8])]
    logger.info(topological_sort(graph_unsorted))
    graph_unsorted = [("macro", ["action", "contact_type"]), ("contact_type", None), ("action", ["contact_type"])]
    logger.info(topological_sort(graph_unsorted))
