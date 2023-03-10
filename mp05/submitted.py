# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

from queue import PriorityQueue

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    path = []
    start = maze.start
    target = maze.waypoints[0]
    
    visited = set()
    visited.add(start)
    queue = []
    queue.append(start)
    prev = {}

    while queue:
        node = queue.pop(0)
        if node == target:
            path.append(node)   #add target to the path
            s = node
            #backtrace the path
            while prev[s] != start:
                path.append(prev[s])
                s = prev[s]
            path.append(start)
            path.reverse()
            return path

        for neighbors in maze.neighbors(node[0], node[1]):
            if neighbors not in visited:
                queue.append(neighbors)
                prev[neighbors] = node  #set parent node
                visited.add(neighbors)  #mark neighbors as visited
    return []




def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    path = []
    start = maze.start
    target = maze.waypoints[0]
    prev = dict()
    distance = {start: 0}
    frontier = PriorityQueue()
    frontier.put((0, start))

    while frontier:
        node = frontier.get()
        priority= node[0]  #get current location and priority from node
        current = node[1]

        if current == target:
            #back trace to get the path
            path.append(current)
            temp = current
            while prev[temp] != start:
                path.append(prev[temp])
                temp = prev[temp]
            path.append(start)
            path.reverse()
            return path
        
        for neighbors in maze.neighbors(current[0], current[1]):
            newDis = distance[current] + 1
            if(neighbors not in distance or newDis < distance[neighbors]):
                distance[neighbors] = newDis
                priority = newDis + abs(target[0] - neighbors[0]) + abs(target[1] - neighbors[1])
                frontier.put((priority, neighbors))
                prev[neighbors] = current
    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    return []
