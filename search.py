from collections import defaultdict

class Graph:
    """Generic graph class"""
    def __init__(self):
        self.graph = defaultdict(list)
        
    def addEdge(self, u, v):
        self.graph[u].append(v)

class BreadthFirstSearch(Graph):
    """Breadth first search"""
    def __init__(self):
        super().__init__()
   
    def BFS(self, s):
        visited = [False] * (len(self.graph))
        queue = []
        queue.append(s)
        visited[s] = True
        while queue:
            s = queue.pop(0)
            print(f"{s} explored")
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

class DepthFirstSearch(Graph):
    """Depth first search"""
    def __init__(self):
        super().__init__()
        
    def DFSUtil(self, v,visited):
        visited[v] = True
        print(f"{v} explored")
        
        for i in self.graph[v]:
            if visited[i] == False:
                self.DFSUtil(i, visited)
                
    def DFS(self, v):
        visited = [False] * (len(self.graph))
        self.DFSUtil(v,visited)