class Edge:
    """
    Class description:
    A data structure that represents an Edge in a network flow graph. It stores to_node to keep similiarty with adjacency list representation of a graph
    where each node maintains a list of outgoing edges corresponding to its adjacent nodes.
    Different to an ordinary edge, it holds a reference to its own reverse edge, which is used when constructing and updating the residual network.
    """
    def __init__(self, to_node, capacity):
        """
            Function description:
            Initialisation of an edge in a network flow graph. Assigns parameters as instance variables and set initial values.

            Input:
                to_node (int): Index of the destination node of the edge.
                capacity (int): The maximum capacity of the flow the edge can hold.

            Postcondition: 
                - Initialises an edge with the specified end node and capacity.
                - Sets the initial flow to zero.
                - Initialises the reverse edge to None, to be set when creating the reverse edge in add_edge function.
            
            Time complexity: O(1)
            Time complexity analysis: Only involves assigning values.

            Space complexity: O(1)
            Space complexity analysis: Only involves storing a fixed number of variables.

        """
        self.to_node = to_node
        self.capacity = capacity
        self.flow = 0
        self.reverse = None

def add_edge(graph, from_node, to_node, capacity):
    """
        Function description:
        Adds an edge in both forward and reverse directions to support the Ford-Fulkerson algorithm.
        The reverse edge represents residual capacity in the opposite direction, allowing the algorithm to reduce flow on the forward edge when searching for augmenting paths.

        Input:
            graph (list of lists of Edge): The adjacency list representation of the graph, 
                                           where each index i holds a list of Edge objects for every outgoing edge from node i.
            from_node (int): Index of the starting node for the edge.
            to_node (int): Index of the destination node for the edge.
            capacity (int): The capacity value assign to original (forward) edge.

        Postcondition:
            - Initialises a edge from `from_node` to `to_node` in a graph with a specified capacity. Then this edge is added to the input graph.
            - Initialises a reverse edge from `to_node` to `from_node` with a capacity of zero to represent the residual capacity. Then this edge is added to the input graph.
            - Creates links the edge with its reversed version, in both ways.

        Time complexity: O(1)
        Time complexity analysis:
            - Initialisation of an Edge is O(1)
            - List operations are O(1)
            - Value assignments are O(1)

        Space complexity:
            - Input: O(V + E)
                - V = number of vertices in input graph
                - E = number of edges in input graph
            - Aux: O(1)
        Space complexity analysis:
        - Input graph follows a adjacency list which has space of O(V + E)
        - Initialisation of Edges, append operation, value assignments all cost O(1) space.
        - The graph itself may grow during the function, but the space required for this function's operations is O(1).
    """
    edge = Edge(to_node, capacity)
    rev_edge = Edge(from_node, 0)

    edge.reverse = rev_edge
    rev_edge.reverse = edge

    graph[from_node].append(edge)
    graph[to_node].append(rev_edge)


def dfs(u, t, bottleneck, visited, graph):
    """
    Function description:
    Depth-First Search algorithm to find an augmenting path in the residual graph.
    This implementation has utilised the pesucode in FIT2004 Course Note.

    Input:
        u (int): Index of the current/starting node.
        t (int): Index of the sink node.
        bottleneck (int): Minimum residual capacity available along the current path
        visited (list of bools): List to track visited nodes, visited[u] = True if u is visited, False otherwise.
        graph (list of lists of Edges): Adjacency list representation of the graph.

    Ouput:
        int: The flow value of the augmenting path found, or 0 if no path is found.

    Time complexity: O(V+E)
    # Note it can be said that the time complexity of dfs is O(E) for this task since the graph is connected and dense, E dominates V
        - V : number of vertices in input graph
        - E : number of edges in input graph
    Time complexity analysis:
        - In worst case, DFS will vist all nodes and all edges
        - Thus the time complexity is number of edges + number of nodes in the graph.

    Space complexity: O(V+E)
        - V : number of vertices in input graph
        - E : number of edges in input graph
    Space complexity analysis:
        - DFS has recursion depth of V, At each recursion step it accesses all the edges stored in the current node. Thus it requires O(V+E) space.
    """
    if u == t:
        return bottleneck

    visited[u] = True

    for edge in graph[u]:
        residual = edge.capacity - edge.flow
        if residual > 0 and not visited[edge.to_node]:
            augment = dfs(edge.to_node, t, min(bottleneck, residual), visited, graph)
            if augment > 0:
                edge.flow += augment
                edge.reverse.flow -= augment
                return augment

    return 0



def ford_fulkerson(graph, s, t):
    """
    Function description:
    Computes the maximum flow in a flow network using the Ford-Fulkerson algorithm.
    Repeatedly calls dfs to find augmenting paths and add flow until no augmenting path remains.
    This implementation has utilised the pesucode in FIT2004 Course Note.
    
    Input:
        graph (list of lists of Edges): The adjacency list representing the flow network.
        s (int): Index of the source node.
        t (int): Index of the sink node.
    
    Ouput:
        int: The maximum flow from source to sink.
    
    Time complexity: O(E*F)
        - E : number of edges in input graph
        - F : maximum flow
       
    Time complexity analysis:
        Each DFS runs in O(E) time (This is because for task, the graph is connected and dense, allowing E to dominate V).
        In the worst case, each augmentation increases the flow by only one unit, resulting in up to F augmentations / DFS calls.

    Space complexity: O(V+E)
        - V : number of vertices in input graph
        - E : number of edges in input graph
    Space complexity analysis:
        - The space complexity comes from the space complexity required for DFS, which is O(V + E)
    """
    flow = 0
    n = len(graph)
    
    while True:
        visited = [False] * n
        augment = dfs(s, t, float('inf'), visited, graph)
        if augment == 0:
            break
        flow += augment
    
    return flow

def allocate(n, m, timePreferences, proposedClasses, minimumSatisfaction):
    """
    Function description:
    This function creates a network flow graph based on the input, uses max flow/ford fulkerson algorithm allocate each student to exactly one time-slot,
    then uses a greedy approach to place those timeslot-allocated students into specific classes. 
    It enforces that:
        - Each class meets in exactly one of 20 discrete time-slots.
        - Every student is assigned to exactly one class.
        - Each class's final enrollment lies between its specified minimum and maximum.
        - At least the required number of students end up in one of their top-5 preferred slots.
    
    Approach description:
    1. For each time-slot, compute exactly how many students should be allocated to that time slot.
        - compute the min and max capacity for each time slot using the proposedClass list.
        - assign students to each timeslots up to its min amount.
        - distribute remainder of students up to each timeslot's max amount
    2. Build a flow network graph with each student's top 5 preferred time slots.
        - source -> each student (capacity=1)
        - each student -> their top 5 preferred time slots (capacity=1)
        - each time slot -> sink (num_tslot[time_slot]) #num_tslot is computed in approach step 1.
    3. Run Ford Fulkerson with current graph
        - Because we only created links between students and their top5 preferences, max flow given my ford fulkerson will equal to the number of students who got allocated to one of their top5 preference.
        - if flow from ford fulkerson < minimumSatisfaction, minimumSatisfaction cannot be satisfied.
    4. For each students who have not been allocated in approach step 3, add edges between them and all time slots not in their top 5 preference (capacity=1)
        - this can be checked by going through each student node and their edges, and find an edge with flow of 1.
    5. Run Ford Fulkerson again
        - During this step, any previously saturated path now has 0 residual capacity, so this ford fulkerson call only computes additional flow along new or leftover edges.
        - Thus, if the function has a valid inputs, flow1 + flow2 == n
    6. Extract each students' allocated time slot.
    7. For each timeslot, extract all classes running at that timeslot
        - compute the sum of their min and max capacity.
        - If sum(min) > num or sum(max) < num, the distribution/allocation is impossible, thus return None.
        - Otherwise, for each class allocate students up to its min capacity and distribute remaining students (num_tslot[timeslot] - sum(min)) up to its max capacity
        - Finally, allocate student IDs to classes according to these computed values.
    
    Input:
        n (int): number of students to be allocated.
        m (int): number of classes available
        timePreferences (list of list of int): a list of length n, where timePreferences[i] is a permutation of {0..19}
                                               ranking student i's time slot preferences (0th = most preferred, 19th = least)
        proposedClasses (list of list of int): a list of length m, where proposedClasses[i] = [time slot for class i, minimum capacity of class i, maximum capacity of class i]
        minimumSatisfaction (int): the minimum number of students that must be allocated to one of their top 5 preferred time slots.

    Output:
        - allocation (list of int): a list of legnth n, containing exactly one possible allocation of the students to the classes that satisfies all constraints.
          allocation[i] denotes the class number to which student i would be allocated.
        - None : if not allocation satisfying all constraints exists.

    Time complexity: O(n^2)
        - n : number of students to be allocated, provided as input
    
    Time complexity analysis:
        The time complexity of this function is dominated by the time complexity of its ford-fulkerson calls.
        The time complexity of ford-fulkerson algorithm is O(E * F) where:
            - E : number of edges in the network flow graph used
            - F : maximum flow of the network flow graph used
        Because the source has exactly n outgoing edges of capacity 1 (one per student), no more than n units of flow can ever leave the source.
        And because we need to allocated every single students, every outgoing edges from the source should be saturated. 
        Thus the maxium flow of the flow network graph created by this function will be n.

        Number of edges between source to student nodes : n
        Number of edges between student nodes to timeslot nodes : n*20
        Number of edges between timeslot nodes to sink : 20
        Thus, there will be n + 20*n + 20 edges. in Big O complexity, this equals to O(n)

        Therefore the time complexity of this function is
        O(n * n) = O(n^2)

    Space complexity (Auxiliary): O(n)
        - n : number of students to be allocated, provided as input
    
    Space complexity analysis:
        - Adjacency list of flow network graph : O(V + E) = O((1 + n + 20 + 1) + (n + 20*n + 20)) = O(n)
        - lists like student_time_allocation, students_in_tslot and etc take up O(n)
        - lists like class_time, class_min class_max and etc take up O(m)

        Thus, auxiliary space complexity is O(n + m).
        However, the assignment specifies that each class should have a minimum capacity of a positive integer, meaning there can't be a class with no students allocated.
        If m > n, there must be a must be a empty class where 0 students are allocated, causing allocation to be impossible give specified constraints.
        So logically speaking, space allocation team would not bother proposing classes more than the student amount.
        Thus, n >= m.
        Thus, the auxiliary space complexity of this function is 
        O(n + m) = O(n)
    """
    V = 2 + n + 20 #number of vertices = sink + source + students + time slots
    
    graph = [] # n + m + 20 + 2 vertices, n + n*20 + 20 edges; O(V + E) = O(n + m + n) = O(n + m) space

    for i in range(V): #O(n+m) time but because n > m, O(n) time
        graph.append([])

    source = 0
    stu_base = 1
    tslot_base = n + 1
    sink = tslot_base + 20

    ##### Step: 1 run ford-fulkerson with only top 5 preference for each student. ####
    #Add edge between source and all students with capacity 
    for i in range(n):
        add_edge(graph, source, stu_base+i, 1)

    #Add edge between students and there top 5 preferred timeslots.
    for i, pref in enumerate(timePreferences):
        for top5 in pref[:5]:
            add_edge(graph, stu_base + i, tslot_base + top5, 1)

    #### Compute how many students can be allocated to each time slot. ####
    #record each class's timeslot, minCap, maxCap in separate arrays.   
    class_time = [] # 3*O(m) space
    class_min = []
    class_max = []

    for time_slot, min, max in proposedClasses: #O(m) time
        class_time.append(time_slot)
        class_min.append(min)
        class_max.append(max)

    #Compute minCap and maxCap of each timeslot
    tslot_min = [0] * 20  # 2 * O(1) space
    tslot_max = [0] * 20

    for i in range (m): #O(m) time
        tslot_min[class_time[i]] += class_min[i]
        tslot_max[class_time[i]] += class_max[i]

    num_tslot = tslot_min.copy()
    remainder = n - sum(tslot_min) #number of students leftover after satifying minCapacity of each class

    for timeslot in range(20):
        space_left = tslot_max[timeslot] - tslot_min[timeslot]
        if space_left > remainder: #if the timeslot has enough space to allocated all the remaining students, just allocate them to that timeslot
            space_left = remainder
        num_tslot[timeslot] += space_left
        remainder -= space_left
        if remainder == 0:
            break
    
    if remainder > 0:  #if remainder > 0 it means there is not enough space to allocated every students
        return None
    
    #Add edge between each timeslot to sink with capacity specified in the num_tslot list (computed above)
    for i in range(20):
        add_edge(graph, tslot_base + i, sink, num_tslot[i])

    # Run ford fulkerson
    flow1 = ford_fulkerson(graph, source, sink) #O(E * F)
    if flow1 < minimumSatisfaction:   #Because current graph only includes top 5 preference of each students, if the maxflow < minSat, it mean minSat will never be satisfied.
        return None

    #### Step2. run ford fulkerson with rest of the preference ####
    
    # check which student has been allocated with their top5 preference
    allocated = [False] * n
    for edge in graph[source]:
        student_node = edge.to_node
        if edge.flow == 1:
            i = student_node - stu_base
            allocated[i] = True

    #only add edge between student and edge if they have not yet been allocated
    for i in range(n):
        if allocated[i] == False:
            for t in timePreferences[i][5:]:
                add_edge(graph, stu_base + i, tslot_base + t, 1)

    #Run ford fulkerson on finialsed graph
    flow2 = ford_fulkerson(graph, source, sink) #Ford fulkerson this time will only push exactly the amount of remaining students
    if flow1 + flow2 != n:                       #Thus sum of flows in step1 and step2 will equal to n if allocation is valid.
        return None
    
    #### Step3. Within each timeslot, allocated students to actual classes ####
    #record which timeslot each students are allocated to
    student_time_allocation = [None]*n

    for i in range(n):   #O(n*20) 
        for edge in graph[stu_base + i]:
            if edge.flow == 1:
                tslot = edge.to_node - tslot_base
                student_time_allocation[i] = tslot
                break
    
    #Record classes based on the time slot            
    classes_in_tslot = [] #O(m) space
    for _ in range(20):
        classes_in_tslot.append([])
    for j in range(m):
        classes_in_tslot[class_time[j]].append(j)

    #Record students based on time slot 
    students_in_tslot = [] #O(n) space
    for _ in range(20):
        students_in_tslot.append([])
    for i in range(n):
        t = student_time_allocation[i]
        students_in_tslot[t].append(i)

    #Allocate students to classes with the computed information
    allocation = [None] * n

    for t in range(20): #O(20*(m + m + ... + m + m*n)) = O(m*n) time, but because n >=m, O(n^2)
        class_list = classes_in_tslot[t]       # classes at time slot t
        student_list = students_in_tslot[t] # student allocated to timeslot t
        num_student = num_tslot[t] # number of students allocted to timeslot t

        if num_student == 0: #if no students are allocated to this timeslot, skip
            continue
        
        class_min_cap = 0
        class_max_cap = 0
        for cls in class_list:
            class_min_cap += class_min[cls]
            class_max_cap += class_max[cls]

        if num_student < class_min_cap or num_student > class_max_cap: #students cant be distrubuted to different classes if number of students allocated to a timeslot 
            return None                                                #if not in between mincap and maxcap of classes in that timeslot

        ## Distrubution ##
        #for every class in current timeslot, allocate students up to its minCap
        ct_len = len(class_list)
        ct_allocation = [0] * ct_len
        for i in range(ct_len):
            ct_allocation[i] = class_min[class_list[i]]

        remainder = num_student - class_min_cap #students left at that timeslot after allocating up to minCap for each class
        #Similar to the computating how many students can be allocated to each time slot (line 192~224)
        # Compute how many students can be allocated to each class 
        for i in range(ct_len):
            space_left = class_max[class_list[i]] - class_min[class_list[i]]
            if space_left > remainder:
                space_left = remainder
            ct_allocation[i] += space_left
            remainder -= space_left
            if remainder == 0:
                break
        
        #Allocate
        index = 0
        for c in range(ct_len):
            cls = class_list[c]
            seats = ct_allocation[c]
            for _ in range(seats):
                i = student_list[index]
                allocation[i] = cls
                index += 1

    return allocation



###### Test Cases ######

import unittest

class TestA2(unittest.TestCase):
    def validate_allocation(
        self, n, m, time_preferences, proposed_classes, minimum_satisfaction, allocation
    ):
        # Check correct type and length
        self.assertIsInstance(allocation, list)
        self.assertEqual(len(allocation), n)

        # Class counts and satisfaction count
        counts = [0] * m
        satisfied = 0

        for i in range(n):
            class_id = allocation[i]
            self.assertTrue(0 <= class_id < m)
            counts[class_id] += 1
            time_slot = proposed_classes[class_id][0]
            if time_slot in time_preferences[i][:5]:
                satisfied += 1

        # Check class capacity constraints
        for j in range(m):
            min_cap, max_cap = proposed_classes[j][1], proposed_classes[j][2]
            self.assertGreaterEqual(counts[j], min_cap)
            self.assertLessEqual(counts[j], max_cap)

        # Check minimum satisfaction
        self.assertGreaterEqual(satisfied, minimum_satisfaction)

    def test_a_1(self):
        n, m = 1, 1
        time_preferences = [[0] + list(range(1, 20))]
        proposed_classes = [[0, 1, 1]]
        min_satisfaction = 1
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, min_satisfaction, allocation
        )

    def test_a_2(self):
        n, m = 4, 2
        time_preferences = [[0] + list(range(1, 20))] * 4
        proposed_classes = [[0, 2, 2], [0, 2, 2]]
        min_satisfaction = 4
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, min_satisfaction, allocation
        )

    def test_a_3(self):
        n, m = 3, 2
        time_preferences = [list(range(20)) for _ in range(n)]
        proposed_classes = [[0, 2, 3], [1, 2, 3]]
        min_satisfaction = 0
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.assertIsNone(allocation)

    def test_a_4(self):
        n, m = 5, 2
        time_preferences = [
            [0] + list(range(1, 20)),
            [0] + list(range(1, 20)),
            [0] + list(range(1, 20)),
            [1] + list(range(2, 20)) + [0],
            [1] + list(range(2, 20)) + [0],
        ]
        proposed_classes = [[0, 1, 3], [1, 1, 3]]
        min_satisfaction = 3
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, min_satisfaction, allocation
        )

    def test_a_5(self):
        import random

        random.seed(42)
        n, m = 10, 3
        time_preferences = [random.sample(range(20), 20) for _ in range(n)]
        proposed_classes = [[0, 2, 4], [5, 2, 4], [10, 2, 4]]
        min_satisfaction = 5
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        if allocation:
            self.validate_allocation(
                n, m, time_preferences, proposed_classes, min_satisfaction, allocation
            )

    def test_a_6(self):
        n = 5
        m = 1
        time_preferences = [list(range(20)) for _ in range(n)]
        proposed_classes = [[0, 5, 5]]
        min_satisfaction = 5
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, min_satisfaction, allocation
        )

    def test_a_7(self):
        n = 6
        m = 3
        time_preferences = [[j] + list(range(20)) for j in [0, 0, 1, 1, 2, 2]]
        proposed_classes = [[0, 2, 2], [1, 2, 2], [2, 2, 2]]
        min_satisfaction = 6
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, min_satisfaction, allocation
        )

    def test_a_8(self):
        n = 4
        m = 2
        time_preferences = [[19, 18, 17, 16, 15] + list(range(15)) for _ in range(n)]
        proposed_classes = [[0, 2, 3], [1, 1, 2]]
        min_satisfaction = 4
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.assertIsNone(allocation)

    def test_a_9(self):
        n = 6
        m = 2
        time_preferences = [[5] + list(range(20)) for _ in range(n)]
        proposed_classes = [[5, 3, 3], [5, 3, 3]]
        min_satisfaction = 6
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, min_satisfaction, allocation
        )

    def test_a_10(self):
        n = 4
        m = 2
        time_preferences = [
            [0] + list(range(1, 20)),
            [1, 2] + list(range(3, 20)) + [0],
            [1, 2] + list(range(3, 20)) + [0],
            [2, 1] + list(range(3, 20)) + [0],
        ]
        proposed_classes = [[0, 1, 2], [1, 1, 2]]
        min_satisfaction = 2
        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, min_satisfaction, allocation
        )

    def test_g_1(self):
        n = 1
        m = 1
        prefs = [list(range(20))]
        classes = [[0, 1, 1]]
        min_sat = 1
        allocation = allocate(n, m, prefs, classes, min_sat)
        self.validate_allocation(n, m, prefs, classes, min_sat, allocation)

    def test_g_2(self):
        n = 2
        m = 1
        prefs = [list(range(20))] * 2
        classes = [[0, 2, 2]]
        min_sat = 2
        allocation = allocate(n, m, prefs, classes, min_sat)
        self.validate_allocation(n, m, prefs, classes, min_sat, allocation)

    def test_g_3(self):
        n = 2
        m = 2
        prefs = [[1, 0] + list(range(2, 20)), list(range(20))]
        classes = [[1, 1, 1], [0, 1, 1]]
        min_sat = 2
        allocation = allocate(n, m, prefs, classes, min_sat)
        self.validate_allocation(n, m, prefs, classes, min_sat, allocation)

    def test_g_4(self):
        n = 3
        m = 2
        prefs = [
            [0, 1] + list(range(2, 20)),
            [1, 0] + list(range(2, 20)),
            [0, 1] + list(range(2, 20)),
        ]
        classes = [[0, 2, 2], [1, 1, 2]]
        min_sat = 2
        allocation = allocate(n, m, prefs, classes, min_sat)
        self.validate_allocation(n, m, prefs, classes, min_sat, allocation)

    def test_g_5(self):
        n = 4
        m = 2
        prefs = [[0, 1] + list(range(2, 20)) for _ in range(2)] + [
            [1, 0] + list(range(2, 20)) for _ in range(2)
        ]
        classes = [[0, 2, 2], [1, 2, 2]]
        min_sat = 4
        allocation = allocate(n, m, prefs, classes, min_sat)
        self.validate_allocation(n, m, prefs, classes, min_sat, allocation)

    def test_g_6(self):
        n = 3
        m = 2
        prefs = [list(range(20)) for _ in range(3)]
        classes = [[0, 1, 2], [0, 1, 2]]
        min_sat = 3
        allocation = allocate(n, m, prefs, classes, min_sat)
        self.validate_allocation(n, m, prefs, classes, min_sat, allocation)

    def test_g_7(self):
        n = 4
        m = 2
        prefs = [list(range(20)) for _ in range(4)]
        classes = [[0, 2, 2], [0, 2, 2]]
        min_sat = 4
        allocation = allocate(n, m, prefs, classes, min_sat)
        self.validate_allocation(n, m, prefs, classes, min_sat, allocation)

    def test_j_1(self):
        n = 3
        m = 2
        time_preferences = [
            [0, 1, 2, 3, 4] + [t for t in range(20) if t not in [0, 1, 2, 3, 4]],
            [0, 1, 2, 3, 4] + [t for t in range(20) if t not in [0, 1, 2, 3, 4]],
            [1, 0, 2, 3, 4] + [t for t in range(20) if t not in [1, 0, 2, 3, 4]],
        ]
        proposed_classes = [[0, 2, 2], [1, 1, 2]]
        min_satisfaction = 2

        allocation = allocate(
            n, m, time_preferences, proposed_classes, min_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, min_satisfaction, allocation
        )

    def test_n_q1_1(self):
        n = 10
        m = 2
        time_preferences = [[5] + [t for t in range(20) if t != 5] for _ in range(n)]
        proposed_classes = [[5, 1, 6], [5, 1, 6]]
        minimum_satisfaction = 10
        allocation = allocate(
            n, m, time_preferences, proposed_classes, minimum_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, minimum_satisfaction, allocation
        )

    def test_n_q1_2(self):
        import random

        random.seed(1337)
        n = 50
        m = 5
        time_preferences = [random.sample(list(range(20)), 20) for _ in range(n)]
        proposed_classes = []
        for j in range(m):
            slot = j * 4
            min_cap = 5
            max_cap = 15
            proposed_classes.append([slot, min_cap, max_cap])
        minimum_satisfaction = 20
        allocation = allocate(
            n, m, time_preferences, proposed_classes, minimum_satisfaction
        )
        if allocation is not None:
            self.validate_allocation(
                n,
                m,
                time_preferences,
                proposed_classes,
                minimum_satisfaction,
                allocation,
            )

    def test_n_q1_3(self):
        n = 5
        m = 2
        time_preferences = [
            [0] + list(range(1, 20)),
            [0] + list(range(1, 20)),
            [0] + list(range(1, 20)),
            [0] + list(range(1, 20)),
            [0] + list(range(1, 20)),
        ]
        proposed_classes = [[19, 2, 3], [19, 2, 3]]
        minimum_satisfaction = 1
        allocation = allocate(
            n, m, time_preferences, proposed_classes, minimum_satisfaction
        )
        self.assertIsNone(allocation)

    def test_n_q1_4(self):
        n = 10
        m = 2
        time_preferences = [list(range(20)) for _ in range(n)]
        proposed_classes = [[0, 1, 4], [1, 1, 4]]
        minimum_satisfaction = 0
        allocation = allocate(
            n, m, time_preferences, proposed_classes, minimum_satisfaction
        )
        self.assertIsNone(allocation)

    def test_n_q1_5(self):
        n = 10
        m = 2
        time_preferences = [list(range(20)) for _ in range(n)]
        proposed_classes = [[0, 5, 5], [1, 5, 5]]
        minimum_satisfaction = 0
        allocation = allocate(
            n, m, time_preferences, proposed_classes, minimum_satisfaction
        )
        self.validate_allocation(
            n, m, time_preferences, proposed_classes, minimum_satisfaction, allocation
        )


    def test_gc_1(self):
        n = 4
        m = 2
        time_preferences = [[0] + list(range(1, 20)),
                            [0] + list(range(1, 20)),
                            [0] + list(range(1, 20)),
                            [0] + list(range(1, 20))]
        proposed_classes = [[0, 1, 20], [5, 3, 20]]
        min_satisfaction = 1
        allocation = allocate(n, m, time_preferences, proposed_classes, min_satisfaction)
        self.validate_allocation(n, m, time_preferences, proposed_classes, min_satisfaction, allocation)
        
    def test_gc_2(self):
        n = 7
        m = 3
        time_preferences = [[0] + list(range(1, 20)),
                            [0] + list(range(1, 20)),
                            [0] + list(range(1, 20)),
                            [0] + list(range(1, 20)),
                            [0] + list(range(1, 20)),
                            [0] + list(range(1, 20)),
                            [0] + list(range(1, 20))]

        proposed_classes = [[0, 1, 20], [0, 3, 5] ,[5, 3, 20]]
        min_satisfaction = 2
        allocation = allocate(n, m, time_preferences, proposed_classes, min_satisfaction)
        self.validate_allocation(n, m, time_preferences, proposed_classes, min_satisfaction, allocation)

    def test_gc_3(self):
        n = 6
        m = 2
        time_preferences = [list(range(0, 20))] * 6
        proposed_classes = [[0, 5, 10], [5, 1, 10]]
        minimum_satisfaction = 6
        allocation = allocate(n, m, time_preferences, proposed_classes, minimum_satisfaction)
        self.assertIsNone(allocation)
    
    def test_e_q1_1(self):
        n = 3
        m = 2
        min_satisfaction = 3

        import itertools
        time_preferences_base = [
            list(range(20)),
            [1, 2, 17, 18, 19] + list(range(3, 17)) + [0],
            [1, 16, 17, 18, 19] + list(range(2, 16)) + [0],
        ]
        proposed_classes_base = [[1, 1, 1], [2, 1, 2]]
        for time_preferences in itertools.permutations(time_preferences_base):
            for proposed_classes in itertools.permutations(proposed_classes_base):
                allocation = allocate(n, m, time_preferences, proposed_classes, min_satisfaction)
                self.validate_allocation(n, m, time_preferences, proposed_classes, min_satisfaction, allocation)

    def test_e_q1_2(self):
        n = 5
        m = 2
        time_preferences = [list(range(0, 20))] * 5
        proposed_classes = [[0, 1, 10], [1, 4, 10]]
        minimum_satisfaction = 4
        # Sample output [0, 1, 1, 1, 1]
        allocation = allocate(n, m, time_preferences, proposed_classes, minimum_satisfaction)
        self.validate_allocation(n, m, time_preferences, proposed_classes, minimum_satisfaction, allocation)

    def test_e_q1_3(self):
        n = 10
        m = 3
        time_preferences = [list(range(0, 20))] * 9 + [[5] + [0, 1, 2, 3, 4] + list(range(6, 20))]
        proposed_classes = [[0, 4, 10], [5, 1, 10], [6, 3, 10]]
        minimum_satisfaction = 6
        allocation = allocate(n, m, time_preferences, proposed_classes, minimum_satisfaction)
        self.validate_allocation(n, m, time_preferences, proposed_classes, minimum_satisfaction, allocation)

if __name__ == "__main__":
    unittest.main()