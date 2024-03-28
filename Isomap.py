#https://github.com/ffp5/Isomap_implementations

import sys
import numpy as np
import math
import types
import matplotlib.pyplot as plt
import networkx as nx
import random
from math import e
from re import M, T
import subprocess
import os
import time


def connected_graph(name,n,bool=False):
    "" "Generate a random biconnected graph with n nodes" ""
    # print("we enter the function connected_graph")
    if name == "Erdos Renyi":
        # print("ER")
       X = nx.gnp_random_graph(n, 0.5, directed=False)
       while(not nx.is_connected(X)):
            X = nx.gnp_random_graph(n, 0.5, directed=False)
    elif name == "Barabasi_albert":
        # print("WS")  
        X = nx.barabasi_albert_graph(n, 6)
        while(not nx.is_connected(X)):
             X = nx.barabasi_albert_graph(n, 6)
    elif name == "random_regular_graph":
        # print("RRG")
        X = nx.random_regular_graph(6, n)
        while(not nx.is_connected(X)):
            X = nx.random_regular_graph(5, n)
    list_pos={}
    for u in X.nodes():
        list_pos[u]=np.random.rand(2)
    for (u, v) in X.edges():
        X[u][v]['weight'] = np.linalg.norm(list_pos[u]-list_pos[v])
    if bool:
        nx.draw(X, with_labels=True)
        plt.show()
    return X, list_pos


def create_ampl_dat_grah(D, filename):
    "" "Create a .dat file for the ampl model" ""
    with open(filename, 'w') as file:
        file.write("# gen by python\n")
        file.write("param Kdim :=" + "2" +";\n")
        file.write("param n := " + str(len(D.nodes())) + ";\n")
        file.write("param : E : c I :=\n")
        for (u, v) in D.edges():
            file.write(""+str(u+1) + " " + str(v+1) + " " + str(D[u][v]['weight'])+ " "+"1\n")
        file.write(";\n")
        file.close()

def compute_DM(method,D):
    if method=="floyd_warshall":
        D_new=floyd_warshall(D)
    elif method=="centers_tracker":
        D_new=centers_tracker_method(D)
    elif method=="random":
        D_new=random_method(D)
    elif method=="BFS":
        D_new=BFS(D)
    elif method=="Push-Pull" or method=="SSV":
        D_new=np.zeros((D.shape[0],D.shape[0]))
        for i in range(D.shape[0]):
            for j in range(i,D.shape[0]):
                if i != j :
                    D_new[i][j]=np.linalg.norm(np.array((D[i][0],D[i][1]))-np.array((D[j][0],D[j][1])), ord=2)
                    D_new[j][i]=D_new[i][j]
    return D_new

                

# Distance matrix creation algos
def floyd_warshall(D):
    D_prime= nx.to_numpy_array(D)
    "" "Floyd-Warshall algorithm that take an nx and return an array" ""
    for i in range(D_prime.shape[0]):
        for j in range(D_prime.shape[1]):
            if i != j and D_prime[i][j] == 0:
                D_prime[i][j] = math.inf
    for k in range(D_prime.shape[0]):
        for i in range(D_prime.shape[1]):
                for j in range(D_prime.shape[0]):
                    if i != k and j != k and i !=j:
                        if D_prime[i][j] > D_prime[i][k] + D_prime[k][j]:
                            D_prime[i][j] = D_prime[i][k] + D_prime[k][j]
    if np.array_equal(D_prime, D_prime.T):
        return D_prime
    else:
        print("error:The graph is not symetric")
        return None

def dijkstra(D):
    "" "Dijkstra algorithm that take an nx and a source node s and return an array" "" 
    D_prime= nx.to_numpy_array(D)
    n = D_prime.shape[0]
    D_new=[]
    for s in range(n):
        dist = [math.inf] * n
        dist[s] = 0
        Q = [i for i in range(n)]
        while len(Q) > 0:
            u = Q[0]
            for i in Q:
                if dist[i] < dist[u]:
                    u = i
            Q.remove(u)
            for v in range(n):
                if D_prime[u][v] != 0 and dist[v] > dist[u] + D_prime[u][v]:
                    dist[v] = dist[u] + D_prime[u][v]
        D_new.append(dist)
    D_new = np.array(D_new)

    if np.array_equal(D_new, D_new.T):
        return D_new
    else:
        print("error:The graph is not symetric")
        return None
    
def extract_data(file_path,dim):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    array=np.zeros((dim,2))
    for line in lines[2:]: 
        print(line) # Skip header lines
        row = line.split()
        if len(row) > 2:
            # print(row)
            i=int(row[0])-1
            j=int(row[1])-1
            # print(i,j)
            array[i][j]=float(row[2])
    return array

def extract_data_dim(file_path,n):
    print("extract_data_for_dim_above_3", file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data1 = []
    data2 = []
    for line in lines[2:]:  # Skip header lines
        row = line.split()
        if len(row) > 1:
            if ':=' in row:
                if data2==[]:
                    data2=data1.copy()
                    data1 = []
                else:
                    data2=np.concatenate((np.array(data2),np.array(data1)))
                    data1=[]
                continue
            row = [float(element) for element in row[1:]]  # Convert elements to float
            data1.append(row)
    if data2 != []:
        # print(data1)
        data2=np.concatenate((np.array(data2),np.array(data1)),axis=1)
    else:
        data2=np.array(data1) 
    return data2


## return the distance matrix of a realization
def distanceMatrix(x, p=2):
    n = len(x[:,0])
    D = np.zeros((n,n))
    for u in range(n-1):
        for v in range(u+1,n):
            D[u,v] = np.linalg.norm(np.subtract(x[u,:],x[v,:]), ord=p)
            D[v,u] = D[u,v]
    return D

## convert a distance matrix to a Gram matrix
def dist2Gram(D):
    n = D.shape[0]
    J = np.identity(n) - (1.0/n)*np.ones((n,n))
    G = -0.5 * np.dot(J,np.dot(np.square(D), J))
    return G

## factor a square matrix
def factor(A):
    n = A.shape[0]
    (evals,evecs) = np.linalg.eigh(A)
    evals[evals < 0] = 0  # closest SDP matrix
    X = evecs #np.transpose(evecs)
    sqrootdiag = np.eye(n)
    for i in range(n):
        sqrootdiag[i,i] = math.sqrt(evals[i])
    X = X.dot(sqrootdiag)
    return np.fliplr(X)

## classic Multidimensional scaling
def MDS(B, eps = 1e-9):
    n = B.shape[0]
    x = factor(B)
    (evals,evecs) = np.linalg.eigh(x)
    K = len(evals[evals > eps])
    if K < n:
        # only first K columns
        x = x[:,0:K]
    return x

## principal component analysis
def PCA(B, K):
    x = factor(B)
    n = B.shape[0]
    if isinstance(K, str):
        K = n
    if K < n:
        # only first K columns
        x = x[:,0:K]
    return x

def complete_PCA_part(D,eps,representation=True):
    G=dist2Gram(D)
    X=MDS(G,eps)
    n=X.shape[0]
    # K=X.shape[1]
    K=2
    # print("dimension can be reduced from", n, "to", K)

    if representation:
        if K > 3:
            K = 3
        elif K < 2:
            K = 2
        print("representing in", K, "dimensions")

        X = PCA(G,K)

        if K == 2:
            plt.scatter(X[:,0], X[:,1])
        elif K == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(X[:,0], X[:,1], X[:,2])
            ax.plot(X[:,0], X[:,1], X[:,2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        plt.show()
    else:
        X=PCA(G,K)
    return X

def centers_tracker_method(D):
    D_prime= nx.to_numpy_array(D)
    # print(f'D_prime={D_prime}')

    # Obtenir le degré de chaque noeud
    degrees = D.degree()

    # Trier les noeuds par degré en ordre décroissant
    sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
    
    # Les noeuds les plus connectés sont les premiers de la liste triée
    most_connected_nodes = sorted_nodes[0]
    # print(f'sorted_nodes={sorted_nodes}')
    Q=[i for i in range(D_prime.shape[0])]
    centers=[]
    reached_list=[]
    all_reached=[]

    i=0
    while len(sorted_nodes) > 0:
        # print(f'Q={Q}')
        # print(f'i={i}')
        add=False
        if len(Q) == 0:
            break
        u = sorted_nodes[i][0]

        # print(f'u={u}')
        reached=[]
        if all_reached == []:
            for v in Q:
                if D_prime[u][v] != 0.:
                    reached.append(v)
                    all_reached.append(v)
                    add=True
        elif u in all_reached:
            for v in Q:
                if D_prime[u][v] != 0.:
                    reached.append(v)
                    all_reached.append(v)
                    add=True
        # print(f'reached={reached}')
        for v in reached:
            Q.remove(v)
        reached.append(u)
        if add:
            sorted_nodes.pop(i)
            i=0
            # print(f'add u={u} to the centers list')
            centers.append(u)
            reached_list.append(reached)
        else:
            i+=1
        
    list=[u for u in centers]

    for i,center in enumerate(list):
        for j,reached in enumerate(reached_list.copy()):
            if center in reached and i != j:
                reached_list[j].remove(center)

    path_between_centers=dijkstra_for_centers_tracker(D,list)
    # print(f'list={list}')
    # print(f'reached_list={reached_list}')
    # print(f'path_between_centers={path_between_centers}')
    for i in range(D_prime.shape[0]):
        i_center = list[next((index for index, sublist in enumerate(reached_list) if i in sublist), None)]
        for j in range(i+1):
            j_center = list[next((index for index, sublist in enumerate(reached_list) if j in sublist), None)]
            if i != j and D_prime[i][j] == 0:
                # print(f"i={i}, j={j}, i_center={i_center}, j_center={j_center}")
                # print(f'D_prime[i][i_center]={D_prime[i][i_center]}')
                # print(f'D_prime[j][j_center]={D_prime[j][j_center]}')
                # print(f'path_between_centers[list.index(i_center)][list.index(j_center)]={path_between_centers[list.index(i_center)][list.index(j_center)]}')
                D_prime[i][j]=D_prime[i][i_center]+D_prime[j][j_center]+path_between_centers[list.index(i_center)][list.index(j_center)]
                D_prime[j][i]=D_prime[i][j]
            elif i != j and D_prime[i][j] != 0:
                D_prime[i][j]=D_prime[i][j]
    
    if np.array_equal(D_prime, D_prime.T):
        return D_prime
    else:
        print("error:The graph is not symetric")
        return None


def dijkstra_for_centers_tracker(D,list):
    "" "Dijkstra algorithm that take an nx and a source node s and return an array" "" 
    D_prime= nx.to_numpy_array(D)
    n = D_prime.shape[0]
    D_new=[]
    for s in list:
        dist = [math.inf] * n
        dist[s] = 0
        Q = list.copy()
        while len(Q) > 0:
            u = Q[0]
            for i in Q:
                if dist[i] < dist[u]:
                    u = i
            Q.remove(u)
            for v in range(n):
                if D_prime[u][v] != 0 and dist[v] > dist[u] + D_prime[u][v]:
                    dist[v] = dist[u] + D_prime[u][v]
        D_new.append(dist)
    D_new = np.array(D_new)
    # print(f'D_new={D_new}')
    return D_new


def random_method(D):
    D_prime= nx.to_numpy_array(D)
    seen=[]
    u=np.random.randint(0, D_prime.shape[0])
    seen.append(u)
    while len(seen) < D_prime.shape[0]:
        # print(f'new node {u} in the seen list {seen}')
        v=0
        weight=0
        Q=[i for i in range(D_prime.shape[0])]
        # print(Q)
        Q.remove(u)
        for k in range(D_prime.shape[0]):
            if D_prime[u][k] != 0:
                Q.remove(k)
                v=k
        weight=D_prime[u][v]

        already_passed=[]
        while len(Q) > 0:
            # print(f'new node {v}, Q list {Q}')
            ancient_v=v
            for k in range(D_prime.shape[0]):
                if D_prime[u][k] == 0 and D_prime[v][k] != 0 and u != k and (k not in already_passed):
                    already_passed=[]
                    Q.remove(k)
                    v=k
                    D_prime[u][k]=weight+D_prime[v][k]
                    D_prime[k][u]=weight+D_prime[v][k]
                    weight=D_prime[u][k]
            if ancient_v==v:
                for k in range(D_prime.shape[0]):
                    if D_prime[u][k] != 0 and (k not in already_passed):
                        # print(f'case ancient_v k={k}, alredy_passed={already_passed}')
                        already_passed.append(k)
                        v=k
                        # print(v)
                        weight=D_prime[u][k]
                        break
        while True:
            u=np.random.randint(0, D_prime.shape[0])
            if u not in seen:
                seen.append(u)
                break
            # print(f'there is an error with the node {u} and the node {v}')
    return D_prime      

def highest_degree_vertex(G):
    """Find the vertex with the highest degree in G."""
    max_degree_node = max(G.nodes, key=G.degree)
    return max_degree_node

def sample_from_sphere(center, distance, dimension):
    """Sample a point from a sphere centered at 'center' with a given 'distance' in a space of 'dimension' dimensions."""
    realiz_sample = np.random.normal(size=dimension)
    realiz_sample /= np.linalg.norm(realiz_sample)  # Normalize to get a unit vector
    point = center + distance * realiz_sample
    return point

def BFS(G, K=2):
    """Modified BFS that samples each node's position in a K-dimensional space based on its connections."""
    root = highest_degree_vertex(G)
    positions = {root: np.zeros(K)}  # Starting at the origin for the root
    explored = {root}
    Q = [root]

    while Q:
        v = Q.pop(0)
        x_v = positions[v]

        for w in G.neighbors(v):
            if w not in explored:
                explored.add(w)
                dvw = G[v][w]['weight']
                x_w = sample_from_sphere(x_v, dvw, K)
                positions[w] = x_w
                Q.append(w)

    # Complete the graph by adding missing edges
    for u in G.nodes():
        for v in G.nodes():
            if u != v and not G.has_edge(u, v):
                # Add an edge with weight equal to the Euclidean distance between u and v
                distance = np.linalg.norm(positions[u] - positions[v])
                G.add_edge(u, v, weight=distance)

    return calculate_distance_matrix(positions)

def calculate_distance_matrix(positions):
    """Calculate the N x N distance matrix from the positions."""
    nodes = list(positions.keys())
    n = len(nodes)
    distance_matrix = np.zeros((n, n))

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(positions[u] - positions[v])
            else:
                distance_matrix[i, j] = 0  # Distance to itself is 0

    return distance_matrix

def final_calcultate_dm(D):
    D_new=np.zeros((D.shape[0],D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(i,D.shape[0]):
            if i != j :
                D_new[i][j]=np.linalg.norm(np.array((D[i][0],D[i][1]))-np.array((D[j][0],D[j][1])), ord=2)
                D_new[j][i]=D_new[i][j]
    return D_new

def complete_isomap(method,D,eps,representation=False):
    if method=="floyd_warshall":
        D_prime=floyd_warshall(D)
    elif method=="centers_tracker":
        D_prime=centers_tracker_method(D)
    elif method=="random":
        D_prime=random_method(D)
    elif method=="BFS":
        D_prime=BFS(D)
    elif method=="Push-Pull":
        D_prime=D #we use already extract data ans compute_DM
    elif method=="SSV":
        D_prime=D #we use already extract data ans compute_DM
    X=complete_PCA_part(D_prime,eps,representation)
    return X
def LDE(complete_dist_matrix, old_distance_matrix, E):
    max_deviation = -np.inf
    for i, j in E:
        d_ij_prime = complete_dist_matrix[i][j]
        d_ij = old_distance_matrix[i][j]
        deviation = np.abs(d_ij_prime - d_ij)
        max_deviation = max(max_deviation, deviation)
    return max_deviation

def MDE(complete_dist_matrix, old_distance_matrix, E):
    total_deviation = 0
    num_pairs = len(E)
    for i, j in E:
        d_ij_prime = complete_dist_matrix[i][j]
        d_ij = old_distance_matrix[i][j]
        # print(f'd_ij_prime={d_ij_prime}, d_ij={d_ij}')
        deviation = np.abs(d_ij_prime - d_ij)
        total_deviation += deviation
    return total_deviation / num_pairs

def rmsd(complete_dist_matrix, old_distance_matrix, E) -> float:
    total_deviation = 0
    num_pairs = len(E)
    for i, j in E:
        d_ij_prime = complete_dist_matrix[i][j]
        d_ij = old_distance_matrix[i][j]
        deviation = (d_ij_prime - d_ij)**2
        total_deviation += deviation

    return np.sqrt(total_deviation / num_pairs)

def compute_measure(metric_name: str, complete_dist_matrix, old_distance_matrix, graph: nx.Graph) -> float:
    metric = 0
    E = list(graph.edges())
    if metric_name == "MDE":
        metric = MDE(complete_dist_matrix, old_distance_matrix, E)
    elif metric_name == "LDE":
        metric = LDE(complete_dist_matrix, old_distance_matrix, E)
    elif metric_name == "RMSD":
        metric = rmsd(complete_dist_matrix, old_distance_matrix, E)

    return metric

def comparison_heuristics(tries_number,number_of_nodes,eps,graph_generator_method,verbose=False):
    mesures=["MDE","LDE","RMSD","time"]
    heuristics=["floyd_warshall","centers_tracker","random","BFS","Push-Pull","SSV"]
    # heuristics=["floyd_warshall"]
    # graph_generator_method=["Erdos Renyi","Barabasi_albert","random_regular_graph"]
    comparaison_results= np.zeros((4,6))

    results = {} 


    if verbose:
        print(f"graph generator method {graph_generator_method}")

    for heuristic in heuristics:
        for mesure in mesures:
            results[f'{heuristic}_{mesure}'] = 0 #might change

    for test in range(tries_number):
        if verbose:
            print(f"test number {test}")
        # print(graph_gen)
        time.sleep(1)
        G,list_nodes=connected_graph(graph_generator_method,number_of_nodes)
        D=nx.to_numpy_array(G)
        for heuristic in heuristics:
            if verbose:
                print(f"heuristic {heuristic}")
            if heuristic=="Push-Pull":
                create_ampl_dat_grah(G,"graph.dat")
                # Get the absolute path to the home directory
                home_dir = os.path.expanduser('~')
                # Define the paths to the ampl executable and the dgp.run file
                ampl_path = os.path.join(home_dir, 'ampl', 'ampl')
                dgp_run_path = './pap.run'
                # Run the subprocess
                start_time = time.time()
                subprocess.run([ampl_path, dgp_run_path])
                end_time = time.time()
                P=extract_data_dim("./graph_pap.rlz",number_of_nodes)
                P=final_calcultate_dm(P)
                elapsed_time_1 = end_time - start_time
                start_time = time.time()
                P=complete_isomap(heuristic,P,eps)
                P=final_calcultate_dm(P)
                end_time = time.time()
                elapsed_time_2 = end_time - start_time
                total_time = elapsed_time_1 + elapsed_time_2
                results[f'{heuristic}_time'] += (total_time)
                results[f'{heuristic}_MDE'] += (compute_measure("MDE",P,D,G))
                results[f'{heuristic}_LDE'] += (compute_measure("LDE",P,D,G))
                results[f'{heuristic}_RMSD'] += (compute_measure("RMSD",P,D,G))
            elif heuristic=="SSV":
                create_ampl_dat_grah(G,"graph.dat")
                # Get the absolute path to the home directory
                home_dir = os.path.expanduser('~')
                # Define the paths to the ampl executable and the dgp.run file
                ampl_path = os.path.join(home_dir, 'ampl', 'ampl')
                dgp_run_path = './ssv.run'
                # Run the subprocess
                start_time = time.time()
                subprocess.run([ampl_path, dgp_run_path])
                end_time = time.time()
                P=extract_data_dim("./graph_ssv.rlz",number_of_nodes)
                P=final_calcultate_dm(P)
                elapsed_time_1 = end_time - start_time
                start_time = time.time()
                # print(P)
                P=complete_isomap(heuristic,P,eps)
                P=final_calcultate_dm(P)
                end_time = time.time()
                elapsed_time_2 = end_time - start_time
                total_time = elapsed_time_1 + elapsed_time_2
                results[f'{heuristic}_time'] += (total_time)
                results[f'{heuristic}_MDE'] += (compute_measure("MDE",P,D,G))
                results[f'{heuristic}_LDE'] += (compute_measure("LDE",P,D,G))
                results[f'{heuristic}_RMSD'] += (compute_measure("RMSD",P,D,G))
            else:
                start_time = time.time()
                P=complete_isomap(heuristic,G,eps)
                P=final_calcultate_dm(P)
                end_time = time.time()
                total_time = end_time - start_time
                results[f'{heuristic}_time'] += (total_time)
                results[f'{heuristic}_MDE'] += (compute_measure("MDE",P,D,G))
                results[f'{heuristic}_LDE'] += (compute_measure("LDE",P,D,G))
                results[f'{heuristic}_RMSD'] += (compute_measure("RMSD",P,D,G))

    if results != {}:
        for u,heuristic in enumerate(heuristics):
            for v,mesure in enumerate(mesures):
                comparaison_results[v][u]=results[f'{heuristic}_{mesure}']/tries_number
        if verbose:
            print(f"comparaison_results={comparaison_results}")
        results = {}
    return comparaison_results


# for graph in ["Erdos Renyi","Barabasi_albert","random_regular_graph"]:
#     res=comparison_heuristics(50,15,1e-4,graph,True)
#     print(f'for graph {graph} we have the following results {res}')
print_verbose=True #set to False to avoid printing the results
res={}
for graph in ["random_regular_graph","Erdos Renyi","Barabasi_albert"]:
    res[f"isomap_{graph}"]=comparison_heuristics(50,15,1e-4,graph,print_verbose)
    open(f'./results_{graph}.txt', 'w').write(str(res[f"isomap_{graph}"]))
    print(f'for graph {graph} we have the following results {res[f"isomap_{graph}"]}')
print(res)

