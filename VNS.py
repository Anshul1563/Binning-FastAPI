import random
import time
import copy
import uuid
import sys
import pandas as pd

#-------------------------------- Initialization --------------------------------

df = pd.read_csv("PLID_Bookings_Problem1.csv")

df = df.drop(['Fiscal Month',"Booking_Date"], axis=1)
df['Fiscal Quarter'] = df['Fiscal Quarter'].str.split(' ').str[::-1].str.join(' ')
df['Booked_Qty'] = df.groupby(['Product Family', 'Fiscal Quarter'])['Booked_Qty'].transform('sum')
df = df.drop_duplicates(subset=['Product Family', 'Fiscal Quarter'])
df = df.sort_values(by=['Business Unit','Fiscal Quarter', 'Booked_Qty'], ascending=[True, True, False])
df = df.reset_index(drop=True)
df = df[df["Fiscal Quarter"] == "FY2012 Q3"]

itemsInfo = {}
for index, row in df.iterrows():
    itemsInfo[row["Product Family"]] = row["Booked_Qty"]


binsInfo = {}
capacitites = [0,4000, 25000,100000, 300000, 700000]
costs = list(map(lambda x : x,capacitites))
probabilites = [0,]
maxTime = 5


for x in range(1,len(capacitites)):
    probabilites.append(capacitites[x]/costs[x])

for x in range(1, len(capacitites)):
    binsInfo[x] = {
        "cost": costs[x],
        "capacity": capacitites[x],
        "probability":  probabilites[x]
    }

#----------------------------------------------------------------------------------


def Cost(x : dict):
    totalCost = 0
    for info in x.values() :
        totalCost += binsInfo[info["bin_type"]]["cost"]
    return totalCost

def RemainingCapacity(x,id):
    capacity = binsInfo[x[id]["bin_type"]]["capacity"]
    weight = 0
    if len(x[id]["item_ids"]) == 0 :
        return capacity
    for itemId in x[id]["item_ids"]:
        weight += itemsInfo[itemId]
    return capacity - weight

def GetWeight(id):
    return itemsInfo[id]

def UnpackRepack(x : dict,binId,newCapacity) :
    
    cap = newCapacity
    unpackedItems = []

    while cap < 0 :
        unpackedItem = random.choice(x[binId]["item_ids"])
        unpackedItems.append(unpackedItem)
        x[binId]["item_ids"].remove(unpackedItem)
        cap += itemsInfo[unpackedItem]
    x[binId]["remaining_capacity"] = cap

    if (len(x[binId]["item_ids"]) == 0) :
        del x[binId]

    unpackedItems.sort(key=GetWeight,reverse=True)
    for id in unpackedItems:
        validBins = list(filter(lambda bin: x[bin]["remaining_capacity"] > itemsInfo[id], x.keys()))
        if (len(validBins) == 0):
            newId = uuid.uuid4()
            validBinTypes = list(filter(lambda bin: binsInfo[bin]["capacity"] > itemsInfo[id], binsInfo))
            validProbabilites = tuple(map(lambda bin: binsInfo[bin]["probability"], validBinTypes))
            selectedBin = random.choices(validBinTypes, weights=validProbabilites, k=1)[0]
            remainingCapacity = binsInfo[selectedBin]["capacity"] - itemsInfo[id]
            x[newId] = {"bin_type": selectedBin, "item_ids": [id], "remaining_capacity": remainingCapacity}
        else :
            validProbabilites = tuple(map(lambda bin: x[bin]["remaining_capacity"], validBins))
            # print(f"Item : {id} {itemsInfo[id]} | Bins : {validBins} {validProbabilites}")
            selectedBin = random.choices(validBins, weights=validProbabilites, k=1)[0]
            x[selectedBin]["item_ids"].append(id)
            x[selectedBin]["remaining_capacity"] = RemainingCapacity(x, selectedBin)
            # print(f"{selectedBin} {x[selectedBin]['remaining_capacity']} ")


    return x

NO_PARENT = -1
 
def dijkstra(adjacency_matrix, start_vertex):
    n_vertices = len(adjacency_matrix[0])
 
    # shortest_distances[i] will hold the
    # shortest distance from start_vertex to i
    shortest_distances = [sys.maxsize] * n_vertices
 
    # added[i] will true if vertex i is
    # included in shortest path tree
    # or shortest distance from start_vertex to
    # i is finalized
    added = [False] * n_vertices
 
    # Initialize all distances as
    # INFINITE and added[] as false
    for vertex_index in range(n_vertices):
        shortest_distances[vertex_index] = sys.maxsize
        added[vertex_index] = False
         
    # Distance of source vertex from
    # itself is always 0
    shortest_distances[start_vertex] = 0
 
    # Parent array to store shortest
    # path tree
    parents = [-1] * n_vertices
 
    # The starting vertex does not
    # have a parent
    parents[start_vertex] = NO_PARENT
 
    # Find shortest path for all
    # vertices
    for i in range(1, n_vertices):
        # Pick the minimum distance vertex
        # from the set of vertices not yet
        # processed. nearest_vertex is
        # always equal to start_vertex in
        # first iteration.
        nearest_vertex = -1
        shortest_distance = sys.maxsize
        for vertex_index in range(n_vertices):
            if not added[vertex_index] and shortest_distances[vertex_index] < shortest_distance:
                nearest_vertex = vertex_index
                shortest_distance = shortest_distances[vertex_index]
 
        # Mark the picked vertex as
        # processed
        added[nearest_vertex] = True
 
        # Update dist value of the
        # adjacent vertices of the
        # picked vertex.
        for vertex_index in range(n_vertices):
            edge_distance = adjacency_matrix[nearest_vertex][vertex_index]
             
            if edge_distance > 0 and shortest_distance + edge_distance < shortest_distances[vertex_index]:
                parents[vertex_index] = nearest_vertex
                shortest_distances[vertex_index] = shortest_distance + edge_distance
 
    return print_solution(start_vertex, shortest_distances, parents)
 
 
# A utility function to print
# the constructed distances
# array and shortest paths
def print_solution(start_vertex, distances, parents):
    n_vertices = len(distances)
    # print("Vertex\t Distance\tPath")
     
    for vertex_index in range(n_vertices):
        if vertex_index != start_vertex:
            # print("\n", start_vertex, "->", vertex_index, "\t\t", distances[vertex_index], "\t\t", end="")
            vertices = print_path(vertex_index, parents,[])
    return vertices
 
 
# Function to print shortest path
# from source to current_vertex
# using parents array
def print_path(current_vertex, parents,vertices):
    # Base case : Source node has
    # been processed
    if current_vertex == NO_PARENT:
        return vertices
    vertices = print_path(parents[current_vertex], parents,vertices)
    # print(current_vertex, end=" ")
    vertices.append(current_vertex)
    return vertices


def BestFit(itemIds):
    w = 0
    for id in itemIds:
        w += itemsInfo[id]
    cost = 0
    capacity = 99999999
    for c in list(binsInfo.values()):
        if (c["capacity"] > w and c["capacity"] < capacity):
            capacity = c["capacity"]
            cost = c["cost"]
    return cost

def GetBin(x,itemId):
    for id,info in x.items():
        if itemId in info["item_ids"]:
            info["bin_id"] = id
            return info

def GetCapacity(bin):
    return bin["remaining_capacity"]

def ChangeBinType(x):
    for id,info in x.items():
        remainingCapacity = info["remaining_capacity"]
        binSize = binsInfo[info["bin_type"]]["capacity"]
        for type,binInfo in binsInfo.items():
            if binSize - remainingCapacity < binInfo["capacity"] and binSize > binInfo["capacity"]:
                x[id]["bin_type"] = type
                x[id]["remaining_capacity"] = RemainingCapacity(x,id)
                # print(type, binInfo["capacity"], remainingCapacity, binSize, RemainingCapacity(x, id), binInfo["capacity"] - binSize + remainingCapacity)
    return x




def GenerateInitialSolution() :
    solution = {}
    s = 1
    for id,w in itemsInfo.items():
        newId = uuid.uuid4()
        validBinTypes = list(filter(lambda bin : binsInfo[bin]["capacity"] > w,binsInfo ))
        validProbabilites = tuple(map(lambda bin: binsInfo[bin]["probability"], validBinTypes))
        selectedBin = random.choices(validBinTypes,weights=validProbabilites,k=1)[0]
        remainingCapacity = binsInfo[selectedBin]["capacity"] - itemsInfo[id] 
        solution[newId] = {"bin_type": selectedBin, "item_ids": [id], "remaining_capacity": remainingCapacity}
        s+= 1
    return solution

def ChooseNeightbourhood(c1,c2,c3) :
    k = random.choices([1,2,3],weights=(c1,c2,c3),k=1)[0]
    return k

def Shaking(x, k):
    if (k == 1):
        shook = N1(x)
    if (k == 2):
        shook = N2(x)
    else:
        shook = N3(x)
    return shook


def LocalSearch(x):
    x3 = N4(x)
    x4 = N5(x3)
    x5 = N6(x4)
    x6 = ChangeBinType(x5)
    return x6


#-------------------------------- Operators --------------------------------

def N1(x: dict):
    binIds = list(x.keys())
    selected_length = int(len(binIds)*0.15)
    bin_samples = random.sample(binIds, selected_length)
    for binId in bin_samples:
        newBinType = random.choice(
            [i for i in range(1, len(capacitites)) if i != x[binId]["bin_type"]])
        x[binId]["bin_type"] = newBinType
        newCapacity = RemainingCapacity(x, binId)
        x[binId]["remaining_capacity"] = newCapacity

        if newCapacity < 0:
            x = UnpackRepack(x, binId, newCapacity)
    return x


def N2(x):
    binIds = list(x.keys())
    selected_length = int(len(binIds)*0.05)
    bin_samples = random.sample(binIds, selected_length)
    for binId in bin_samples:
        itemIds = x[binId]["item_ids"]
        del x[binId]
        for id in itemIds:
            newId = uuid.uuid4()
            w = itemsInfo[id]
            validBinTypes = list(
                filter(lambda bin: binsInfo[bin]["capacity"] > w, binsInfo))
            validProbabilites = tuple(
                map(lambda bin: binsInfo[bin]["probability"], validBinTypes))
            selectedBin = random.choices(
                validBinTypes, weights=validProbabilites, k=1)[0]
            remainingCapacity = binsInfo[selectedBin]["capacity"] - \
                itemsInfo[id]
            x[newId] = {"bin_type": selectedBin, "item_ids": [
                id], "remaining_capacity": remainingCapacity}
    return x


def N3(x):
    items = []
    alt = 0
    for bin in x.values():
        alt += 1
        itemIds = bin["item_ids"]
        itemIds.sort(key=GetWeight, reverse = (alt % 2 == 0))
        items +=itemIds

    J = list(range(len(items)))
    edges = []
    for j in J:
        edges.append((j, j+1, BestFit([items[j]])))
    for j in J:
        for k in range(j+2,len(items)+1):
            if BestFit(items[j:k]) != 0:
                edges.append((j, k, BestFit(items[j:k])))
    

    num_vertices = max(max(edge[0], edge[1]) for edge in edges) + 1
    adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]
    
    paths = {}
    for edge in edges:
        paths[(edge[0],edge[1])] = edge[2]
        adj_matrix[edge[0]][edge[1]] = edge[2]
    vertices = dijkstra(adj_matrix, 0)
    binList = {}
    for i in range(1,len(vertices)):
        itemIds = []
        for q in range(vertices[i - 1],vertices[i]):
            itemIds.append(items[q])
        binList[f"bin{i}"] = {"item_ids" : itemIds}
        binType = 0
        for type,info in binsInfo.items() :
            if info["cost"] == paths[(vertices[i - 1], vertices[i])]:
                binType = type
        binList[f"bin{i}"]["bin_type"] = binType
        binList[f"bin{i}"]["remaining_capacity"] = RemainingCapacity(binList, f"bin{i}")
    return binList

def N4(x : dict):
    itemStuff = {}
    for id,info in x.items() :
        for itemId in info["item_ids"]:
            itemStuff[itemId] = {"bin_id": id, "remaining_capacity": info["remaining_capacity"]}
    for id in itemsInfo :
        binA = GetBin(x,id) 
        validBins = list(filter(lambda bin: x[bin]["remaining_capacity"] > itemsInfo[id] and x[bin]["remaining_capacity"] - itemsInfo[id] < binA["remaining_capacity"] and bin != binA["bin_id"], x.keys()))
        validBinInfo = list(map(lambda bin: {"remaining_capacity": x[bin]["remaining_capacity"], "bin_id": bin}, validBins))
        validBinInfo.sort(key=GetCapacity)
        if len(validBinInfo) == 0 :
            continue 
        binB = validBinInfo[0]
        # print(itemsInfo[id],binA, binB)

        x[binA["bin_id"]]["item_ids"].remove(id)
        x[binA["bin_id"]]["remaining_capacity"] = RemainingCapacity(x, binA["bin_id"])
        if (len(x[binA["bin_id"]]["item_ids"]) == 0):
            del x[binA["bin_id"]]

        x[binB["bin_id"]]["item_ids"].append(id)
        x[binB["bin_id"]]["remaining_capacity"] = RemainingCapacity(x, binB["bin_id"])
        
        # print("After Shift :")
        # print(x[binA["bin_id"]], x[binB["bin_id"]])
        # print()
    return x

def N5(x : dict):
    itemStuff = {}
    for id,info in x.items() :
        for itemID in info["item_ids"]:
            itemStuff[itemID] = {"bin_id": id, "remaining_capacity": info["remaining_capacity"]}
    for itemId in itemsInfo:
        # print(itemId,x)
        binA = GetBin(x,itemId)
        aRemainingCApacity = binA["remaining_capacity"]
        validIds = {}
        for id,info in x.items():
            itemIds = info["item_ids"]
            bRemainingCapacity = info["remaining_capacity"]
            itemIds.sort(key=GetWeight,reverse=True)
            swappedId = ""
            for sId in itemIds:
                # print(itemsInfo[sId], bRemainingCapacity + itemsInfo[sId], itemsInfo[itemId])
                if bRemainingCapacity + itemsInfo[sId] > itemsInfo[itemId] and itemsInfo[itemId] > itemsInfo[sId] and bRemainingCapacity + itemsInfo[sId] - itemsInfo[itemId] < aRemainingCApacity:
                    swappedId = sId
            if swappedId == "" :
                continue
            # print(
            #     f"SWAP : {swappedId} {itemsInfo[swappedId]} | Item : {itemId} {itemsInfo[itemId]}   ")
            validIds[swappedId] =  bRemainingCapacity +itemsInfo[swappedId] - itemsInfo[itemId]
        # print("..................")
        finalId = ""
        for ab in validIds.keys() :
            if finalId == "":
                finalId = ab
                continue
            if validIds[finalId] > validIds[ab]:
                finalId = ab
        # print(validIds,finalId)
        if (finalId == ""):
            continue
        binB = GetBin(x,finalId)
        # print(f"Swapping {x[binA['bin_id']]} and {x[binB['bin_id']]}")
        # print(f"binA item : {itemId} {itemsInfo[itemId]} | binB item : {finalId} {itemsInfo[finalId]}")
        # print(f"binA capacity : {binA['remaining_capacity']} binB capacity : {binB['remaining_capacity']}  ")

        x[binA["bin_id"]]["item_ids"].remove(itemId)
        x[binA["bin_id"]]["item_ids"].append(finalId)
        x[binA["bin_id"]]["remaining_capacity"] = RemainingCapacity(x, binA["bin_id"])
        
        x[binB["bin_id"]]["item_ids"].remove(finalId)
        x[binB["bin_id"]]["item_ids"].append(itemId)
        x[binB["bin_id"]]["remaining_capacity"] = RemainingCapacity(x, binB["bin_id"])
        # print(f"new binA capacity : {x[binA['bin_id']]['remaining_capacity']} new binB capacity : {x[binB['bin_id']]['remaining_capacity']}")
    return x

def N6(x):
    high = 0
    removeId = ""
    for id,info in x.items():
        if info["remaining_capacity"] > high:
            removeId = id
            high = info["remaining_capacity"]
    itemIds = x[removeId]["item_ids"]

    for itemId in itemIds:
        low = 9999
        addId = ""
        weight = itemsInfo[itemId]
        for id,info in x.items():
            if (info["remaining_capacity"] < low and info["remaining_capacity"] > weight):
                addId = id
                low = info["remaining_capacity"]
        
        if (addId == ""):
            continue
        x[removeId]["item_ids"].remove(itemId)
        x[removeId]["remaining_capacity"] = RemainingCapacity(x, removeId)

        x[addId]["item_ids"].append(itemId)
        x[addId]["remaining_capacity"] = RemainingCapacity(x, addId)
    
    if len(x[removeId]["item_ids"]) == 0:
        del x[removeId]

    return x


def ValidateSolution(x : dict):
    wastedSpace = 0
    for id,info in x.items():
        wastedSpace += info["remaining_capacity"]
        print(f"Bin type : {info['bin_type']} | Item ID : {info['item_ids']} | Remaining Capacity :{info['remaining_capacity']}")
        weight = 0
        for itemId in info['item_ids'] :
            weight += itemsInfo[itemId]
        capacity = binsInfo[info['bin_type']]['capacity']
        # print(f"Calculated Weight {weight} | Calculated Capacity {capacity} | Calculated Remaining Capacity : {capacity - weight}")
        # print("")
    print(f"Wasted Space : {wastedSpace}")


def Waste(x: dict):
    wastedSpace = 0
    for id,info in x.items():
        wastedSpace += info["remaining_capacity"]
    return wastedSpace

def VNS():
    x = GenerateInitialSolution()
    finish = False
    c1, c2, c3 = 0, 0, 0
    total = 0
    startTime = time.time()
    while not finish:
        total +=1
        k = ChooseNeightbourhood(c1 + 1, c2 + 1, c3 + 1)
        oldX = copy.deepcopy(x)
        x1 = Shaking(x, k)
        x2 = LocalSearch(x1)
        print(f"Iteration {total} | OC : {Cost(oldX)} NC : {Cost(x2)} N : {k} W : {Waste(x2)} P : {c1} {c2} {c3}")
        if (Cost(x2) <= Cost(oldX) ):
            x = x2
            if Cost(x2) < Cost(oldX):
                if (k == 1):
                    c1 += 1
                if (k == 2):
                    c2 += 1
                if (k == 3):
                    c3 += 1
                updatedTime = time.time()

        else:
            x = oldX
        if (time.time() - startTime > maxTime):
            finish = True
    totalTime = updatedTime - startTime
    totalBins = len(x.keys())
    print("Total Improved Iterations : ", c1+c2+c3)
    print(f"Total Bins Used {totalBins}")
    print(f"Final Optimised Cost {Cost(x)}")
    print(f"Last updated time : {totalTime}")
    print("")
    ValidateSolution(x)

    solution = {}
    solution["Bin Types"] = binsInfo
    solution["Items"] = itemsInfo
    solution["Bins"] = x
    solution["Total Bins"] = totalBins
    solution["Cost"] = Cost(x)
    solution["Wasted Space"] = Waste(x)
    return solution


