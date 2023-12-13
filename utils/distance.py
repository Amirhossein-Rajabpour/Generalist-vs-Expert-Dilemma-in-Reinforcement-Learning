import numpy as np

def top_left(map):
    '''returns the coordinates of the top-most, left-most 1 in the map'''    
    for i in range(len(map)):
        for j in range(len(map[i])):
                if map[i][j] == 1:
                    return i, j
    
    return -1, -1
    

def distance(map1, map2):
    '''Returns the minimum number of edits required to transform map1 into map2.
    We assume that the maps are of the same size'''
    map1 = np.array(map1)
    map2 = np.array(map2)
        
    length = len(map1) # number of rows
    try:
        width = len(map1[0]) # number of columns
    except:
        return 0
    
    padded_map = np.zeros((3*length-2, 3*width-2))
    padded_map1 = padded_map.copy()
    padded_map2 = padded_map.copy()
    padded_map1[length-1:2*length-1, width-1:2*width-1] = map1
    padded_map2[length-1:2*length-1, width-1:2*width-1] = map2    
    
    min_edits = np.inf
    for i in range(len(padded_map2)-len(map1)+1):
        for j in range(len(padded_map2[0])-len(map1[0])+1):
            transfom = np.sum(map1) + np.sum(map2) - 2*np.sum(padded_map2[i:i+length, j:j+width] * map1)
            padded_map3 = padded_map.copy()
            padded_map3[i:i+length, j:j+width] = map1
            x1, y1 = top_left(padded_map1)
            x3, y3 = top_left(padded_map3)
            transfer = abs(x1-x3) + abs(y1-y3)
            edits = transfom + transfer
            if edits < min_edits:
                min_edits = edits

    return min_edits



