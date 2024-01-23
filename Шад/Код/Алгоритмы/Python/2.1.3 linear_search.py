from array_generation import generate_array
searched_index = None

def linear_search(k, v):
    global searched_index
    for i in range(1,len(k)):
        if k[i]==v:    
            searched_index = i
            break
    return searched_index

arr = generate_array(100)
print(arr)
print(linear_search(arr,14))