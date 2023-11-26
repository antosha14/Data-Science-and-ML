import random

def generate_array(len):
    output = []
    for i in range(1,len):
        output.append(random.randint(1,100))
    return output
