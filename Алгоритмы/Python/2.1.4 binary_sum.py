def AddBinary(A,B):
    carry = 0 
    n = max(len(A),len(B))
    C = [0 for i in range(n+1)]
    for i in range(n):
        C[i] = (A[i] + B[i]+carry) % 2
        print(C[i])
        carry = (A[i] + B[i]+carry) // 2
    C[n] = carry
    return C


print(AddBinary([1,1,1,1,1,0,0,1,1,0],[1,1,1,1,1,0,0,1,1,1]))