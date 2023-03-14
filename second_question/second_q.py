import numpy as np

M = int(input("M: "))
N = int(input("N: "))

def get_map(M:int,N:int) -> list:
    is_map = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            is_map[i][j] = int(input(f"Введіть число в позиції[{i}][{j}]: "))
    return is_map

visited = []
plan = get_map(M,N)
res = 0

def check_visited(i:int,j:int):
    visited.append((i,j))
    X = [i+1, i, i-1, i]
    Y = [j, j+1, j, j-1]
    for i in range(4):
        x = X[i]
        y = Y[i]
        if x in range(M) and y in range(N) and plan[x][y] == 1 and (x,y) not in visited:
            check_visited(x,y)


for i in range(M):
    for j in range(N):
        if plan[i][j] == 1 and (i,j) not in visited:
            res += 1
            check_visited(i,j)

print(f"Відповідь: {res}")
