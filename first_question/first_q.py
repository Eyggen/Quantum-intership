N = int(input("N = "))

######First example##########
def using_for_loop(N:int) -> int:
    res = 0
    for i in range(N):
        res += i+1
    print(res)

using_for_loop(N)

######Second example##########
def using_formula(N:int) -> int:
    print((N*(N+1))/2)

using_formula(N)
