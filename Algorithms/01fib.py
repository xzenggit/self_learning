# Uses python3
# Compute Fibanacci Number
def calc_fib(n):
    if (n <= 1):
        return n
    F = [0] * n
    F[1] = 1
    for i in range(2, n):
        F[i] = F[i-1] + F[i-2]
    return F[n-1]

n = int(input())
print(calc_fib(n))
