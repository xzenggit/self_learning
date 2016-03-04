# Uses python3
import sys

def calc_fib(n):
# calculae Fibonacci Number
if (n <= 1):
        return n
    F = [0] * n
    F[1] = 1
    for i in range(2, n):
        F[i] = F[i-1] + F[i-2]
    return F[n-1]

def get_fibonaccihuge(n, m):
# calculate Fn mod m
    if (n <= 1):
        return n
    F = [0] * n
    F[1] = 1
    for i in range(2, n):
        F[i] = (F[i-1] + F[i-2]) % 10
    return F[n-1]

if __name__ == '__main__':
    input = sys.stdin.read();
    n, m = map(int, input.split())
    print(get_fibonaccihuge(n, m))
