# Uses python3
import sys

def get_fibonaccihuge(n, m):
# calculate Fn mod m
    if (n <= 1):
        return n%m
    else:
        F = [0] * 1000
        F[1] = 1
        F[2] = 1
        i = 1
        while (F[i]%m != 0 or F[i+1]%m != 1) and i <= n:
            F[i+2] = F[i+1] + F[i]
            F.append(100)
            i = i+1
        if i == n:
            return F[n]%m
        else:
            return F[n%i]%m

if __name__ == '__main__':
    input = sys.stdin.read();
    n, m = map(int, input.split())
    print(get_fibonaccihuge(n, m))
