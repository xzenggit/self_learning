# Uses python3
import sys

'''
def gcd(a, b):
    current_gcd = 1
    for d in range(2, min(a, b) + 1):
        if a % d == 0 and b % d == 0:
            if d > current_gcd:
                current_gcd = d

    return current_gcd
'''
def gcd(a, b):
    '''Use Euclid method to calcualte GCD '''
    if b = 0:
        return a
    ap = a % b
    return gcd(b, ap)

if __name__ == "__main__":
    input = sys.stdin.read()
    a, b = map(int, input.split())
    print(gcd(a, b))
