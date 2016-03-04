# Uses python3
import sys

def gcd(a, b):
    '''Use Euclid method to calcualte GCD '''
    if b = 0:
        return a
    ap = a % b
    return gcd(b, ap)

def lcm(a, b):
    '''LCD(a,b) = a * b / gcd(a,b)'''
    return a*b/gcd(a,b)

if __name__ == '__main__':
    input = sys.stdin.read()
    a, b = map(int, input.split())
    print(lcm(a, b))

