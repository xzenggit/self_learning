# Uses python3
import sys

def optimal_summands(n):
    summands = []
    #write your code here
    k = n
    l = 1
    while k >= l:
        if k <= 2*l:
            summands.append(k)
            return summands
        else:
            summands.append(l)
            k = k-l
            l = l+1

    return summands

if __name__ == '__main__':
    input = sys.stdin.read()
    n = int(input)
    summands = optimal_summands(n)
    print(len(summands))
    for x in summands:
        print(x, end=' ')
