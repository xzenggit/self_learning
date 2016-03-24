# Uses python3
import sys

def get_majority_element(a, n):
    tmp = {}
    for x in a:
        if x not in tmp:
            tmp[x] = 1
        else:
            tmp[x] += 1
    how_many = int(n/2)
    for x in tmp.keys():
        if tmp[x] > how_many:
            return x
    return -1


if __name__ == '__main__':
    input = sys.stdin.read()
    n, *a = list(map(int, input.split()))
    if get_majority_element(a, n) != -1:
        print(1)
    else:
        print(0)
