# Uses python3
import sys

def get_change(n):
    # given a mount of money, calcualte the smallest amount of coins for change.
    change = [10, 5, 1]
    m = 0
    for i in range(3):
        while n >= change[i]:
            n = n - change[i]
            m = m + 1
    return m

if __name__ == '__main__':
    n = int(sys.stdin.read())
    print(get_change(n))
