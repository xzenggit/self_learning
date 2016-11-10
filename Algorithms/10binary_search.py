# Uses python3
import sys
'''
def binary_search(a, x):
    left, right = 0, len(a)
    # write your code here
    if right > left:
        mid = left + (right - left)//2
        if a[mid] == x:
            return mid
        elif a[mid] > x:
            tmp = binary_search(a[left:mid], x)
            if tmp != -1:
                return tmp + left
            else:
                return -1
        else:
            tmp = binary_search(a[mid+1:right], x)
            if tmp != -1:
                return tmp + mid +1
            else:
                return -1
    else:
        return -1
'''

def binary_search(a, x):
    left, right = 0, len(a)
    mid = 0
    while left <= right and (left < len(a)):
        mid = (right-left) // 2 + left
        if x == a[mid]:
            return mid
        if x > a[mid]:
            left = mid + 1
        else:
            right = mid-1
    return -1


def linear_search(a, x):
    for i in range(len(a)):
        if a[i] == x:
            return i
    return -1

if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    m = data[n + 1]
    a = data[1 : n + 1]
    for x in data[n + 2:]:
        # replace with the call to binary_search when implemented
        print(binary_search(a, x), end = ' ')
