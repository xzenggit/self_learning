# Uses python3
import sys

def get_optimal_value(capacity, weights, values):
    value = 0.
    # greedy algorithm
    uval = [0] * len(values)
    for i in range(len(values)):
        uval[i] = values[i]/weights[i]
    tmp = sorted(uval, reverse=True)
    w = 0
    for i in range(len(values)):
        w = 0
        while capacity > 0 and w < weights[uval.index(tmp[i])]:
            capacity -= 1
            w += 1
            value += tmp[i]
    return value


if __name__ == "__main__":
    data = list(map(int, sys.stdin.read().split()))
    n, capacity = data[0:2]
    values = data[2:(2 * n + 2):2]
    weights = data[3:(2 * n + 2):2]
    opt_value = get_optimal_value(capacity, weights, values)
    print("{:.10f}".format(opt_value))
