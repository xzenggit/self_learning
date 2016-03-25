# Python program for Memoized version of nth Fibonacci number
 
# function to calcualte nth Fibonacci number
def fib(n, lookup):
 
    # Base case
    if n == 0 or n == 1 :
        lookup[n] = n
 
    # If the value is not calculated previously then calculate it
    if lookup[n] is None:
        lookup[n] = fib(n-1 , lookup)  + fib(n-2 , lookup) 
 
    # return the value corresponding to that value of n
    return lookup[n]
# end of function
# This code is contributed by Nikhil Kumar Singh
