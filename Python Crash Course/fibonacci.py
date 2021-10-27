import time
import functools

def fibonacci_naive(n):
    if n < 3:
        return 1
    else:
        return fibonacci_naive(n-1) + fibonacci_naive(n-2)

start = time.time()
print(fibonacci_naive(30))
end = time.time()

print(end-start)

@functools.lru_cache
def fibonacci_fast(n):
    if n < 3:
        return 1
    else:
        return fibonacci_fast(n-1) + fibonacci_fast(n-2)

start_fast = time.time()
print(fibonacci_fast(100))
end_fast = time.time()

print(end_fast-start_fast)