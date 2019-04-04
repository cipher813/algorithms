import itertools
import numpy as np

def generator(item_list):
    """Singly iterate through items in a list; helpful to iterate 
    through large datasets that can't fit into memory.
    
    generate = generator(item_list)
    next(generate)
    """
    for item in item_list:
        yield item

def isprime(number):
    if number==2:
        return True
    elif number%2==0 or number<2:
        return False
    else:
        for i in range(3,int(np.sqrt(number))+1,2):
            if number%i==0:
                return False
    return True

def isprime2(number):
    carmichael_list = [5,7,11,13]
    for div in carmichael_list:
        if number%div==0:
            return False
    return 2 in [number,pow(2, number, number)]

def isprime3(number):
    """Note that this misses carmichael numbers"""
    return 2 in [number,pow(2, number, number)]

functions = [isprime, isprime2] 

for func in functions:
    assert func(19)==True
    assert func(25)==False
    assert func(341)==False
    assert func(561)==False
    assert func(1105)==False
    assert func(41257781)==True
    assert func(2064991007)==True
    
def prime_list(max_number):
    primes = []
    i=2
    while len(primes)==0 or primes[-1]<max_number:
        if isprime(i):
            primes.append(i)
        i+=1
    return primes[:-1]

assert prime_list(30)==[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def find_primes(number, factors=[]):
    """Find prime factors of a number."""
    for i in range(2, number+1):
        if number%i==0:
            factors.append(i)
            number = number//i
            return find_primes(number,factors)
    return factors

assert find_primes(64,[])==[2]*6
assert find_primes(2062889050,[])==[2, 5, 5, 41257781]

def gcd(a, b):
    """Find greatest common denominator of two numbers."""
    a_primes = find_primes(a,[])
    b_primes = find_primes(b,[])
    c_primes = []
    
    for prime in a_primes:
        if prime in b_primes:
            popped = b_primes.pop(b_primes.index(prime))
            c_primes.append(popped)
    return np.prod(c_primes)

assert gcd(30,45)==15

def fibonacci(idx):
    a, b = 0, 1
    fib_list = []
    
    for i in range(idx):
        a, b = b, a + b
        fib_list.append(b)
        
    return fib_list

assert fibonacci(6)==[1, 2, 3, 5, 8, 13]

def multiply_binary(a,b):
    """Multiply two binary numbers
    Dasgupta Fig 2.1"""
    n = max(len(a)-2,len(b)-2)
    if n==1:
        return np.prod(a,b)
    
    a, b = a[2:], b[2:]
    al, ar = int("0b" + a[:len(a)//2],2), int("0b" + a[len(a)//2:],2)
    bl, br = int("0b" + b[:len(b)//2],2), int("0b" + b[len(b)//2:],2)
    p1 = al*bl
    p2 = ar*br
    p3 = (al+ar)*(bl+br)
    return bin(int(p1*2**n+(p3-p1-p2)*2**(n/2)+p2))

assert multiply_binary('0b10110110','0b10110110')=='0b1000000101100100'

def mergesort(arr):
    
    def merge(arr):
        print(f"1.\t{arr}")
        if len(arr)>1:
            mid = len(arr)//2
            l = arr[:mid]
            r = arr[mid:]

            merge(l)
            merge(r)

            i=j=k=0

            while i<len(l) and j<len(r):
                if l[i]<r[j]:
                    print(f"2.\tl[{i}]<r[{j}]: {l[i]}<{r[j]}")
                    arr[k]=l[i]
                    print(f"2.\tarr[{k}]=l[{i}]={arr[k]},{arr}")
                    i+=1
                else:
                    print(f"2.\tl[{i}]>r[{j}]: {l[i]}>{r[j]}")
                    arr[k]=r[j]
                    print(f"2.\tarr[{k}]=r[{j}]={arr[k]},{arr}")
                    j+=1
                k+=1

            # catches residuals not caught by the combined while above
            while i<len(l):
                arr[k]=l[i]
                print(f"3.\tarr[{k}]=l[{i}]={arr[k]},{arr}")
                i+=1
                k+=1

            while j<len(r):
                arr[k]=r[j]
                print(f"3.\tarr[{k}]=r[{j}]={arr[k]},{arr}")
                j+=1
                k+=1
    print(f"Unsorted array: {arr}")
    print(f"Applying MergeSort")
    merge(arr)
    print(f"Sorted array: {arr}")
    return arr
    
assert mergesort([9,1,8,2,12,7])==[1, 2, 7, 8, 9, 12]

def quicksort(arr):
    
    def partition(arr, low, high):
        i = low - 1
        pivot = arr[high]

        for j in range(low, high):
            if arr[j]<=pivot:
                i+=1
                arr[i],arr[j]=arr[j],arr[i]

        arr[i+1],arr[high]=arr[high],arr[i+1]
        return i+1

    def sort(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            sort(arr, low, pi - 1)
            sort(arr, pi+1, high)
            
    print(f"Unsorted:\t{arr}")
    sort(arr, 0, len(arr)-1)
    print(f"Sorted:\t{arr}")

def binary_search(arr,num):
    """Binary search and array to find the index of a value.
    
    :arr: sorted list
    :num: number to search for
    """
    mid = len(arr)//2
    if mid==0:
        return False
    elif arr[mid]==num:
        return True
    else:
        if arr[mid]<num:
            return binary_search(arr[mid:],num)
        else:
            return binary_search(arr[:mid],num)    
        
assert binary_search([7,8,9,10],8)==True
assert binary_search([7,8,9,10],11)==False
