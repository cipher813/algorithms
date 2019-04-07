import os
import re
import requests
import itertools
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def download_url_to_filepath(filepath, url):
    """Create path and download data from url."""
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath) 
    else:
        print(f"{filepath} already exists.")
    return filepath

def parse_table_from_url(url):
    """Parse table from url."""
    response=requests.get(url)
    page=response.text
    soup=BeautifulSoup(page,"lxml")
    tables=soup.find_all("table")
    df = pd.read_html(tables[0].prettify(),index_col=[0])[0]
    df.columns = [re.sub(r'[\W+]','',str(x)) for x in df.columns]
    return df

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
