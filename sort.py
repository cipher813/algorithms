"""
Sorting algorithms
April 2019
"""
def mergesort(arr):
    
    def merge(arr):
        if len(arr)>1:
            mid = len(arr)//2
            l = arr[:mid]
            r = arr[mid:]

            merge(l)
            merge(r)

            i=j=k=0

            while i<len(l) and j<len(r):
                if l[i]<r[j]:
                    arr[k]=l[i]
                    i+=1
                else:
                    arr[k]=r[j]
                    j+=1
                k+=1

            # catches residuals not caught by the combined while above
            while i<len(l):
                arr[k]=l[i]
                i+=1
                k+=1

            while j<len(r):
                arr[k]=r[j]
                j+=1
                k+=1
    merge(arr)
    return arr


def mergesort_printed(arr):
    
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
    print(f"Unsorted:\t{arr}")
    print(f"Applying MergeSort")
    merge(arr)
    print(f"Sorted:\t{arr}")
    return arr
    
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
            
    sort(arr, 0, len(arr)-1)
    return arr

def quicksort_printed(arr):
    
    def partition(arr, low, high):
        i = low - 1
        pivot = arr[high]
        print(f"Pivot=arr[{high}]={pivot}")

        for j in range(low, high):
            if arr[j]<=pivot:
                print(f"arr[{j}]<={pivot}")
                i+=1
                print(f"swapping arr[{i}] and arr[{j}]")
                arr[i],arr[j]=arr[j],arr[i]
                      
        print(f"swapping arr[{i+1}] and arr[{high}]")
        arr[i+1],arr[high]=arr[high],arr[i+1]
        return i+1

    def sort(arr, low, high):
        if low < high:
            print(f"{low}<{high}")
            print(f"pi=partition at {arr},{low},{high}")
            pi = partition(arr, low, high)
            print(f"Applying sort at {arr},{low},{pi - 1}")
            sort(arr, low, pi - 1)
            print(f"Applying sort at {arr},{pi + 1},{high}")
            sort(arr, pi+1, high)
            
    print(f"Unsorted:\t{arr}")
    print("Applying QuickSort")
    sort(arr, 0, len(arr)-1)
    print(f"Sorted:\t{arr}")
    return arr
  
functions = [mergesort, quicksort] 

for func in functions:
    assert func([9,1,8,2,12,7])==[1, 2, 7, 8, 9, 12]
    assert func([10, 7, 8, 9, 1, 5])==[1, 5, 7, 8, 9, 10]

