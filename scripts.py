# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else
if __name__ == '__main__':
    n = int(input().strip())
if n%2!=0:
    print("Weird")
if (n<=5 and n>=2 and n%2==0):
    print("Not Weird")
if (n>=6 and n<=20 and n%2==0):
    print("Weird")
if (n>20 and n%2==0):
    print("Not Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i)

# Write a function
def is_leap(year):
    leap = False
    # Write your logic here
    leap = (year%4==0) and ((year%100!=0) or (year%400==0))
    return leap

# Print Function
if __name__ == '__main__':
    n = int(input())
    s=""
    for i in range(1,n+1):
        s+=str(i)
    print(s)

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    l=sorted(arr,reverse=True)
    p=l[0]
    i=0
    while (l[i]==p):
        i+=1
    print(l[i])

# Nested Lists
if __name__ == '__main__':
    Nm=[]
    Sc=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        Nm.append(name)
        Sc.append(score)
    s=sorted(Sc)
    p=s[0]
    i=0
    while(s[i]==p):
        i+=1
    c=s.count(s[i])
    ris=[]
    st=0
    for j in range(c):
        pos=Sc.index(s[i],st)
        st=pos+1
        ris.append(Nm[pos])
    ris=sorted(ris)
    for k in ris:
        print(k)
        
# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    l=student_marks[query_name]
    m=sum(l)/3
    print("{:.2f}".format(m))

# Lists
if __name__ == '__main__':
    N = int(input())
    l=[]
    for i in range(N):
        l.append(input())
    ris=[]
    for j in l:
        c=j.split()
        if (c[0]=="insert"):
            ris.insert(int(c[1]),int(c[2]))
        if (c[0]=="print"):
            print(ris)
        if (c[0]=="remove"):
            ris.remove(int(c[1]))
        if (c[0]=="append"):
            ris.append(int(c[1]))
        if (c[0]=="sort"):
            ris.sort()
        if (c[0]=="pop"):
            ris.pop()
        if (c[0]=="reverse"):
            ris.reverse()
            
# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=(tuple(integer_list))
    print(hash(t))

# sWAP cASE
def swap_case(s):
    r=""
    for st in s:
        if(st.isupper()):
            r+=st.lower()
        elif (st.islower()):
            r+=st.upper()
        else:
            r+=st
    return r


# String Split and Join
def split_and_join(line):
    # write your code here
    ris=line.split(" ")
    ris="-".join(ris)
    return ris
    
# What's Your Name?
def print_full_name(first, last):
    ris="Hello "+first+" "+last+"! You just delved into python."
    print(ris)

# Mutations
def mutate_string(string, position, character):
    return string[:position]+character+string[position+1:]

# String Validators
if __name__ == '__main__':
    s = input()
    alnum=False
    alpha=False
    dig=False
    low=False
    upp=False
    for i in s:
        alnum=alnum or (i.isalnum())
        alpha=alpha or (i.isalpha())
        dig=dig or i.isdigit()
        low=low or i.islower()
        upp=upp or i.isupper()
    print(alnum)
    print(alpha)
    print(dig)
    print(low)
    print(upp)

# Text Alignment
thickness = int(input()) #This must be an odd number
c = 'H'
 #Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
 #Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
 #Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
 #Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
 #Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Find a string
def count_substring(string, sub_string):
    c=0
    for i in range(len(string)-len(sub_string)+1):
        if sub_string in string[i:i+len(sub_string)]:
             c+=1
    return c

# Text Wrap
def wrap(string, max_width):
    res=""
    i=0
    while i<(len(string)):
        res+=string[i:i+max_width]
        res+="\n"
        i+=max_width
    return res

# Designer Door Mat
i=input().split()
N=int(i[0])
M=int(i[1])
w="WELCOME"
pat=".|."
for i in range(N):
    ris=""
    if(i!=(N-1)/2):
        if(i<N/2):
            np=(i+1)*2-1
            nc=M-np*3
        else:
            nc=6*(i-(N-1)/2)
            np=(M-nc)/3
        ris+=int(nc/2)*"-"+pat*int(np)+int(nc/2)*"-"
        print(ris)
    else:
        nc=M-7
        ris+=int(nc/2)*"-"+w+int(nc/2)*"-"
        print(ris)

# String Formatting
def print_formatted(number):
    # your code goes here
    s=len(bin(number)[2:])
    for i in range(1,number+1):
        ris=""
        d=str(i)
        o=str(oct(i)[2:])
        h=str(hex(i)[2:]).upper()
        b=str(bin(i)[2:])
        ris=" "*(s-len(d))+d+" "*(s-len(o)+1)+o+" "*(s-len(h)+1)+h+" "*(s-len(b)+1)+b
        print(ris)

# The Minion Game
def minion_game(string):
    # your code goes here
    stu=0
    kev=0
    for i in range(len(string)):
        if (string[i] in ("A","E","I","O","U"))==False:
                stu+=len(range(len(string)-i))
        else:
                kev+=len(range(len(string)-i))
    if (stu>kev):
        print("Stuart "+str(stu))
    elif(kev>stu):
        print("Kevin "+str(kev))
    else:
        print("Draw")
        
# Merge the Tools!
def merge_the_tools(string, k):
    # your code goes here
    ris=[]
    u=[]
    i=0
    while i<(len(string)):
        ris.append(string[i:i+k])
        i+=k
    for j in ris:
        l=k-1
        while l>0:
            if(j[l] in j[:l]):
                j=j[:l]+j[l+1:]
            l-=1
        u.append(j)
    for p in u:
        print(p)
        
# Introduction to Sets
def average(array):
    # your code goes here
    a=array
    s=set(a)
    return round(sum(s)/len(s),3)

# Set .add()
N = int(input())
ris = set()
for i in range(N):
    ris.add(input())
print(len(ris))

# Symmetric Difference
M = int(input())
a = set(input().split())
N = int(input())
b = set(input().split())
c = map(int,list(a.difference(b).union(b.difference(a))))
for i in sorted(c):
    print(i)
    
# Set .discard(), .remove() & .pop()
n = int(input())
r = set(map(int, input().split()))
N=int(input())
for i in range(N):
    j=input().split()
    if j[0]=="pop":
        r.pop()
    if j[0]=="remove":
        r.remove(int(j[1]))
    if j[0]=="discard":
        r.discard(int(j[1]))
print(sum(r))

# Set .union() Operation
n = int(input())
e = set(map(int, input().split()))
m = int(input())
f = set(map(int, input().split()))
print(len(e.union(f)))

# Set .intersection() Operation
n = int(input())
e = set(map(int, input().split()))
m= int(input())
f = set(map(int, input().split()))
print(len(e.intersection(f)))

# Set .difference() Operation
n = int(input())
e = set(map(int, input().split()))
m = int(input())
f = set(map(int, input().split()))
print(len(e.difference(f)))

# Set .symmetric_difference() Operation
n = int(input())
e = set(map(int, input().split()))
m = int(input())
f = set(map(int, input().split()))
print(len(e.symmetric_difference(f)))

# Set Mutations
n = int(input())
A = set(map(int,input().split()))
N = int(input())

for i in range(N):
    j = input().split()
    B = set(map(int,input().split()))
    if j[0] == 'update':
        A.update(B)
    elif j[0] == 'intersection_update':
        A.intersection_update(B)
    elif j[0] == 'difference_update':
        A.difference_update(B)
    else:
        A.symmetric_difference_update(B)

print(sum(list(A)))

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    per=[[i,j,k] 
        for i in range(x+1)
            for j in range(y+1)
                for k in  range(z+1)
        ]
    ris = [ l for l in per if sum(l) != n]
    print(ris)
    
# Alphabet Rangoli
def print_rangoli(size):
    # your code goes here
    alfa=list(map(chr, range(97, 123)))
    a=alfa[:size]
    ris=""
    for i in range(1,size+1):
        l=a[size-i:size]
        l=l[::-1]+l[1:]
        r="-".join(l)
        ris+="-"*(size*2-2-2*(i-1))+r+"-"*(size*2-2-2*(i-1))+"\n"
    for i in range(1,size):
        l=a[i:size]
        l=l[::-1]+l[1:]
        r="-".join(l)
        ris+="-"*(2*i)+r+"-"*(2*i)+"\n"
    print(ris)

# Check Subset
N=int(input())
for i in range(N):
    na=int(input())
    a=set(input().split())
    nb=int(input())
    b=set(input().split())
    print(a.issubset(b))
    
# Check Strict Superset
a=set(input().split())
n=int(input())
ris=True
for i in range(n):
    b=set(input().split())
    ris=ris and b!=a and b.issubset(a)
print(ris)

# No Idea!
l=list(input().split())
s=list(map(int,input().split()))
A=set(map(int,input().split()))
B=set(map(int,input().split()))
ris=0
for i in s:
    if(i in A):
        ris+=1
    elif(i in B):
        ris+=-1
print(ris)

# collections.Counter()
x=int(input())
s=list(map(int,input().split())) 
n=int(input())
ris=0
for i in range(n):
    c= list(map(int, input().split()))
    if c[0] in s:
        ris+=c[1]
        s.remove(c[0])
print(ris)

# DefaultDict Tutorial
nm = list(map(int, input().split()))
n=nm[0]
m=nm[1]
A=[]
B=[]
for i in range(n):
    A.append(input())
for i in range(m):
    B.append(input())
for i in range(m):
    ris=""
    e=B[i]
    c=A.count(e)
    if c>0:
        k=0
        for j in range(c):
            pos=A.index(e,k)
            k=pos+1
            ris+=" "+str(k)
        print(ris[1:])
    else: 
        print(-1)
        
# Collections.namedtuple()
from collections import namedtuple
n=int(input())
stud=namedtuple('stud', " ".join(input().split()))
m=0
for i in range(n):
    ss=stud(*input().split())
    m+=int(ss.MARKS)
print(round(m/n,2))

# Collections.OrderedDict()
n=int(input())
l={}
I=[]
ris=""
for i in range(n):
    k=list(map(str,input().split()))
    name=" ".join(k[:len(k)-1])
    if name in I:
        l[name]+=int(k[len(k)-1])
    else:
        l[name]=int(k[len(k)-1])
        I.append(name)
for j in I:
    print(j,l[j])

# Word Order
n=int(input())
d={}
o=[]
for i in range(n):
    l=str(input())
    if l in d:
        d[l]+=1
    else:
        d[l]=1
        o.append(l)
print(len(o))
ris=""
for j in o:
    ris+=" "+str(d[j])
print(ris[1:])

# Collections.deque()
from collections import deque
n=int(input())
d=deque()
for i in range(n):
    l=list(map(str,input().split()))
    if l[0]=="append":
        d.append(int(l[1]))
    elif l[0]=="pop":
        d.pop()
    elif l[0]=="popleft":
        d.popleft()
    elif l[0]=="appendleft":
        d.appendleft(int(l[1]))
ris=""
for k in d:
    ris+=" "+str(k)
print(ris[1:])

# Piling Up!
t=int(input())
for i in range(t):
    n=int(input())
    l=list(map(int,input().split()))
    j=0
    ris="Yes"
    c=max(l[0],l[n-1])
    if c==l[0]:
        l.pop(0)
    else:
        l.pop(n-1)
    while j<n-1:
        if c<l[0] and c<l[len(l)-1]:
            ris="No"
            j=n
        elif l[0]<c and (l[0]>=l[len(l)-1] or l[len(l)-1] > c):
            c=l[0]
            l.pop(0)
        elif l[len(l)-1]<c and (l[0]<l[len(l)-1] or l[0]>c):
            c=l[len(l)-1]
            l.pop(len(l)-1)
        j+=1
    print(ris)
    
# Company Logo
if __name__ == '__main__':
    s = input()
    d={}
    for i in s:
        if i in d:
            d[i]+=1
        else:
            d[i]=1
    j=0
    while j<3:
        m=max(d.values())
        c=list(d.values()).count(m)
        l=[]
        for k in range(c):
            p=list(d.keys())[list(d.values()).index(m)]
            d.pop(p)
            l.append(p)
        l=sorted(l)[:3-j]
        for q in range(len(l)):
            print(l[q],m)
        j+=c
    
# Calendar Module
import calendar
mdy=list(map(int,input().split()))
day = calendar.weekday(mdy[2], mdy[0], mdy[1])
print(calendar.day_name[day].upper())

# Exceptions
T=int(input())
for i in range(T):
    try:
        l=list(map(int,input().split()))
        print (int(l[0])//int(l[1]))
    except ValueError as e:
        print ("Error Code:",e)
    except ZeroDivisionError as e:
        print ("Error Code:",e)
        
# Zipped!
I=list(map(int,input().split()))
n=I[0]
x=I[1]
R=[0]*x
l=[]
for i in range(x):
    l.append(map(float,input().split()))
L=zip(*l)
for j in L:
    print(round(sum(j)/x,1))

# Athlete Sort
arr.sort(key=lambda x: x[k])
for i in arr:
    ris=""
    for k in i:
        ris+=" "+str(k)
    print(ris[1:])
      
# ginortS
s=input()
l=[]
u=[]
o=[]
e=[]
for i in s:
    if i.isalpha():
        if i.isupper():
            u.append(i)
        else:
            l.append(i)
    else:
        if int(i)%2==0:
            e.append(i)
        else:
            o.append(i)
r="".join(sorted(l))+"".join(sorted(u))+"".join(sorted(o))+"".join(sorted(e))
print(r)

# Map and Lambda Function
cube = lambda x:x**3 # complete the lambda function
def fibonacci(n):
    # return a list of fibonacci numbers
    r=[]
    a=0
    b=1
    for i in range(n):
        r.append(a)
        c=a
        a=b
        b=c+b
    return r

# Detect Floating Point Number
n=int(input())
for i in range(n):
    l=input()
    try:
        j=float(l)
        print(j!=0)
    except:
        print("False")
        
# Re.split()
regex_pattern = r"[,.]"	# Do not delete 'r'.

# XML 1 - Find the Score
def get_attr_number(node):
    # your code goes here
    ris = 0
    for n in node.iter():
        ris+=len(n.attrib)
    return ris

# Arrays
def arrays(arr):
    # complete this function
    # use numpy.array
    a=numpy.array(arr,float)
    return a[::-1]

# Shape and Reshape
import numpy
l=numpy.array(list(map(int, input().split())))
print (numpy.reshape(l,(3,3)))

# Transpose and Flatten
import numpy
nm=list(map(int, input().split()))
n=nm[0]
m=nm[1]
ris=[]
for i in range(n):
    l = list(map(int, input().split()))
    ris.append(l)
ris= numpy.array(ris)
print (numpy.transpose(ris))
print (ris.flatten())

# Concatenate
import numpy

z=list(map(int, input().split()))

a=[]
for i in range(z[0]):
    a.append(list(map(int, input().split())))
b=[]
for i in range(z[1]):
    b.append(list(map(int, input().split())))
a=numpy.array(a)
b=numpy.array(b)
print(numpy.concatenate((a,b),axis=0))

# Zeros and Ones
import numpy
n=list(map(int, input().split()))
print(numpy.zeros((n),int))
print(numpy.ones((n),int))

# Eye and Identity
import numpy
numpy.set_printoptions(legacy = '1.13')
z=list(map(int,input().split()))
print (numpy.eye(z[0],z[1]))

# Array Mathematics
import numpy
z=list(map(int,input().split()))
a=[]
for i in range(z[0]):
    a.append(list(map(int,input().split())))
b=[]
for i in range(z[0]):
    b.append(list(map(int,input().split())))
a=numpy.array(a)
b=numpy.array(b)
print(numpy.add(a,b))
print(numpy.subtract(a,b))
print(numpy.multiply(a,b))
print(a//b)
print(numpy.mod(a,b))
print(numpy.power(a,b))

# Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
a=list(map(float,input().split()))
a=numpy.array(a)
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))

# Sum and Prod
import numpy
z=list(map(int,input().split()))
ris=[]
for i in range(z[0]):
    ris.append(list(map(int,input().split())))
ris=numpy.array(ris)
print (numpy.prod(numpy.sum(ris,axis=0)))

# Min and Max
z=list(map(int,input().split()))
a=[]
for i in range(z[0]):
    a.append(list(map(int,input().split())))
a=numpy.array(a)
print(numpy.max(numpy.min(a,axis=1)))

# Mean, Var, and Std
z=list(map(int,input().split()))
a=[]
for i in range(z[0]):
    a.append(list(map(int,input().split())))
a=numpy.array(a)
print(numpy.mean(a,axis=1))
print(numpy.var(a,axis=0))
print(round(numpy.std(a,axis=None),11))

# Dot and Cross
n=int(input())
a=[]
for i in range(n):
    a.append(list(map(int,input().split())))
a=numpy.array(a)
b=[]
for i in range(n):
    b.append(list(map(int,input().split())))
b=numpy.array(b)
print(numpy.dot(a,b))

# Inner and Outer
a=[]
a.append(list(map(int,input().split())))
a=numpy.array(a)
b=[]
b.append(list(map(int,input().split())))
b=numpy.array(b)
print(numpy.inner(a,b)[0][0])
print(numpy.outer(a,b))

# Polynomials
p=list(map(float,input().split()))
x=float(input())
print(numpy.polyval(p, x))

# Linear Algebra
n = int(input())
a=[]
for i in range(n):
    a.append(list(map(float, input().split())))
a=numpy.array(a)
print(round(numpy.linalg.det(a), 2))

# Birthday Cake Candles
def birthdayCakeCandles(candles):
    # Write your code here
    return candles.count(max(candles))

# Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    # Write your code here
    while x1<x2 and v1>v2:
        x1+=v1
        x2+=v2
    if (x1==x2):
        return "YES"
    else:
        return "NO"
    
# Viral Advertising
def viralAdvertising(n):
    # Write your code here
    l=0
    p=5
    for i in range(n):
        c=p//2
        l+=c
        p=c*3
    return l

# Recursive Digit Sum
def superDigit(n, k):
    # Write your code here
    ris=0
    if len(n)==1:
        return int(n)
    else:
        for i in n:
            ris+=int(i)
        ris=ris*k
        return superDigit(str(ris),1)
    
# Insertion Sort - Part 2
def insertionSort2(n, arr):
    # Write your code here
    for i in range(1,n):
        p=arr[i]
        j=i-1
        while j>-1 and arr[j]>p:
            arr[j+1]=arr[j]
            j+=-1
        arr[j+1]=p
        print(*arr)

# Capitalize!
def solve(s):
    ris=""
    for i in range(0,len(s)):
        if (i==0 & s[i].isalpha()) or (i>0 & s[i].isalpha() and s[i-1]==" "):
            ris+=s[i].capitalize()
        else:
            ris+=s[i]
    return ris

# The Captain's Room
n=int(input())
l=list(map(int,input().split()))
d={}
for i in l:
    if i in d:
        d[i]+=1
    else:
        d[i]=1
for key, value in d.items():
    if value != n:
        print(key)

# XML2 - Find the Maximum Depth
maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)
        
# Standardize Mobile Number Using Decorators 
def wrapper(f):
    def fun(l):
        # complete the function
        ris = []
        for i in l:
            ris.append('+91 '+i[-10:-5]+' '+ i[-5:])
        f(ris)
    return fun

# Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        # complete the function
            ris=sorted(people,key=lambda x: int(x[2]))
            return (f(i) for i in ris)
    return inner

# Group(), Groups() & Groupdict()
import re
n=input()
p = re.compile(r"([0-9a-zA-Z])\1")
ris=re.search(p,n)
if ris:
    print (ris.group()[0])
else:
    print(-1)
    
# Re.findall() & Re.finditer()
import re
n=input()
l=re.findall(r"(?<=[bcdfghjklmnpqrstvwxtyzBCDFGHKLMNPQRSTVWXYZ])([aeiouAEIOU]{2,})(?=[bcdfghjklmnpqrstvwxtyzBCDFGHKLMNPQRSTVWXYZ])",n)
if len(l)>0:
    for i in l:
        print(i)
else: 
    print(-1)
    
# Validating Roman Numerals
regex_pattern = r"^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	# Do not delete 'r'.

# Validating phone numbers
import re
n=int(input())
for i in range(n):
    ris="NO"
    if re.match("^[987]\d{9}$",input()):
        ris="YES"
    print(ris)
    
# Hex Color Code
import re
p = re.compile(r'(#[A-Fa-f0-9]{3}|#[A-Fa-f0-9]{6})\s*[,;)]')
n=int(input())
for i in range(n):
    m= re.findall(p, input())
    if m:
        for j in m:
            print(j)
            
# HTML Parser - Part 1
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for x,y in attrs:
            print("->", x, '>', y)
    def handle_endtag(self, tag):
        print('End   :', tag)
    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for x,y in attrs:
            print("->", x, '>', y)
p = MyHTMLParser()
N=int(input())
for i in range(N):
    p.feed(input())
    
# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data.count('\n') > 0:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        if data != '\n':
            print(data)
    def handle_data(self,data):
        if data!='\n':
            print(">>> Data")
            print(data)

# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for i in attrs:
            if len(i)<2:
                print("->", i[0])
            else:
                print("->", i[0],">", i[1])
N=int(input())
p = MyHTMLParser()
for i in range(N):
    ris=input()
    p.feed(ris)

# Validating UID
N=int(input())
for i in range(N):
    j=0
    n=0
    u=0
    r=[]
    s=input()
    ris=(len(s)==10) and s.isalnum()
    while (ris and j<10):
        if s[j] in r:
            ris=False
        else:
            r.append(s[j])
            if s[j].isupper():
                u+=1
            elif s[j].isnumeric():
                n+=1
            j+=1
    ris=ris and u>1 and n>2
    if ris==True:
        print("Valid")
    else:
        print("Invalid")
        
# Regex Substitution
import re
n=int(input())
for i in range(n):
    s=input()
    s1=re.sub(r'(?<= )(&&)(?= )',"and",s)
    s2=re.sub(r'(?<= )(\|\|)(?= )',"or",s1)
    print(s2)
    
# Validating Postal Codes
regex_integer_in_range = r"^[1-9]\d{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"	# Do not delete 'r'.

# Validating Credit Card Numbers
import re
n=int(input())
p=r"^([456])\d{3}\-*\d{4}\-*\d{4}\-*\d{4}"
for i in range(n):
    s=input()
    m=len(s)
    if re.match(p,s):
        if m!=16 and m!=19:
            print("Invalid")
        else:
            c=1
            mx=3
            for i in range(1,m):
                if (s[i]==s[i-1]) or (s[i-1]=="-" and s[i-2]==s[i]):
                    c+=1
                elif (s[i]!="-"):
                    c=1
                mx=max(c,mx)
            if mx<4:
                print("Valid")
            else:
                print("Invalid")       
    else:
        print("Invalid")
        
# Re.start() & Re.end()
import re
n=input()
m=input()
N=len(n)
M=len(m)
p=re.compile(m)
i=0
c=0
while i<=N-M:
    ris=p.search(n,i)
    if ris==None:
        i=N
    else:
        a=ris.start()
        print((a,a+M-1))
        c+=1
        i=a+1
if c<1:
    print((-1, -1))

# Validating and Parsing Email Addresses
import re
import email.utils
n=int(input())
for i in range(n):
    l=email.utils.parseaddr(input())
    e=l[1]
    p=r"^[a-zA-Z]([\w\.\-])+?@([a-zA-Z])+\.([a-zA-Z]){1,3}$"
    if re.match(p,e):
        print(email.utils.formataddr(l))
        
# Matrix Script
p=r"(?<=[a-zA-Z0-9])[\s!@#$%&]+(?=[a-zA-Z0-9])"
t=zip(*matrix)
ris=""
for i in t:
    ris+="".join(i)
ris=re.sub(p," ", ris)
print(ris)

#

#































































