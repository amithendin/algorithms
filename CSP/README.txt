mmn_15, CSP solver for cryptarthimentic problems, by Amit Hendin

In order to solve the cryptarithmetic problems using CSP I did the follwing steps:
1. reduce the problem to CSP with only binary constraints in order to use AC3
2. run the backtracking search algorithm on the resulting CSP problem to get the solution

Step 1 reduce to CSP:
variables: for each letter we will create a variable and for each letter in the result
string we will create a carry variable. in addition, we will create 2 auxiliary variables
for each letter in the longer of the first two argument

domains: for each letter we will have the domain of base 10 digits, if the third argument
is longer than the previous two then it can only be longer by 1 letter and that letter will
necesarly have the domain [1] and the carry variable from the prevoius letter will also be 1
for each carry variable we will have the domain [0,1].
for adding the letter from the first arg to the letter from the second arg we will
have an auxilary variable with the domain of tuples of 3 elements where the first
element in each tuple is 0 or 1 and the rest are 0...9.
for adding the letter from the third arg to the carry*10 we will have an auxiliary variable
with the domain of tuples of 2 elements where the first element in each tuple is 0 or 1 and
the second is 0...9

constraints: for each two letters a,b we need to add we will have the following constraints;
prev_carry = aux_a[0]
a = aux_a[1]
b = aux_a[2]
and for their result c we will have the constraints;
c = aux_b[0]
next_carry = aux_b[1]
where prev/next_carry are the carry variables from the previous/next two letters. Then for
the auxiliary variables we will have the constraints
aux_a[0] + aux_a[1] + aux_a[2] = aux_b[0] + 10*aux_b[1]
which all together will give us the desired 5-ary constraint made of binary constraints
prev_carry + a + b = c + 10*next_carry

Step 2 backtracking search:
we implemented the backtracking search as specified in the book, in the method backtracking_search
we ran AC3 preproccesing on the CSP to reduce the domains of the variables.
Then we ran the backtrack method and in the select unassignd variable method we use the MRV hueristic
with degree hueristic where two vairables have the same level of constraint, and in the order domain
values step we used the LCV hueristic.

let's take for example the problem
 SEND
+MORE
-----
MONEY

for this we will generate the following variables
S,E,N,D,M,O,R,Y,C0,C1,C2,C3,C4,aux0,aux1,aux2,aux3,aux4,aux5,aux6,aux7

where the domain of C0 is [0]. and we will generate the constraints
C0 + D + E = Y + 10*C1
C1 + N + R = E + 10*C2
C2 + E + O = N + 10*C3
C3 + S + M = 0 + 10*C4
M = 1, C4=1

and after running our backtracking-search algorithm on it we will end up with the result.
sample runs:
(you can pass a 4th argument no_print to disable printing all the steps)

command: python csp.py SEND MORE MONEY
M=1 O=0 S=9 E=5 N=6 D=7 Y=2 R=8

command: python csp.py TWO TWO FOUR
result: F=1 T=7 O=5 W=6 U=3 R=0

command: python csp.py A B BC
result: B=1 C=0 A=9

command: python csp.py A B C
result: C=3 A=1 B=2

command: python csp.py A BA CC
result: C=2 B=1 A=6

command: python csp.py ON TW THR no_print
result: T=1 N=3 H=0 R=2 W=9 O=8

command: python csp.py A A AB no_print
result: No solution found