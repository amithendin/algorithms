'''
Cryptarithmentic solver
by: Amit Hendin
course: 20551 Intro to Artificial Intelligence Open University of Israel

the following program solves cryptarthimetic problems where letters
are positive integers, input the two words to add as the first two paramters
and their result as the last.

the program solves the puzzle by reducing the problem to a Constraint Satisfaction Problem
and the solving it using the backtracking-search algorithm as
mentioned in (page 176) Artificial Intelligence: A Modern Approach 4th edition, Pearson;
combined with AC3 preprocessing of problem and MRV and LCV heuristics in the SELECT-UNASSIGNED-VARIABLE
and ORDER-DOMAIN-VALUES functions respectivley
'''
import sys

#flag to enable detailed prints
ENABLE_PRINT = True

#input: constraint, values a and b
#output: True if a and b satify the constraint otherwise False
#raises exception if undefined constraint is provided
def resolveConstraint(constraint, a, b):
    if constraint == "Eq": return a==b
    if constraint == "Neq": return a!=b
    if constraint.startswith("Elem"):
        return b[int(constraint[4:])] == a #take the element number from the string and use it as index
    if constraint == "Sum": #implement the equation constraint
        return a[0] + a[1] + a[2] == b[0] + 10*b[1] #C0 + A + B = C + 10*C1
    raise "undefined constraint: "+constraint

#input: list of constraints, values a and b
#output: returns True is a and b satisfy all constraints
#otherwise returns False
def resolveConstraints(constraints, a, b):
    for constraint in constraints:
        #check if constraint is statisfied by a and b
        if not resolveConstraint(constraint, a, b):
            #if ENABLE_PRINT: print("failed on constraint", constraint)
            return False
    return True

#input: constraints dictionary, variables a and b, a constraint
#output: adds the constraint to the constraints dictionary over
#the provided variables
def addConstraint(contr, a, b, constraint):
    if not((a,b) in contr):
        contr[(a,b)] = set()
    contr[(a,b)].add(constraint)

#input: first, second, and third arguments of a cryptarithmetic puzzle
#output: the CSP reduction of the problem consisting of set of variables,
#dictionary of domains for each variable, dictionary of arcs in the dual graph
#and graph, dictionary of constraints over the variables
def cryptarToCSP(first, second, third):
    a = "".join(reversed(first))
    b = "".join(reversed(second))
    c = "".join(reversed(third))
    letter_vars = list(set(a + b + c))

    if len(b) < len(a):
        tmp = a
        a = b
        b = tmp

    doms = {}
    arcs = {}
    contr = {}
    vars = set(letter_vars) #add different letters as vars

    #for each letter in the third argument we add a carry variable
    #notice that even though the first letters dont have a carry added
    #to them, we add C0 to them any and set C0 later to have the domain [0]
    #this is done for consistency across vairbles and makes for simpler program
    for i in range(len(c)):
        vars.add("C" + str(i))
    #for each letter in first a, second b, third c arguments we have the equation
    #C(i) + a + b = c + C(i+1) where i is the index of the letters in their respective words
    #so for the left side we have an auxiliary variable named aux(i*2) which is a tuple of 3
    #integers where the first is carry integer, and for the right side we have an auxiliary
    #variable similar to the left just with two integers with one of them begin the carry integer
    #since b is the longer of the two arguments, we need 2 aux variables for each letter of b
    #where if b is longer than a we simply constrain it with C0 (which has domain [0]) to get
    # C(i) + C0 + b = c + 10*C(i+1) => C(i) + 0 + b = c + 10*C(i+1) => C(i) + b = c + 10*C(i+1)
    for i in range(len(b)*2):
        vars.add("aux" + str(i))

    #initialize a set of arcs for each variable
    for var in vars:
        arcs[var] = set()

    #create the domains for the different variables
    cary_dom = list(range(2))
    var_dom = list(range(10))
    aux1_dom = []
    for i in range(10):
        for j in range(10):
            #the first element of the left size aux variable is carry so can only be 1 or 0
            aux1_dom.append((0, i, j))
            aux1_dom.append((1, i, j))
    aux2_dom = []
    for i in range(10):
        #same with the right aux variable
        aux2_dom.append((i, 0))
        aux2_dom.append((i, 1))

    for i in range(len(b)):
        #name our variables at this index for convinieces
        varA = a[i] if i < len(a) else "C0" #if we run out of digits in a contrain with 0
        varB = b[i]
        varC = c[i]
        varCarry0 = "C"+str(i)
        varCarry1 = "C"+str(i+1)
        varAux0 = "aux"+str(i*2)
        varAux1 = "aux"+str(i*2+1)
        #set the domains for out variables
        doms[varA] = var_dom.copy()
        doms[varB] = var_dom.copy()
        doms[varC] = var_dom.copy()
        doms[varCarry0] = cary_dom.copy()
        doms[varAux0] = aux1_dom.copy()
        doms[varAux1] = aux2_dom.copy()

        #build the aux(i*2) = (C(i), a, b) constraints
        addConstraint(contr, varCarry0, varAux0, "Elem0")
        addConstraint(contr, varA, varAux0, "Elem1")
        addConstraint(contr, varB, varAux0, "Elem2")
        #build the aux(i*2+1) = (c, 10*C(i+1)) constraints
        addConstraint(contr, varC, varAux1, "Elem0")
        addConstraint(contr, varCarry1, varAux1, "Elem1")
        #build the sum(aux(i*2)) = sum(aux(i*2+1)) => C(i) + a + b = c + 10*C(i+1) constraint
        addConstraint(contr, varAux0, varAux1, "Sum")

    #for each two varaibles, a,b add constraint a != b
    for i in range(len(letter_vars)):
        for j in range(i+1,len(letter_vars)):
            addConstraint(contr, letter_vars[i], letter_vars[j], "Neq")

    #set domain of C0 to [0] since it's the carry to the first
    #digits and the first digits dont have a carry
    doms["C0"] = [0]

    #if the result c is longer than the arguments a and b it
    #can only be longer by 1 digit at most and in that case the
    #digit must be 1 and therefore the last carry variable must also
    #be 1, in other words in SEND + MORE = MONEY we get that Y = C3 = 1
    if len(c) > len(b):
        doms[c[len(c) - 1]] = [1]
        doms["C"+str(len(c) - 1)] = [1]
    #construct tha arcs dictionary such that each variable x
    #will have an arc to a variable y iff (x,y) is constrained of (y,x) is
    #constraint that way we get an undirected graph
    for con in contr:
        if not con[0] in arcs:
            arcs[con[0]] = set()
            doms[con[0]] = cary_dom.copy()
            vars.add(con[0])
        arcs[con[0]].add(con[1])
        arcs[con[1]].add(con[0])

    #return the parameters of the CSP problem
    return vars, doms, arcs, contr

#the implementations of revise (page 171) Artificial Intelligence: A Modern Approach 4th edition, Pearson;
#adjusted for our CSP format
def revise(csp, xi, xj):
    vars, doms, arcs, contr = csp
    revised = False

    #iterate over all values in xi's domain
    for x in doms[xi]:
        satisfied = False
        for y in doms[xj]: #iterate over all values in xj's domain
            if (xi,xj) in contr:  #check if the values satisfy all constraints between xi and xj
                satisfied = resolveConstraints(contr[(xi,xj)], x, y)
            else:
                satisfied = resolveConstraints(contr[(xj, xi)], y, x)
            #if satisfied, stop searching
            if satisfied: break
        #if there is no y in xj's domain that satisfies the constraints with value x
        #remove x from xi's domain and mark the function as having done a revision
        if not satisfied:
            if ENABLE_PRINT: print("no value of",xj,"satifies it's constraints with ",xi,'=',x,"removing it domain")
            doms[xi].remove(x)
            revised = True

    #return if revised or not
    return revised

#the implementations of AC3 (page 171) Artificial Intelligence: A Modern Approach 4th edition, Pearson;
#adjusted for our CSP format
def AC3(csp):
    vars, doms, arcs, contr = csp

    #create queue of all the arcs in the graph
    q = []
    for var in arcs:
        for nvar in arcs[var]:
            q.insert(0, (var, nvar))

    #while there are arcs in the queue, revise each arc and remove it from the queue
    #if a variables domain is left empty after revision then the CSP is inconsistent
    #meanig no solution can be found and so we return False
    while len(q) > 0:
        xi,xj = q.pop()
        if revise(csp,xi,xj):
            if len(doms[xi]) == 0:
                if ENABLE_PRINT: print("CSP inconsistent; domain of",xi," reduced to []")
                return False
            for xk in arcs[xi]:
                if xk != xj:
                    q.insert(0,(xk,xi))
    return True

#input: a CSP problem
#output: returns True if for each constraint there exists values in it's
#variables' domain that satisfy it, otherwise returns False
def isConsistent(csp):
    vars, doms, arcs, contr = csp

    for con in contr:
        a = con[0] #first var of the constraint
        b = con[1] #second var of the constraint

        #search all value pair in the domains of a and b for one that satisfies the constraints between them
        satisfied = False
        for x in doms[a]:
            for y in doms[b]:
                satisfied = resolveConstraints(contr[con], x, y)
                if satisfied:
                    break
            if satisfied: break
        #if no values found return False
        if not satisfied:
            if ENABLE_PRINT: print("CSP is inconsistent; no value in", b,"satisfies",contr[con],"with",a)
            return False
    #if we didn't return false then all are satisfiable
    return True

#input: a CSP problem, a variable var
#output: a list of values in the variables domain sorted by thier level of constraintedness
#meaning order values using LCV heuristic
def order_domain_values(csp,var):
    vars, doms, arcs, contr = csp
    vals = doms[var]
    vals_rank = {}

    for val in vals:
        rank = 1
        for neigh in arcs[var]:
            for y in doms[neigh]:
                if (var, neigh) in contr:  # check if the values satisfy all constraints between xi and xj
                    satisfied = resolveConstraints(contr[(var, neigh)], val, y)
                else:
                    satisfied = resolveConstraints(contr[(neigh, var)], y, val)
                if satisfied:
                    rank += 1
        vals_rank[val] = rank

    res =  sorted(vals, key=lambda var: 1/vals_rank[var]) #sort the values by their constraintedness
    return res

#input: a CSP problem, an assignment dictionary
#output: the variable in the CSP with the minimal options left for assignment
#to it meaning select unassigned variable using MRV heuristic and use the degree
#heuristic when two values have same number of options left
def select_unassigned_var(csp, assignment):
    vars, doms, arcs, contr = csp
    min = None

    for var in vars:
        if not(var in assignment): #iterate only unassigned variables
            if min == None:
                min = var
            else:
                n_arcs = len(arcs[var])
                n_vals = len(doms[var])
                min_arcs = len(arcs[min])
                min_vals = len(doms[min])
                if n_vals < min_vals:
                    min = var
                elif n_vals == min_vals:
                    if min_arcs < n_arcs:
                        min = var
    #return the minimal variable
    return min

def cloneDomain(doms):
    doms_copy = {}
    for var in doms:
        doms_copy[var] = doms[var].copy()
    return doms_copy

#the implementations of backtrack (page 176) Artificial Intelligence: A Modern Approach 4th edition, Pearson;
#adjusted for our CSP format
def backtrack(csp, assignment):
    vars, doms, arcs, contr = csp

    if ENABLE_PRINT: print("begin backtrack")
    if len(assignment) == len(vars): #if all variables are assigned then return the solution
        if ENABLE_PRINT: print("found solution!")
        return assignment

    var = select_unassigned_var(csp, assignment) #select unassigned variable
    if ENABLE_PRINT: print("searching for a value for",var)
    ordered_vals = order_domain_values(csp, var)
    for val in ordered_vals: #iterate over its domain by constraintedness of values
        if ENABLE_PRINT: print("trying the value",var,'=',val)
        assignment[var] = val #assign the value to the variable
        old_dom = doms[var] #save the domain
        doms[var] = [val] #apply assigment of variable to the CSP

        #if isConsistent(csp): #check if CSP is still consistent after assignment
        if ENABLE_PRINT: print("value assignment is consistent, inferring..")
        tmp_csp = (vars, cloneDomain(doms), arcs, contr)
        # apply inference using AC3, if failed then
        # no solution exists in the current execution path
        if AC3(tmp_csp):
            if ENABLE_PRINT: print("applied inference")
            result = backtrack(tmp_csp, assignment) #backtrack
            if result != None: #if solution found, propagate it to the top
                if ENABLE_PRINT: print("end backtrack")
                return result
            if ENABLE_PRINT: print("removed inference")

        if ENABLE_PRINT: print("failed value",var,'=',val)
        del assignment[var] #undo assignment of variable
        doms[var] = old_dom

    if ENABLE_PRINT: print("end backtrack")
    return None

#the implementations of backtracking-search (page 176) Artificial Intelligence: A Modern Approach 4th edition, Pearson;
#adjusted for our CSP format
def backtracking_search(csp):
    vars, doms, arcs, contr = csp

    if ENABLE_PRINT:
        print("created CSP\n---variable domains---")
        for var in doms:
            print(var,":",doms[var])
        print("---constraints---")
        for con in contr:
            print(con,":",contr[con])

    #do the backtracking algorithm
    return backtrack(csp, {})

def main():
    global ENABLE_PRINT

    if len(sys.argv) < 4:
        print("please supply 3 arguments")
        return

    #argument from command line
    a = sys.argv[1]
    b = sys.argv[2]
    c = sys.argv[3]

    if len(sys.argv) > 4:
        ENABLE_PRINT = (sys.argv[4] != "no_print") #to disable printing from command line
    #Step 1 reduce to CSP
    csp = cryptarToCSP(a, b, c)
    #Step 2 backtracking-search
    result = backtracking_search(csp)
    #print results as requested
    if result:
        for var in result:
            if len(var) == 1:
                print(var+"="+str(result[var]),end=' ')
        print()
    else:
        print("No solution found")

if __name__ == "__main__":
    main()