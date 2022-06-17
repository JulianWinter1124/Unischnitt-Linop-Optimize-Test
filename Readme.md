```python
import gurobipy as gp
from gurobipy import GRB
```


```python
import cvxpy as cp
import cvxpy
import numpy as np
```


```python
print(cvxpy.installed_solvers())
```

    ['CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'GUROBI', 'OSQP', 'SCIPY', 'SCS']



```python
w = [3.3, 2.3, 2.3, 3.3, 3.3, 1.7, 1., 1., 1.3, 1.7, 2.3, 2.0, 3.3, 2.3] #grades
cps = [ 5,   5,  5,    5,  10,   5,  5,  5,   5,   5,   5,   5,  10,   5] #CPs
theo = [3,4,5,12,13] #theoretical CS, course idx 
prakt = [0, 1, 2, 6, 7, 8, 9, 10, 11, 13] #thechnical CS, course idx 
print(len(w), len(cps))
print(theo, prakt)
```

    14 14
    [3, 4, 5, 12, 13] [0, 1, 2, 6, 7, 8, 9, 10, 11, 13]



```python
#Grade average calculation
#Correct formula but not linear...
print('(2*(', end='')
for i, note in enumerate(w):
    print(f'xs{i}*{note}*{cps[i]} + ', end='')#Main emphasis weight formula
print('0) /(', end='')
for i, note in enumerate(w):
    print(f'xs{i}*{cps[i]} + ', end='')
print('0) + ', end='')

print('(', end='')
for i, note in enumerate(w):
    if not i in theo:
        continue
    print(f'xt{i}*{note}*{cps[i]} + ', end='')#Theo weight formula
print('0) /(', end='')
for i, note in enumerate(w):
    if not i in theo:
        continue
    print(f'xt{i}*{cps[i]} + ', end='')
print('0) + ', end='')
    
print('(', end='')
for i, note in enumerate(w):
    if not i in prakt:
        continue
    print(f'xp{i}*{note}*{cps[i]} + ', end='')#Tech weight formula
print('0) / (', end='')
for i, note in enumerate(w):
    if not i in prakt:
        continue
    print(f'xp{i}*{cps[i]} + ', end='')
print('0))/4')   
```

    (2*(xs0*3.3*5 + xs1*2.3*5 + xs2*2.3*5 + xs3*3.3*5 + xs4*3.3*10 + xs5*1.7*5 + xs6*1.0*5 + xs7*1.0*5 + xs8*1.3*5 + xs9*1.7*5 + xs10*2.3*5 + xs11*2.0*5 + xs12*3.3*10 + xs13*2.3*5 + 0) /(xs0*5 + xs1*5 + xs2*5 + xs3*5 + xs4*10 + xs5*5 + xs6*5 + xs7*5 + xs8*5 + xs9*5 + xs10*5 + xs11*5 + xs12*10 + xs13*5 + 0) + (xt3*3.3*5 + xt4*3.3*10 + xt5*1.7*5 + xt12*3.3*10 + xt13*2.3*5 + 0) /(xt3*5 + xt4*10 + xt5*5 + xt12*10 + xt13*5 + 0) + (xp0*3.3*5 + xp1*2.3*5 + xp2*2.3*5 + xp6*1.0*5 + xp7*1.0*5 + xp8*1.3*5 + xp9*1.7*5 + xp10*2.3*5 + xp11*2.0*5 + xp13*2.3*5 + 0) / (xp0*5 + xp1*5 + xp2*5 + xp6*5 + xp7*5 + xp8*5 + xp9*5 + xp10*5 + xp11*5 + xp13*5 + 0))/4



```python
#Grade average calculation
#Linear but incorrect
print('2*(', end='')
for i, note in enumerate(w):
    print(f'xs{i}*{note}*{cps[i]} + ', end='')
print('0) +', end='')

print('(', end='')
for i, note in enumerate(w):
    if not i in theo:
        continue
    print(f'xt{i}*{note}*{cps[i]} + ', end='')
print('0) +', end='')
    
print('(', end='')
for i, note in enumerate(w):
    if not i in prakt:
        continue
    print(f'xp{i}*{note}*{cps[i]} + ', end='')
print('0)', end='')
```

    2*(xs0*3.3*5 + xs1*2.3*5 + xs2*2.3*5 + xs3*3.3*5 + xs4*3.3*10 + xs5*1.7*5 + xs6*1.0*5 + xs7*1.0*5 + xs8*1.3*5 + xs9*1.7*5 + xs10*2.3*5 + xs11*2.0*5 + xs12*3.3*10 + xs13*2.3*5 + 0) +(xt3*3.3*5 + xt4*3.3*10 + xt5*1.7*5 + xt12*3.3*10 + xt13*2.3*5 + 0) +(xp0*3.3*5 + xp1*2.3*5 + xp2*2.3*5 + xp6*1.0*5 + xp7*1.0*5 + xp8*1.3*5 + xp9*1.7*5 + xp10*2.3*5 + xp11*2.0*5 + xp13*2.3*5 + 0)


```python
#constraints
#only one variable is 1
for i in range(len(w)):
    print(f'xs{i} + ', end='')
    print(f'xi{i} + ', end='')
    if i in theo:
        print(f'xt{i} + ', end='')
    if i in prakt:
        print(f'xp{i} + ', end='')
    print(f'0 <= 1')
```

    xs0 + xi0 + xp0 + 0 <= 1
    xs1 + xi1 + xp1 + 0 <= 1
    xs2 + xi2 + xp2 + 0 <= 1
    xs3 + xi3 + xt3 + 0 <= 1
    xs4 + xi4 + xt4 + 0 <= 1
    xs5 + xi5 + xt5 + 0 <= 1
    xs6 + xi6 + xp6 + 0 <= 1
    xs7 + xi7 + xp7 + 0 <= 1
    xs8 + xi8 + xp8 + 0 <= 1
    xs9 + xi9 + xp9 + 0 <= 1
    xs10 + xi10 + xp10 + 0 <= 1
    xs11 + xi11 + xp11 + 0 <= 1
    xs12 + xi12 + xt12 + 0 <= 1
    xs13 + xi13 + xt13 + xp13 + 0 <= 1



```python
#Select enough cps
for i, cpi in enumerate(cps):
    print(f'xs{i}*{cpi} + ', end='')
    print(f'xi{i}*{cpi} + ', end='')
    if i in theo:
        print(f'xt{i}*{cpi} + ', end='')
    if i in prakt:
        print(f'xp{i}*{cpi} + ', end='')
print(f'0 >= 80')
```

    xs0*5 + xi0*5 + xp0*5 + xs1*5 + xi1*5 + xp1*5 + xs2*5 + xi2*5 + xp2*5 + xs3*5 + xi3*5 + xt3*5 + xs4*10 + xi4*10 + xt4*10 + xs5*5 + xi5*5 + xt5*5 + xs6*5 + xi6*5 + xp6*5 + xs7*5 + xi7*5 + xp7*5 + xs8*5 + xi8*5 + xp8*5 + xs9*5 + xi9*5 + xp9*5 + xs10*5 + xi10*5 + xp10*5 + xs11*5 + xi11*5 + xp11*5 + xs12*10 + xi12*10 + xt12*10 + xs13*5 + xi13*5 + xt13*5 + xp13*5 + 0 >= 80



```python
for i, cpi in enumerate(cps):
    print(f'xs{i}*{cpi} + ', end='')
print(f'0 >= 30')
for i, cpi in enumerate(cps):
    if i in theo:
        print(f'xt{i}*{cpi} + ', end='')
print(f'0 >= 15')
for i, cpi in enumerate(cps):
    if i in prakt:
        print(f'xp{i}*{cpi} + ', end='')
print(f'0 >= 15')
for i, cpi in enumerate(cps):
    print(f'xi{i}*{cpi} + ', end='')
print(f'0 >= 10')
```

    xs0*5 + xs1*5 + xs2*5 + xs3*5 + xs4*10 + xs5*5 + xs6*5 + xs7*5 + xs8*5 + xs9*5 + xs10*5 + xs11*5 + xs12*10 + xs13*5 + 0 >= 30
    xt3*5 + xt4*10 + xt5*5 + xt12*10 + xt13*5 + 0 >= 15
    xp0*5 + xp1*5 + xp2*5 + xp6*5 + xp7*5 + xp8*5 + xp9*5 + xp10*5 + xp11*5 + xp13*5 + 0 >= 15
    xi0*5 + xi1*5 + xi2*5 + xi3*5 + xi4*10 + xi5*5 + xi6*5 + xi7*5 + xi8*5 + xi9*5 + xi10*5 + xi11*5 + xi12*10 + xi13*5 + 0 >= 10



```python
for i in range(len(cps)):
    print(f'xs{i} >= 0', end=',\n')    
    print(f'xi{i} >= 0', end=',\n')    
    if i in theo:
        print(f'xt{i} >= 0', end=',\n')    
    if i in prakt:
        print(f'xp{i} >= 0', end=',\n')
```

    xs0 >= 0,
    xi0 >= 0,
    xp0 >= 0,
    xs1 >= 0,
    xi1 >= 0,
    xp1 >= 0,
    xs2 >= 0,
    xi2 >= 0,
    xp2 >= 0,
    xs3 >= 0,
    xi3 >= 0,
    xt3 >= 0,
    xs4 >= 0,
    xi4 >= 0,
    xt4 >= 0,
    xs5 >= 0,
    xi5 >= 0,
    xt5 >= 0,
    xs6 >= 0,
    xi6 >= 0,
    xp6 >= 0,
    xs7 >= 0,
    xi7 >= 0,
    xp7 >= 0,
    xs8 >= 0,
    xi8 >= 0,
    xp8 >= 0,
    xs9 >= 0,
    xi9 >= 0,
    xp9 >= 0,
    xs10 >= 0,
    xi10 >= 0,
    xp10 >= 0,
    xs11 >= 0,
    xi11 >= 0,
    xp11 >= 0,
    xs12 >= 0,
    xi12 >= 0,
    xt12 >= 0,
    xs13 >= 0,
    xi13 >= 0,
    xt13 >= 0,
    xp13 >= 0,



```python
for i, note in enumerate(w):
    exec(f'xs{i} = cp.Variable(1, boolean=True)')
    exec(f'xi{i} = cp.Variable(1, boolean=True)')
for i, note in enumerate(w):
    if i in theo:
        exec(f'xt{i} = cp.Variable(1, boolean=True)')
for i, note in enumerate(w):
    if i in prakt:
        exec(f'xp{i} = cp.Variable(1, boolean=True)')
```


```python
constraints = [
xs0 + xi0 + xp0 + 0 == 1,
xs1 + xi1 + xp1 + 0 == 1,
xs2 + xi2 + xp2 + 0 == 1,
xs3 + xi3 + xt3 + 0 == 1,
xs4 + xi4 + xt4 + 0 == 1,
xs5 + xi5 + xt5 + 0 == 1,
xs6 + xi6 + xp6 + 0 == 1,
xs7 + xi7 + xp7 + 0 == 1,
xs8 + xi8 + xp8 + 0 == 1,
xs9 + xi9 + xp9 + 0 == 1,
xs10 + xi10 + xp10 + 0 == 1,
xs11 + xi11 + xp11 + 0 == 1,
xs12 + xi12 + xt12 + 0 == 1,
xs13 + xi13 + xt13 + xp13 + 0 == 1,
xs0 >= 0,
xi0 >= 0,
xp0 >= 0,
xs1 >= 0,
xi1 >= 0,
xp1 >= 0,
xs2 >= 0,
xi2 >= 0,
xp2 >= 0,
xs3 >= 0,
xi3 >= 0,
xt3 >= 0,
xs4 >= 0,
xi4 >= 0,
xt4 >= 0,
xs5 >= 0,
xi5 >= 0,
xt5 >= 0,
xs6 >= 0,
xi6 >= 0,
xp6 >= 0,
xs7 >= 0,
xi7 >= 0,
xp7 >= 0,
xs8 >= 0,
xi8 >= 0,
xp8 >= 0,
xs9 >= 0,
xi9 >= 0,
xp9 >= 0,
xs10 >= 0,
xi10 >= 0,
xp10 >= 0,
xs11 >= 0,
xi11 >= 0,
xp11 >= 0,
xs12 >= 0,
xi12 >= 0,
xt12 >= 0,
xs13 >= 0,
xi13 >= 0,
xt13 >= 0,
xp13 >= 0,
xs0*5 + xi0*5 + xp0*5 + xs1*5 + xi1*5 + xp1*5 + xs2*5 + xi2*5 + xp2*5 + xs3*5 + xi3*5 + xt3*5 + xs4*10 + xi4*10 + xt4*10 + xs5*5 + xi5*5 + xt5*5 + xs6*5 + xi6*5 + xp6*5 + xs7*5 + xi7*5 + xp7*5 + xs8*5 + xi8*5 + xp8*5 + xs9*5 + xi9*5 + xp9*5 + xs10*5 + xi10*5 + xp10*5 + xs11*5 + xi11*5 + xp11*5 + xs12*10 + xi12*10 + xt12*10 + xs13*5 + xi13*5 + xt13*5 + xp13*5 + 0 >= 80,
xs0*5 + xs1*5 + xs2*5 + xs3*5 + xs4*10 + xs5*5 + xs6*5 + xs7*5 + xs8*5 + xs9*5 + xs10*5 + xs11*5 + xs12*10 + xs13*5 + 0 >= 30,
xt3*5 + xt4*10 + xt5*5 + xt12*10 + xt13*5 + 0 >= 15,
xp0*5 + xp1*5 + xp2*5 + xp6*5 + xp7*5 + xp8*5 + xp9*5 + xp10*5 + xp11*5 + xp13*5 + 0 >= 15,
xi0*5 + xi1*5 + xi2*5 + xi3*5 + xi4*10 + xi5*5 + xi6*5 + xi7*5 + xi8*5 + xi9*5 + xi10*5 + xi11*5 + xi12*10 + xi13*5 + 0 >= 10
]
```


```python
# constraints.append(xi0 + xi1 + xi2 + xi3 + xi4 + xi5 + xi6 + xi7 + xi8 + xi9 + xi10 + xi11 + xi12 + xi13 <= 3)
```


```python
objective = cp.Minimize((2*(xs0*3.3*5 + xs1*2.3*5 + xs2*2.3*5 + xs3*3.3*5 + xs4*3.3*10 + xs5*1.7*5 + xs6*1.0*5 + xs7*1.0*5 + xs8*1.3*5 + xs9*1.7*5 + xs10*2.3*5 + xs11*2.0*5 + xs12*3.3*10 + xs13*2.3*5 + 0) /(xs0*5 + xs1*5 + xs2*5 + xs3*5 + xs4*10 + xs5*5 + xs6*5 + xs7*5 + xs8*5 + xs9*5 + xs10*5 + xs11*5 + xs12*10 + xs13*5 + 0) + (xt3*3.3*5 + xt4*3.3*10 + xt5*1.7*5 + xt12*3.3*10 + xt13*2.3*5 + 0) /(xt3*5 + xt4*10 + xt5*5 + xt12*10 + xt13*5 + 0) + (xp0*3.3*5 + xp1*2.3*5 + xp2*2.3*5 + xp6*1.0*5 + xp7*1.0*5 + xp8*1.3*5 + xp9*1.7*5 + xp10*2.3*5 + xp11*2.0*5 + xp13*2.3*5 + 0) / (xp0*5 + xp1*5 + xp2*5 + xp6*5 + xp7*5 + xp8*5 + xp9*5 + xp10*5 + xp11*5 + xp13*5 + 0))/4)
```


```python
prob = cp.Problem(objective, constraints)
```


```python
result = prob.solve(solver='GUROBI', qcp=True)
```


    ---------------------------------------------------------------------------

    DQCPError                                 Traceback (most recent call last)

    /tmp/ipykernel_13859/4235230865.py in <module>
    ----> 1 result = prob.solve(solver='GUROBI', qcp=True)
    

    ~/anaconda3/envs/web/lib/python3.9/site-packages/cvxpy/problems/problem.py in solve(self, *args, **kwargs)
        479         else:
        480             solve_func = Problem._solve
    --> 481         return solve_func(self, *args, **kwargs)
        482 
        483     @classmethod


    ~/anaconda3/envs/web/lib/python3.9/site-packages/cvxpy/problems/problem.py in _solve(self, solver, warm_start, verbose, gp, qcp, requires_grad, enforce_dpp, ignore_dpp, **kwargs)
        984             if qcp and not self.is_dcp():
        985                 if not self.is_dqcp():
    --> 986                     raise error.DQCPError("The problem is not DQCP.")
        987                 if verbose:
        988                     s.LOGGER.info(


    DQCPError: The problem is not DQCP.



```python
print('\t sch\tind\tthe\tprak')
for i in range(len(cps)):
    print('Fach:', i+1, end=': ')
    eval(f'print(xs{i}.value[0], end="\t")')    
    eval(f'print(xi{i}.value[0], end="\t")')  
    if i in theo:
        eval(f'print(xt{i}.value[0], end="\t")')
    else:
        print(0, end='\t')
    if i in prakt:
        eval(f'print(xp{i}.value[0], end="")')    
    else:
        print(0, end='')
    print()
```

    	 sch	ind	the	prak
    Fach: 1: 0.0	1.0	0	0.0
    Fach: 2: 0.0	0.0	0	1.0
    Fach: 3: 0.0	0.0	0	1.0
    Fach: 4: 0.0	1.0	0.0	0
    Fach: 5: 0.0	1.0	0.0	0
    Fach: 6: 1.0	0.0	0.0	0
    Fach: 7: 1.0	0.0	0	0.0
    Fach: 8: 1.0	0.0	0	0.0
    Fach: 9: 1.0	0.0	0	0.0
    Fach: 10: 1.0	0.0	0	0.0
    Fach: 11: 0.0	0.0	0	1.0
    Fach: 12: 1.0	0.0	0	0.0
    Fach: 13: 0.0	0.0	1.0	0
    Fach: 14: 0.0	0.0	1.0	0.0



```python
print(prob.value)
```

    166.0



```python
ind_sum = sum([eval(f'xi{i}.value[0]') for i in range(len(cps))])
schwer_sum = sum([eval(f'xs{i}.value[0]') for i in range(len(cps))])
prakt_sum = sum([eval(f'xp{i}.value[0]') for i in range(len(cps)) if i in prakt])
theo_sum = sum([eval(f'xt{i}.value[0]') for i in range(len(cps)) if i in theo])
print('Individuell: ', ind_sum)
print('Schwerpunkt: ', schwer_sum)
print('Theo: ', theo_sum)
print('Prakt: ', prakt_sum)
# eval(f'print(xi{i}.value[0], end="\t")')  
# if i in theo:
#     eval(f'print(xt{i}.value[0], end="\t")')
# else:
#     print(0, end='\t')
# if i in prakt:
#     eval(f'print(xp{i}.value[0], end="")')    
```

    Individuell:  3.0
    Schwerpunkt:  6.0
    Theo:  2.0
    Prakt:  3.0



```python
schwer_mean = sum([eval(f'xs{i}.value[0]')*cps[i]*w[i] for i in range(len(cps))])/sum([eval(f'xs{i}.value[0]')*cps[i] for i in range(len(cps))])
prakt_mean = sum([eval(f'xp{i}.value[0]')*cps[i]*w[i] for i in range(len(cps)) if i in prakt]) / sum([eval(f'xp{i}.value[0]')*cps[i] for i in range(len(cps)) if i in prakt])
theo_mean = sum([eval(f'xt{i}.value[0]')*cps[i]*w[i] for i in range(len(cps)) if i in theo]) / sum([eval(f'xt{i}.value[0]')*cps[i] for i in range(len(cps)) if i in theo])
print(schwer_mean)
print(prakt_mean)
print(theo_mean)
```

    1.45
    2.3
    2.966666666666667



```python
# objective.value/(2*schwer_sum+prakt_sum+theo_sum)
(schwer_mean*2+prakt_mean+theo_mean)/4
```




    2.0416666666666665




```python
from numba import jit
@jit
def uni_schnitt(x, *params):
    xi0, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9, xi10, xi11, xi12, xi13, \
    xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, \
    xp0, xp1, xp2, xp6, xp7, xp8, xp9, xp10, xp11, xp13, \
    xt3, xt4, xt5, xt12, xt13 = x
    c, n = params
    return (2*(sum([ xs0*c[0]*n[0], xs1*c[1]*n[1], xs2*c[2]*n[2], xs3*c[3]*n[3], xs4*c[4]*n[4], xs5*c[5]*n[5], xs6*c[6]*n[6], xs7*c[7]*n[7], xs8*c[8]*n[8], xs9*c[9]*n[9], xs10*c[10]*n[10], xs11*c[11]*n[11], xs12*c[12]*n[12], xs13*c[13]*n[13] ]) /sum([ xs0*c[0], xs1*c[1], xs2*c[2], xs3*c[3], xs4*c[4], xs5*c[5], xs6*c[6], xs7*c[7], xs8*c[8], xs9*c[9], xs10*c[10], xs11*c[11], xs12*c[12], xs13*c[13] ])) +\
(sum([ xp0*c[0]*n[0], xp1*c[1]*n[1], xp2*c[2]*n[2], xp6*c[6]*n[6], xp7*c[7]*n[7], xp8*c[8]*n[8], xp9*c[9]*n[9], xp10*c[10]*n[10], xp11*c[11]*n[11], xp13*c[13]*n[13] ]) /sum([ xp0*c[0], xp1*c[1], xp2*c[2], xp6*c[6], xp7*c[7], xp8*c[8], xp9*c[9], xp10*c[10], xp11*c[11], xp13*c[13] ])) +\
(sum([ xt3*c[3]*n[3], xt4*c[4]*n[4], xt5*c[5]*n[5], xt12*c[12]*n[12], xt13*c[13]*n[13] ]) /sum([ xt3*c[3], xt4*c[4], xt5*c[5], xt12*c[12], xt13*c[13] ])))/4
```


```python
print(', '.join([f'xi{i}' for i in range(len(cps))]))
print(', '.join([f'xs{i}' for i in range(len(cps))]))
print(', '.join([f'xp{i}' for i in range(len(cps)) if i in prakt]))
print(', '.join([f'xt{i}' for i in range(len(cps)) if i in theo]))
```

    xi0, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9, xi10, xi11, xi12, xi13
    xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13
    xp0, xp1, xp2, xp6, xp7, xp8, xp9, xp10, xp11, xp13
    xt3, xt4, xt5, xt12, xt13



```python
uni_schnitt(np.zeros(43), None)
```




    0.0




```python
print('(sum([', ', '.join([f'xs{i}*c[{i}]*n[{i}]' for i in range(len(cps))]), '])', '/sum([', ', '.join([f'xs{i}*c[{i}]' for i in range(len(cps))]), ']))')
print('(sum([', ', '.join([f'xp{i}*c[{i}]*n[{i}]' for i in range(len(cps)) if i in prakt]), '])', '/sum([', ', '.join([f'xp{i}*c[{i}]' for i in range(len(cps)) if i in prakt]), ']))')
print('(sum([', ', '.join([f'xt{i}*c[{i}]*n[{i}]' for i in range(len(cps)) if i in theo]), '])', '/sum([', ', '.join([f'xt{i}*c[{i}]' for i in range(len(cps)) if i in theo]), ']))')
```

    (sum([ xs0*c[0]*n[0], xs1*c[1]*n[1], xs2*c[2]*n[2], xs3*c[3]*n[3], xs4*c[4]*n[4], xs5*c[5]*n[5], xs6*c[6]*n[6], xs7*c[7]*n[7], xs8*c[8]*n[8], xs9*c[9]*n[9], xs10*c[10]*n[10], xs11*c[11]*n[11], xs12*c[12]*n[12], xs13*c[13]*n[13] ]) /sum([ xs0*c[0], xs1*c[1], xs2*c[2], xs3*c[3], xs4*c[4], xs5*c[5], xs6*c[6], xs7*c[7], xs8*c[8], xs9*c[9], xs10*c[10], xs11*c[11], xs12*c[12], xs13*c[13] ]))
    (sum([ xp0*c[0]*n[0], xp1*c[1]*n[1], xp2*c[2]*n[2], xp6*c[6]*n[6], xp7*c[7]*n[7], xp8*c[8]*n[8], xp9*c[9]*n[9], xp10*c[10]*n[10], xp11*c[11]*n[11], xp13*c[13]*n[13] ]) /sum([ xp0*c[0], xp1*c[1], xp2*c[2], xp6*c[6], xp7*c[7], xp8*c[8], xp9*c[9], xp10*c[10], xp11*c[11], xp13*c[13] ]))
    (sum([ xt3*c[3]*n[3], xt4*c[4]*n[4], xt5*c[5]*n[5], xt12*c[12]*n[12], xt13*c[13]*n[13] ]) /sum([ xt3*c[3], xt4*c[4], xt5*c[5], xt12*c[12], xt13*c[13] ]))



```python
uni_schnitt(np.ones(43), cps, w)
```




    2.397767857142857




```python
import scipy
```


```python
def rec_print(i):
    if i in theo and i in prakt:
        print("".join(["  "]*i)+f"for i{i} in range(4): #{faecher[i]}")
    elif i in theo:
        print("".join(["  "]*i)+f"for i{i} in [0,1,3]: #{faecher[i]}")
    elif i in prakt:
        print("".join(["  "]*i)+f"for i{i} in [0,2,3]: #{faecher[i]}")
    else:
        print("".join(["  "]*i)+f"for i{i} in [0,3]: #{faecher[i]}")
    print("".join(["  "]*(i+1))+f"x[{i}]=i{i}")
    print("".join(["  "]*(i+1))+f"xbool[i{i},{i}]=True")
    if i<13:
        rec_print(i+1)
    print("".join(["  "]*(i+1))+f"xbool[i{i},{i}]=False")
rec_print(0)
```

    for i0 in [0,2,3]: #Bio
      x[0]=i0
      xbool[i0,0]=True
      for i1 in [0,2,3]: #Fuzzy
        x[1]=i1
        xbool[i1,1]=True
        for i2 in [0,2,3]: #Geoinfo
          x[2]=i2
          xbool[i2,2]=True
          for i3 in [0,1,3]: #Causality
            x[3]=i3
            xbool[i3,3]=True
            for i4 in [0,1,3]: #GraphenII
              x[4]=i4
              xbool[i4,4]=True
              for i5 in [0,1,3]: #NP-Graphen
                x[5]=i5
                xbool[i5,5]=True
                for i6 in [0,2,3]: #InforTheo
                  x[6]=i6
                  xbool[i6,6]=True
                  for i7 in [0,2,3]: #Multimedia
                    x[7]=i7
                    xbool[i7,7]=True
                    for i8 in [0,2,3]: #MachineLearning
                      x[8]=i8
                      xbool[i8,8]=True
                      for i9 in [0,2,3]: # Linop
                        x[9]=i9
                        xbool[i9,9]=True
                        for i10 in [0,2,3]: #Deep Learning
                          x[10]=i10
                          xbool[i10,10]=True
                          for i11 in [0,2,3]: #Advances Data
                            x[11]=i11
                            xbool[i11,11]=True
                            for i12 in [0,1,3]: #GerechteAufteilung
                              x[12]=i12
                              xbool[i12,12]=True
                              for i13 in range(4): #ReinforcementLearning
                                x[13]=i13
                                xbool[i13,13]=True
                                xbool[i13,13]=False
                              xbool[i12,12]=False
                            xbool[i11,11]=False
                          xbool[i10,10]=False
                        xbool[i9,9]=False
                      xbool[i8,8]=False
                    xbool[i7,7]=False
                  xbool[i6,6]=False
                xbool[i5,5]=False
              xbool[i4,4]=False
            xbool[i3,3]=False
          xbool[i2,2]=False
        xbool[i1,1]=False
      xbool[i0,0]=False



```python
x = np.zeros((4, len(w)), dtype=bool)
#scipy.optimize.brute(unischnitt2, [slice(0, 4, 1)]*14, args=[x, cps, w])

from numba import njit
@njit
def unischnitt2(x, wt, cpt, theo, prakt):
#     for i in theo:
#         if xt[i]==1:
#             return np.inf
#     for i in prakt:
#         if xt[i]==2:
#             return np.inf
    if np.sum(x[0, :]*cpt) < 30: #sch
        return np.inf
    if np.sum(x[1, :]*cpt) < 15: #theo
        return np.inf
    if np.sum(x[2, :]*cpt) < 15: #prakt
        return np.inf
    if np.sum(x[3, :]*cpt) < 10: #inf
        return np.inf
    if np.sum(x*cpt) < 70: #gesamt
        return np.inf
    
    res = (2*np.sum(x[0]*wt*cpt)/np.sum(x[0]*cpt) + np.sum(x[1]*wt*cpt)/np.sum(x[1]*cpt) + np.sum(x[2]*wt*cpt)/np.sum(x[2]*cpt))/4

    return res
    

#@njit
def run(w, cps, theo, prakt, xbool, x):
    mini = np.inf
    for i0 in [0,2,3]: #Bio
      x[0]=i0
      xbool[i0,0]=True
      for i1 in [0,2,3]: #Fuzzy
        x[1]=i1
        xbool[i1,1]=True
        for i2 in [0,2,3]: #Geoinfo
          x[2]=i2
          xbool[i2,2]=True
          for i3 in [0,1,3]: #Causality
            x[3]=i3
            xbool[i3,3]=True
            for i4 in [0,1,3]: #GraphenII
              x[4]=i4
              xbool[i4,4]=True
              for i5 in [0,1,3]: #NP-Graphen
                x[5]=i5
                xbool[i5,5]=True
                for i6 in [0,2,3]: #InforTheo
                  x[6]=i6
                  xbool[i6,6]=True
                  for i7 in [0,2,3]: #Multimedia
                    x[7]=i7
                    xbool[i7,7]=True
                    for i8 in [0,2,3]: #MachineLearning
                      x[8]=i8
                      xbool[i8,8]=True
                      for i9 in [0,2,3]: # Linop
                        x[9]=i9
                        xbool[i9,9]=True
                        for i10 in [0,2,3]: #Deep Learning
                          x[10]=i10
                          xbool[i10,10]=True
                          for i11 in [0,2,3]: #Advances Data
                            x[11]=i11
                            xbool[i11,11]=True
                            for i12 in [0,1,3]: #GerechteAufteilung
                              x[12]=i12
                              xbool[i12,12]=True
                              for i13 in range(4): #ReinforcementLearning
                                x[13]=i13
                                xbool[i13,13]=True
                                res = unischnitt2(xbool, w, cps, theo, prakt)
                                if res<np.inf and res<=mini:
                                    print(res, x)
                                    mini=res
                                xbool[i13,13]=False
                              xbool[i12,12]=False
                            xbool[i11,11]=False
                          xbool[i10,10]=False
                        xbool[i9,9]=False
                      xbool[i8,8]=False
                    xbool[i7,7]=False
                  xbool[i6,6]=False
                xbool[i5,5]=False
              xbool[i4,4]=False
            xbool[i3,3]=False
          xbool[i2,2]=False
        xbool[i1,1]=False
      xbool[i0,0]=False
                                
                                


```


```python
w = np.array([3.3, 2.3, 2.3, 3.3, 3.3, 1.7, 1., 1., 1.3, 1.7, 2.3, 2.0, 3.3, 2.3])
cps = np.array([ 5,   5,  5,    5,  10,   5,  5,  5,   5,   5,   5,   5,  10,   5])
theo = np.array([3,4,5,12,13])
prakt = np.array([0, 1, 2, 6, 7, 8, 9, 10, 11, 13])
faecher = "Bio,Fuzzy,Geoinfo,Causality,GraphenII,NP-Graphen,InforTheo,Multimedia,MachineLearning, Linop,Deep Learning,Advances Data,GerechteAufteilung,ReinforcementLearning".split(',')
xbool = np.zeros((4, 14), dtype=bool)
x = np.zeros(14, dtype=int)
run(w, cps, theo, prakt, xbool, x)


```

    2.35625 [0 0 0 0 0 0 0 2 2 2 3 3 1 1]
    2.35625 [0 0 0 0 0 0 2 0 2 2 3 3 1 1]
    2.3500000000000005 [0 0 0 0 0 0 2 2 0 2 3 3 1 1]
    2.3416666666666663 [0 0 0 0 0 0 2 2 2 0 3 3 1 1]
    2.3395833333333336 [0 0 0 0 0 1 0 2 0 2 3 2 1 3]
    2.33125 [0 0 0 0 0 1 0 2 2 0 3 2 1 3]
    2.325 [0 0 0 0 0 1 0 2 2 2 3 0 1 3]
    2.325 [0 0 0 0 0 1 2 0 2 2 3 0 1 3]
    2.325 [0 0 0 0 0 1 2 2 0 0 3 2 1 3]
    2.3187500000000005 [0 0 0 0 0 1 2 2 0 2 3 0 1 3]
    2.3104166666666663 [0 0 0 0 0 1 2 2 2 0 3 0 1 3]
    2.2541666666666664 [0 0 0 0 1 0 0 0 0 2 2 2 3 1]
    2.2458333333333336 [0 0 0 0 1 0 0 0 2 0 2 2 3 1]
    2.2333333333333334 [0 0 0 0 1 0 0 0 2 2 0 2 3 1]
    2.2226190476190477 [0 0 0 0 1 0 0 0 2 2 3 2 3 1]
    2.219047619047619 [0 0 0 0 1 0 0 2 0 2 3 2 3 1]
    2.21875 [0 0 0 0 1 0 0 2 2 0 0 2 3 1]
    2.2142857142857144 [0 0 0 0 1 0 0 2 2 0 3 2 3 1]
    2.2125 [0 0 0 0 1 0 0 2 2 2 0 0 3 1]
    2.210714285714286 [0 0 0 0 1 0 0 2 2 2 3 0 3 1]
    2.210714285714286 [0 0 0 0 1 0 2 0 2 2 3 0 3 1]
    2.210714285714286 [0 0 0 0 1 0 2 2 0 0 3 2 3 1]
    2.20625 [0 0 0 0 1 0 2 2 0 2 0 0 3 1]
    2.1979166666666665 [0 0 0 0 1 0 2 2 2 0 0 0 3 1]
    2.193452380952381 [0 0 0 0 1 1 0 0 2 0 2 2 3 1]
    2.186309523809524 [0 0 0 0 1 1 0 0 2 2 0 2 3 1]
    2.1791666666666667 [0 0 0 0 1 1 0 0 2 2 3 2 3 1]
    2.1791666666666667 [0 0 0 0 1 1 0 2 0 2 3 2 3 1]
    2.1779761904761905 [0 0 0 0 1 1 0 2 2 0 0 2 3 1]
    2.174404761904762 [0 0 0 0 1 1 0 2 2 2 0 0 3 1]
    2.174404761904762 [0 0 0 0 1 1 2 0 2 2 0 0 3 1]
    2.174404761904762 [0 0 0 0 1 1 2 2 0 0 0 2 3 1]
    2.1708333333333334 [0 0 0 0 1 1 2 2 0 2 0 0 3 1]
    2.1660714285714286 [0 0 0 0 1 1 2 2 2 0 0 0 3 1]
    2.1660714285714286 [0 0 0 0 3 1 2 2 2 0 0 0 1 1]
    2.1645833333333333 [0 0 0 1 0 1 2 2 2 0 0 0 3 1]
    2.145833333333333 [0 0 0 1 1 0 0 0 2 2 3 2 3 1]
    2.145833333333333 [0 0 0 1 1 0 0 2 2 0 3 2 3 1]
    2.145833333333333 [0 0 0 1 1 0 2 0 2 0 3 2 3 1]
    2.145833333333333 [0 0 0 1 1 0 2 2 0 2 3 0 3 1]
    2.1283333333333334 [0 0 0 1 1 1 0 0 0 2 2 2 3 1]
    2.1283333333333334 [0 0 0 1 1 1 0 0 2 0 2 2 3 1]
    2.128333333333333 [0 0 0 1 1 1 0 0 2 2 0 2 3 1]
    2.128333333333333 [0 0 0 1 1 1 0 0 2 2 2 0 3 1]
    2.128333333333333 [0 0 0 1 1 1 0 2 0 0 2 2 3 1]
    2.128333333333333 [0 0 0 1 1 1 0 2 0 2 2 0 3 1]
    2.128333333333333 [0 0 0 1 1 1 2 0 0 0 2 2 3 1]
    2.128333333333333 [0 0 0 1 1 1 2 0 0 2 2 0 3 1]
    2.128333333333333 [0 0 0 1 1 1 2 2 0 2 0 0 3 1]
    2.0416666666666665 [0 0 0 1 3 1 0 0 0 2 2 2 3 1]
    2.0416666666666665 [0 0 0 1 3 1 0 0 2 0 2 2 3 1]
    2.0416666666666665 [0 0 0 1 3 1 0 0 2 2 0 2 3 1]
    2.0416666666666665 [0 0 0 1 3 1 0 2 0 2 0 2 3 1]
    2.0416666666666665 [0 0 0 1 3 1 0 2 0 2 2 0 3 1]
    2.0416666666666665 [0 0 0 1 3 1 0 2 2 0 0 2 3 1]
    2.0416666666666665 [0 0 0 1 3 1 0 2 2 0 2 0 3 1]
    2.0416666666666665 [0 0 0 1 3 1 2 0 0 2 0 2 3 1]
    2.0416666666666665 [0 0 0 1 3 1 2 0 0 2 2 0 3 1]
    2.0416666666666665 [0 0 0 1 3 1 2 0 2 0 0 2 3 1]
    2.0416666666666665 [0 0 0 1 3 1 2 0 2 0 2 0 3 1]
    2.0416666666666665 [0 0 0 1 3 1 2 2 0 0 2 0 3 1]
    2.0416666666666665 [0 0 0 1 3 1 2 2 2 0 0 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 0 0 0 2 0 2 3 1]
    2.0416666666666665 [0 0 2 1 3 1 0 0 0 2 2 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 0 0 2 0 0 2 3 1]
    2.0416666666666665 [0 0 2 1 3 1 0 0 2 0 2 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 0 2 0 0 2 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 0 2 0 2 0 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 0 2 2 0 0 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 2 0 0 0 2 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 2 0 0 2 0 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 2 0 2 0 0 0 3 1]
    2.0416666666666665 [0 0 2 1 3 1 2 2 0 0 0 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 0 0 0 2 0 2 3 1]
    2.0416666666666665 [0 2 0 1 3 1 0 0 0 2 2 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 0 0 2 0 0 2 3 1]
    2.0416666666666665 [0 2 0 1 3 1 0 0 2 0 2 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 0 2 0 0 2 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 0 2 0 2 0 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 0 2 2 0 0 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 2 0 0 0 2 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 2 0 0 2 0 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 2 0 2 0 0 0 3 1]
    2.0416666666666665 [0 2 0 1 3 1 2 2 0 0 0 0 3 1]
    2.0416666666666665 [0 2 2 1 3 1 0 0 0 0 2 0 3 1]
    2.0416666666666665 [0 2 2 1 3 1 0 0 0 2 0 0 3 1]
    2.0416666666666665 [0 2 2 1 3 1 0 0 2 0 0 0 3 1]
    2.0416666666666665 [0 2 2 1 3 1 0 2 0 0 0 0 3 1]
    2.0416666666666665 [0 2 2 1 3 1 2 0 0 0 0 0 3 1]
    2.0416666666666665 [2 0 0 1 3 1 0 0 0 0 2 2 3 1]
    2.0416666666666665 [2 0 0 1 3 1 0 0 0 2 0 2 3 1]
    2.0416666666666665 [2 0 0 1 3 1 0 0 0 2 2 0 3 1]
    2.0416666666666665 [2 0 0 1 3 1 0 0 2 0 2 0 3 1]
    2.0416666666666665 [2 0 0 1 3 1 0 0 2 2 0 0 3 1]
    2.0416666666666665 [2 0 0 1 3 1 0 2 0 0 0 2 3 1]
    2.0416666666666665 [2 0 0 1 3 1 0 2 0 2 0 0 3 1]
    2.0416666666666665 [2 0 0 1 3 1 0 2 2 0 0 0 3 1]
    2.0416666666666665 [2 0 0 1 3 1 2 0 0 0 0 2 3 1]
    2.0416666666666665 [2 0 0 1 3 1 2 0 0 2 0 0 3 1]
    2.0416666666666665 [2 0 0 1 3 1 2 0 2 0 0 0 3 1]
    2.0416666666666665 [2 0 2 1 3 1 0 0 0 0 0 2 3 1]
    2.0416666666666665 [2 0 2 1 3 1 0 0 0 0 2 0 3 1]
    2.0416666666666665 [2 0 2 1 3 1 0 0 0 2 0 0 3 1]
    2.0416666666666665 [2 0 2 1 3 1 0 0 2 0 0 0 3 1]
    2.0416666666666665 [2 2 0 1 3 1 0 0 0 0 0 2 3 1]
    2.0416666666666665 [2 2 0 1 3 1 0 0 0 0 2 0 3 1]
    2.0416666666666665 [2 2 0 1 3 1 0 0 0 2 0 0 3 1]
    2.0416666666666665 [2 2 0 1 3 1 0 0 2 0 0 0 3 1]
    2.0416666666666665 [2 2 2 1 3 1 0 0 0 0 0 0 3 1]
    2.0416666666666665 [3 0 0 3 1 0 0 0 2 2 0 2 3 1]
    2.0416666666666665 [3 0 0 3 1 0 0 2 0 2 0 2 3 1]
    2.0416666666666665 [3 0 0 3 1 0 0 2 0 2 2 0 3 1]
    2.0416666666666665 [3 0 0 3 1 0 0 2 2 0 0 2 3 1]
    2.0416666666666665 [3 0 0 3 1 0 0 2 2 0 2 0 3 1]
    2.0416666666666665 [3 0 0 3 1 0 2 0 0 2 0 2 3 1]
    2.0416666666666665 [3 0 0 3 1 0 2 0 0 2 2 0 3 1]
    2.0416666666666665 [3 0 0 3 1 0 2 0 2 0 0 2 3 1]
    2.0416666666666665 [3 0 0 3 1 0 2 0 2 0 2 0 3 1]
    2.0416666666666665 [3 0 0 3 1 0 2 2 0 0 2 0 3 1]
    2.0416666666666665 [3 0 0 3 1 0 2 2 2 0 0 0 3 1]
    2.0416666666666665 [3 0 0 3 1 1 0 0 0 2 0 2 3 2]
    2.0416666666666665 [3 0 0 3 1 1 0 0 0 2 2 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 0 0 0 2 2 2 3 0]
    2.0416666666666665 [3 0 0 3 1 1 0 0 2 0 0 2 3 2]
    2.0416666666666665 [3 0 0 3 1 1 0 0 2 0 2 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 0 0 2 0 2 2 3 0]
    2.0416666666666665 [3 0 0 3 1 1 0 0 2 2 0 2 3 0]
    2.0416666666666665 [3 0 0 3 1 1 0 2 0 0 2 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 0 2 0 2 0 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 0 2 0 2 0 2 3 0]
    2.0416666666666665 [3 0 0 3 1 1 0 2 0 2 2 0 3 0]
    2.0416666666666665 [3 0 0 3 1 1 0 2 2 0 0 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 0 2 2 0 0 2 3 0]
    2.0416666666666665 [3 0 0 3 1 1 0 2 2 0 2 0 3 0]
    2.0416666666666665 [3 0 0 3 1 1 0 2 2 2 0 0 3 0]
    2.0416666666666665 [3 0 0 3 1 1 2 0 0 0 2 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 2 0 0 2 0 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 2 0 0 2 0 2 3 0]
    2.0416666666666665 [3 0 0 3 1 1 2 0 0 2 2 0 3 0]
    2.0416666666666665 [3 0 0 3 1 1 2 0 2 0 0 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 2 0 2 0 0 2 3 0]
    2.0416666666666665 [3 0 0 3 1 1 2 0 2 0 2 0 3 0]
    2.0416666666666665 [3 0 0 3 1 1 2 0 2 2 0 0 3 0]
    2.0416666666666665 [3 0 0 3 1 1 2 2 0 0 0 0 3 2]
    2.0416666666666665 [3 0 0 3 1 1 2 2 0 0 0 2 3 0]
    2.0416666666666665 [3 0 0 3 1 1 2 2 0 0 2 0 3 0]
    2.0416666666666665 [3 0 0 3 1 1 2 2 2 0 0 0 3 0]
    2.0416666666666665 [3 0 0 3 3 0 0 0 2 2 0 2 1 1]
    2.0416666666666665 [3 0 0 3 3 0 0 2 0 2 0 2 1 1]
    2.0416666666666665 [3 0 0 3 3 0 0 2 0 2 2 0 1 1]
    2.0416666666666665 [3 0 0 3 3 0 0 2 2 0 0 2 1 1]
    2.0416666666666665 [3 0 0 3 3 0 0 2 2 0 2 0 1 1]
    2.0416666666666665 [3 0 0 3 3 0 2 0 0 2 0 2 1 1]
    2.0416666666666665 [3 0 0 3 3 0 2 0 0 2 2 0 1 1]
    2.0416666666666665 [3 0 0 3 3 0 2 0 2 0 0 2 1 1]
    2.0416666666666665 [3 0 0 3 3 0 2 0 2 0 2 0 1 1]
    2.0416666666666665 [3 0 0 3 3 0 2 2 0 0 2 0 1 1]
    2.0416666666666665 [3 0 0 3 3 0 2 2 2 0 0 0 1 1]
    2.0416666666666665 [3 0 0 3 3 1 0 0 0 2 0 2 1 2]
    2.0416666666666665 [3 0 0 3 3 1 0 0 0 2 2 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 0 0 0 2 2 2 1 0]
    2.0416666666666665 [3 0 0 3 3 1 0 0 2 0 0 2 1 2]
    2.0416666666666665 [3 0 0 3 3 1 0 0 2 0 2 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 0 0 2 0 2 2 1 0]
    2.0416666666666665 [3 0 0 3 3 1 0 0 2 2 0 2 1 0]
    2.0416666666666665 [3 0 0 3 3 1 0 2 0 0 2 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 0 2 0 2 0 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 0 2 0 2 0 2 1 0]
    2.0416666666666665 [3 0 0 3 3 1 0 2 0 2 2 0 1 0]
    2.0416666666666665 [3 0 0 3 3 1 0 2 2 0 0 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 0 2 2 0 0 2 1 0]
    2.0416666666666665 [3 0 0 3 3 1 0 2 2 0 2 0 1 0]
    2.0416666666666665 [3 0 0 3 3 1 0 2 2 2 0 0 1 0]
    2.0416666666666665 [3 0 0 3 3 1 2 0 0 0 2 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 2 0 0 2 0 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 2 0 0 2 0 2 1 0]
    2.0416666666666665 [3 0 0 3 3 1 2 0 0 2 2 0 1 0]
    2.0416666666666665 [3 0 0 3 3 1 2 0 2 0 0 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 2 0 2 0 0 2 1 0]
    2.0416666666666665 [3 0 0 3 3 1 2 0 2 0 2 0 1 0]
    2.0416666666666665 [3 0 0 3 3 1 2 0 2 2 0 0 1 0]
    2.0416666666666665 [3 0 0 3 3 1 2 2 0 0 0 0 1 2]
    2.0416666666666665 [3 0 0 3 3 1 2 2 0 0 0 2 1 0]
    2.0416666666666665 [3 0 0 3 3 1 2 2 0 0 2 0 1 0]
    2.0416666666666665 [3 0 0 3 3 1 2 2 2 0 0 0 1 0]
    2.0416666666666665 [3 0 2 3 1 0 0 0 0 2 2 0 3 1]
    2.0416666666666665 [3 0 2 3 1 0 0 0 2 0 2 0 3 1]
    2.0416666666666665 [3 0 2 3 1 0 0 2 0 2 0 0 3 1]
    2.0416666666666665 [3 0 2 3 1 0 0 2 2 0 0 0 3 1]
    2.0416666666666665 [3 0 2 3 1 0 2 0 0 2 0 0 3 1]
    2.0416666666666665 [3 0 2 3 1 0 2 0 2 0 0 0 3 1]
    2.0416666666666665 [3 0 2 3 1 0 2 2 0 0 0 0 3 1]
    2.0416666666666665 [3 0 2 3 1 1 0 0 0 2 0 0 3 2]
    2.0416666666666665 [3 0 2 3 1 1 0 0 0 2 0 2 3 0]
    2.0416666666666665 [3 0 2 3 1 1 0 0 0 2 2 0 3 0]
    2.0416666666666665 [3 0 2 3 1 1 0 0 2 0 0 0 3 2]
    2.0416666666666665 [3 0 2 3 1 1 0 0 2 0 0 2 3 0]
    2.0416666666666665 [3 0 2 3 1 1 0 0 2 0 2 0 3 0]
    2.0416666666666665 [3 0 2 3 1 1 0 2 0 0 0 0 3 2]
    2.0416666666666665 [3 0 2 3 1 1 0 2 0 0 2 0 3 0]
    2.0416666666666665 [3 0 2 3 1 1 0 2 0 2 0 0 3 0]
    2.0416666666666665 [3 0 2 3 1 1 0 2 2 0 0 0 3 0]
    2.0416666666666665 [3 0 2 3 1 1 2 0 0 0 0 0 3 2]
    2.0416666666666665 [3 0 2 3 1 1 2 0 0 0 2 0 3 0]
    2.0416666666666665 [3 0 2 3 1 1 2 0 0 2 0 0 3 0]
    2.0416666666666665 [3 0 2 3 1 1 2 0 2 0 0 0 3 0]
    2.0416666666666665 [3 0 2 3 1 1 2 2 0 0 0 0 3 0]
    2.0416666666666665 [3 0 2 3 3 0 0 0 0 2 2 0 1 1]
    2.0416666666666665 [3 0 2 3 3 0 0 0 2 0 2 0 1 1]
    2.0416666666666665 [3 0 2 3 3 0 0 2 0 2 0 0 1 1]
    2.0416666666666665 [3 0 2 3 3 0 0 2 2 0 0 0 1 1]
    2.0416666666666665 [3 0 2 3 3 0 2 0 0 2 0 0 1 1]
    2.0416666666666665 [3 0 2 3 3 0 2 0 2 0 0 0 1 1]
    2.0416666666666665 [3 0 2 3 3 0 2 2 0 0 0 0 1 1]
    2.0416666666666665 [3 0 2 3 3 1 0 0 0 2 0 0 1 2]
    2.0416666666666665 [3 0 2 3 3 1 0 0 0 2 0 2 1 0]
    2.0416666666666665 [3 0 2 3 3 1 0 0 0 2 2 0 1 0]
    2.0416666666666665 [3 0 2 3 3 1 0 0 2 0 0 0 1 2]
    2.0416666666666665 [3 0 2 3 3 1 0 0 2 0 0 2 1 0]
    2.0416666666666665 [3 0 2 3 3 1 0 0 2 0 2 0 1 0]
    2.0416666666666665 [3 0 2 3 3 1 0 2 0 0 0 0 1 2]
    2.0416666666666665 [3 0 2 3 3 1 0 2 0 0 2 0 1 0]
    2.0416666666666665 [3 0 2 3 3 1 0 2 0 2 0 0 1 0]
    2.0416666666666665 [3 0 2 3 3 1 0 2 2 0 0 0 1 0]
    2.0416666666666665 [3 0 2 3 3 1 2 0 0 0 0 0 1 2]
    2.0416666666666665 [3 0 2 3 3 1 2 0 0 0 2 0 1 0]
    2.0416666666666665 [3 0 2 3 3 1 2 0 0 2 0 0 1 0]
    2.0416666666666665 [3 0 2 3 3 1 2 0 2 0 0 0 1 0]
    2.0416666666666665 [3 0 2 3 3 1 2 2 0 0 0 0 1 0]
    2.0416666666666665 [3 2 0 3 1 0 0 0 0 2 2 0 3 1]
    2.0416666666666665 [3 2 0 3 1 0 0 0 2 0 2 0 3 1]
    2.0416666666666665 [3 2 0 3 1 0 0 2 0 2 0 0 3 1]
    2.0416666666666665 [3 2 0 3 1 0 0 2 2 0 0 0 3 1]
    2.0416666666666665 [3 2 0 3 1 0 2 0 0 2 0 0 3 1]
    2.0416666666666665 [3 2 0 3 1 0 2 0 2 0 0 0 3 1]
    2.0416666666666665 [3 2 0 3 1 0 2 2 0 0 0 0 3 1]
    2.0416666666666665 [3 2 0 3 1 1 0 0 0 2 0 0 3 2]
    2.0416666666666665 [3 2 0 3 1 1 0 0 0 2 0 2 3 0]
    2.0416666666666665 [3 2 0 3 1 1 0 0 0 2 2 0 3 0]
    2.0416666666666665 [3 2 0 3 1 1 0 0 2 0 0 0 3 2]
    2.0416666666666665 [3 2 0 3 1 1 0 0 2 0 0 2 3 0]
    2.0416666666666665 [3 2 0 3 1 1 0 0 2 0 2 0 3 0]
    2.0416666666666665 [3 2 0 3 1 1 0 2 0 0 0 0 3 2]
    2.0416666666666665 [3 2 0 3 1 1 0 2 0 0 2 0 3 0]
    2.0416666666666665 [3 2 0 3 1 1 0 2 0 2 0 0 3 0]
    2.0416666666666665 [3 2 0 3 1 1 0 2 2 0 0 0 3 0]
    2.0416666666666665 [3 2 0 3 1 1 2 0 0 0 0 0 3 2]
    2.0416666666666665 [3 2 0 3 1 1 2 0 0 0 2 0 3 0]
    2.0416666666666665 [3 2 0 3 1 1 2 0 0 2 0 0 3 0]
    2.0416666666666665 [3 2 0 3 1 1 2 0 2 0 0 0 3 0]
    2.0416666666666665 [3 2 0 3 1 1 2 2 0 0 0 0 3 0]
    2.0416666666666665 [3 2 0 3 3 0 0 0 0 2 2 0 1 1]
    2.0416666666666665 [3 2 0 3 3 0 0 0 2 0 2 0 1 1]
    2.0416666666666665 [3 2 0 3 3 0 0 2 0 2 0 0 1 1]
    2.0416666666666665 [3 2 0 3 3 0 0 2 2 0 0 0 1 1]
    2.0416666666666665 [3 2 0 3 3 0 2 0 0 2 0 0 1 1]
    2.0416666666666665 [3 2 0 3 3 0 2 0 2 0 0 0 1 1]
    2.0416666666666665 [3 2 0 3 3 0 2 2 0 0 0 0 1 1]
    2.0416666666666665 [3 2 0 3 3 1 0 0 0 2 0 0 1 2]
    2.0416666666666665 [3 2 0 3 3 1 0 0 0 2 0 2 1 0]
    2.0416666666666665 [3 2 0 3 3 1 0 0 0 2 2 0 1 0]
    2.0416666666666665 [3 2 0 3 3 1 0 0 2 0 0 0 1 2]
    2.0416666666666665 [3 2 0 3 3 1 0 0 2 0 0 2 1 0]
    2.0416666666666665 [3 2 0 3 3 1 0 0 2 0 2 0 1 0]
    2.0416666666666665 [3 2 0 3 3 1 0 2 0 0 0 0 1 2]
    2.0416666666666665 [3 2 0 3 3 1 0 2 0 0 2 0 1 0]
    2.0416666666666665 [3 2 0 3 3 1 0 2 0 2 0 0 1 0]
    2.0416666666666665 [3 2 0 3 3 1 0 2 2 0 0 0 1 0]
    2.0416666666666665 [3 2 0 3 3 1 2 0 0 0 0 0 1 2]
    2.0416666666666665 [3 2 0 3 3 1 2 0 0 0 2 0 1 0]
    2.0416666666666665 [3 2 0 3 3 1 2 0 0 2 0 0 1 0]
    2.0416666666666665 [3 2 0 3 3 1 2 0 2 0 0 0 1 0]
    2.0416666666666665 [3 2 0 3 3 1 2 2 0 0 0 0 1 0]
    2.0416666666666665 [3 2 2 3 1 0 0 0 0 2 0 0 3 1]
    2.0416666666666665 [3 2 2 3 1 0 0 0 2 0 0 0 3 1]
    2.0416666666666665 [3 2 2 3 1 1 0 0 0 2 0 0 3 0]
    2.0416666666666665 [3 2 2 3 1 1 0 0 2 0 0 0 3 0]
    2.0416666666666665 [3 2 2 3 1 1 0 2 0 0 0 0 3 0]
    2.0416666666666665 [3 2 2 3 1 1 2 0 0 0 0 0 3 0]
    2.0416666666666665 [3 2 2 3 3 0 0 0 0 2 0 0 1 1]
    2.0416666666666665 [3 2 2 3 3 0 0 0 2 0 0 0 1 1]
    2.0416666666666665 [3 2 2 3 3 1 0 0 0 2 0 0 1 0]
    2.0416666666666665 [3 2 2 3 3 1 0 0 2 0 0 0 1 0]
    2.0416666666666665 [3 2 2 3 3 1 0 2 0 0 0 0 1 0]
    2.0416666666666665 [3 2 2 3 3 1 2 0 0 0 0 0 1 0]



```python

```
