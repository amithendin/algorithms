import numpy as np
from ca_engine import CAEngine

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def get_state_color(state):
    if state == 0:
        return BLACK
    elif state == 1:
        return WHITE
    elif state == 2:
        return RED
    elif state == 3:
        return GREEN
    elif state == 4:
        return BLUE
    return BLACK

grid = np.array([
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
], dtype=int).reshape(30,30).T

def update_cell(x, y, ca_engine):
    u = ca_engine.get_cell(x,y-1)
    ul = ca_engine.get_cell(x-1,y-1)
    ur = ca_engine.get_cell(x+1,y-1)
    l = ca_engine.get_cell(x-1,y)
    m = ca_engine.get_cell(x,y)
    r = ca_engine.get_cell(x+1,y)
    d = ca_engine.get_cell(x,y+1)
    dl = ca_engine.get_cell(x-1,y+1)
    dr = ca_engine.get_cell(x+1,y+1)

    v = 0

    if m:
        #tips of the cross cell
        if u and not(l) and not(r) and not(d) and not(dl) and not(dr):
            v = 2
        if not(u) and not(ur) and l and not(r) and not(d) and not(dr):
            v = 2
        if not(u) and not(ul) and not(l) and r and not(d) and not(dl):
            v = 2
        if not(u) and not(ul) and not(ur) and not(l) and not(r) and d:
            v = 2

        #vertical middle survive
        if u and not(l) and not(r) and d and not(dl) and not(dr):
            v = 1
        if u and not(l) and not(r) and d and not(ul) and not(ur):
            v = 1

        #horizontal middle survive
        if not(u) and l and r and not(d) and not(dl) and not(ul):
            v = 1
        if not(u) and l and r and not(d) and not(dr) and not(ur):
            v = 1

        #vertical middle color
        if u==2 and d and not(l) and not(r):
            v = 3
        if u==3 and d and not (l) and not (r):
            v = 2

        if u and d==2 and not(l) and not(r):
            v = 3
        if u and d==3 and not (l) and not (r):
            v = 2

        #horizontal middle color
        if not(u) and l==2 and r and not(d):
            v = 3
        if not (u) and l==3 and r and not (d):
            v = 2

        if not(u) and l and r==2 and not(d):
            v = 3
        if not (u) and l and r==3 and not (d):
            v = 2

        #center survive
        if u==d and l==r and l==u and u and not(ul) and not(ur) and not(dl) and not(dr):
            v = 1
        if u and d and l and r and not(ul) and not(ur) and not(dl) and not(dr) and (l != r or d != u):
            v = 4

        #cleanup
        if l == 4 or r == 4 or u == 4 or d == 4:
            v = 4
        if m == 4:
            v = 0

    return v


engine = CAEngine(
    cell_size=10, fps=10,
    state_color_fn=get_state_color,
    grid=grid,
    update_callback=update_cell
)

engine.start()