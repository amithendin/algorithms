import numpy as np
from ca_engine import CAEngine

def get_state_color(state):
    if state == 1:
        return (255, 255, 255)
    return (0, 0, 0)

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

    sum_neighbors = sum([u,ul,ur,d,dl,dr,l,r])

    if m == 1 and sum_neighbors == 2 or sum_neighbors == 3:
        return 1
    if m == 0 and sum_neighbors ==3:
        return 1

    return 0

grid = np.random.choice([0, 1], size=10000, p=[0.9, 0.1]).reshape(100,100)

engine = CAEngine(
    cell_size=20, fps=10,
    state_color_fn=get_state_color,
    grid=grid,
    update_callback=update_cell,
    animate=False
)

engine.start()