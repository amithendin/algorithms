import pygame
import numpy as np

class CAEngine(object):

    def _life_get_state_color(state):
        if state == 1:
            return (255, 255, 255)
        return (0, 0, 0)

    def _life__update_cell(x, y, ca_engine):
        u = ca_engine.get_cell(x, y - 1)
        ul = ca_engine.get_cell(x - 1, y - 1)
        ur = ca_engine.get_cell(x + 1, y - 1)
        l = ca_engine.get_cell(x - 1, y)
        m = ca_engine.get_cell(x, y)
        r = ca_engine.get_cell(x + 1, y)
        d = ca_engine.get_cell(x, y + 1)
        dl = ca_engine.get_cell(x - 1, y + 1)
        dr = ca_engine.get_cell(x + 1, y + 1)

        sum_neighbors = sum([u, ul, ur, d, dl, dr, l, r])

        if m == 1 and sum_neighbors == 2 or sum_neighbors == 3:
            return 1
        if m == 0 and sum_neighbors == 3:
            return 1

        return 0

    def __init__(self, cell_size, fps, grid, state_color_fn=_life_get_state_color, update_callback=_life__update_cell, cell_zero=0, is_circular=False, post_update_callback=None, animate=False):
        self.CELL_SIZE = cell_size
        self.FPS = fps
        self.state_color_fn = state_color_fn
        self.update_callback = update_callback
        self.post_update_callback = post_update_callback
        self.grid = grid
        self.cell_zero = cell_zero
        self.GRID_WIDTH = grid.shape[0]
        self.GRID_HEIGHT = grid.shape[1]
        self.is_circular = is_circular
        self.animate = animate
        self.WIDTH, self.HEIGHT = self.GRID_WIDTH*cell_size, self.GRID_HEIGHT*cell_size

    def get_cell(self, x, y):
        if self.is_circular:
            new_x = x
            if x < 0:
                new_x = self.GRID_WIDTH - 1
            elif x >= self.GRID_WIDTH:
                new_x = 0
            new_y = y
            if y < 0:
                new_y = self.GRID_HEIGHT - 1
            elif y >= self.GRID_HEIGHT:
                new_y = 0

            return self.grid[new_x, new_y]

        else:

            if x < 0 or y < 0 or x >= self.GRID_WIDTH or y >= self.GRID_HEIGHT:
                return self.cell_zero
            else:
                return self.grid[x, y]

    def update_grid(self):
        new_grid = np.zeros(self.grid.shape)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                new_grid[x, y] = self.update_callback(x,y,self)

        if self.post_update_callback is not None:
            self.post_update_callback(self)

        return new_grid


    def run(self, num_iter):
        for _ in range(num_iter):
            self.grid = self.update_grid()

    def start(self, num_iter=-1):
        pygame.init()
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Cellular Automata")
        clock = pygame.time.Clock()

        running = True
        while running and num_iter != 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if not self.animate and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.grid = self.update_grid()

            if self.animate:
                self.grid = self.update_grid()

            screen.fill((0,0,0))
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    color = self.state_color_fn(self.grid[x, y])
                    pygame.draw.rect(screen, color,
                                     (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

            pygame.display.flip()
            clock.tick(self.FPS)
            num_iter -= 1

        pygame.quit()