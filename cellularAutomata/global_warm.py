import numpy as np
import matplotlib.pyplot as plt
import util
from ca_engine import CAEngine

NUM_DAYS = 365
PATH_TO_GRID_FILE = 'earth0'
RAND_SEED = 42
FRAMES_PER_SECOND = 20

# elevation 1-...2000 heighest mountain,
# cell type 0-barren ground, 1-water body, 2-vegetation, 3-city/town, 4-ice,
# wind x speed (1,....60 kph),
# wind y speed (1,....60 kph),
# air pollution level (1 - clean air, 2 -... 100 unbreathable toxic hellscape)
# temp (-100,... 100)
# cloudiness level (1,... 100)

ELEVATION_RANGE = (1, 2000)
TILE_TYPE_RANGE = (0, 4)
POLLUTION_CLOUD_RANGE = (0,100)
WIND_SPEED_RANGE = (0, 60)
TEMP_RANGE = (-100, 100)

GROUND_COLOR = [239, 169, 119]
WATER_COLOR = [0, 0, 255]
ICE_COLOR = [0, 255, 255]
CLOUD_COLOR = np.array([230, 230, 230], dtype=float)
VEGETATION_COLOR = [0, 255, 0]
URBAN_COLOR = [255, 0, 255]
POLLUTION_COLOR = np.array([203, 0, 218], dtype=float)

# read grid from file
def load_config(config_path, rand_seed):
    np.random.seed(rand_seed)
    f = open(config_path, 'r')
    lines = f.readlines()
    cell_types = []
    for l in lines:
        row = [ int(x) for x in l.split(',') if x.isdigit() ]
        cell_types.append(row)
    cell_types = np.array(cell_types).T

    GRID_SHAPE = (cell_types.shape[0], cell_types.shape[1], 7)
    grid = np.zeros(GRID_SHAPE, dtype=float)
    grid[:, :, 1] = np.array(cell_types)

    #generate terrain
    grid[:, :, 0] = np.random.choice(ELEVATION_RANGE[1], GRID_SHAPE[0:2]) + ELEVATION_RANGE[0]
    #plants cat grow at heigher elevation
    plants = grid[grid[:, :, 1] == 2]
    plants[:, 0] = np.clip(plants[:, 0], 0, int(ELEVATION_RANGE[1]/2))
    kernel = np.array([
        1,1,1,
        1,0,1,
        1,1,1
    ]).reshape(3,3)*(1/9)
    grid[:, :, 0] = util.convolve2d(grid[:, :, 0], kernel) #smooth out the height values to get more representative of real terrain

    # #generate wind: speed and direction
    grid[:, :, 2] = np.random.choice(3, GRID_SHAPE[0:2]) -1 #choose dir
    grid[:, :, 3] = np.random.choice(3, GRID_SHAPE[0:2]) -1 #choose dir
    grid[:,:,2:4] *= np.random.choice(WIND_SPEED_RANGE[1]+1, GRID_SHAPE[0:2]+(1,)) + WIND_SPEED_RANGE[0] #apply speed

    #generate clouds and pollution
    clouds_prob_dist = [1 / (2 ** x) for x in range(1, int(POLLUTION_CLOUD_RANGE[1] / 2) + 1)]
    clouds_prob_dist = list(reversed(clouds_prob_dist)) + clouds_prob_dist
    clouds_prob_dist = np.array(clouds_prob_dist) / 2
    grid[:, :, 6] = np.random.choice(POLLUTION_CLOUD_RANGE[1], GRID_SHAPE[0:2], p=clouds_prob_dist) + POLLUTION_CLOUD_RANGE[0]
    grid[:, :, 6] /= 4

    grid[:,:, 4] = 0

    #generate temps
    #grid[:, :, 5] = np.random.choice(TEMP_RANGE[1]*2+1, GRID_SHAPE[0:2]) + TEMP_RANGE[0]
    grid[:, :, 5] = 20
    grid[grid[:,:,1]==1, 5]= 0
    grid[grid[:,:,1]==4, 5]= -20 # freeze the ice

    return grid


def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 * norm_vector2 == 0:
        return 0

    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    angle_in_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_in_degrees = np.degrees(angle_in_radians)
    angle_sign = 1 if angle_in_radians > 0 else -1

    return angle_sign*(round(angle_in_degrees) % 180)/180


def state_color(state):
    elevation_percent = state[0]/ELEVATION_RANGE[1]
    pollution_percent = state[4]/POLLUTION_CLOUD_RANGE[1]
    cloud_percent = state[6]/POLLUTION_CLOUD_RANGE[1]

    OUT_COLOR = GROUND_COLOR
    if state[1] == 1:
        OUT_COLOR = WATER_COLOR
    elif state[1] == 2:
        OUT_COLOR = VEGETATION_COLOR
    elif state[1] == 3:
        OUT_COLOR = URBAN_COLOR
    elif state[1] == 4:
        OUT_COLOR = ICE_COLOR

    OUT_COLOR = np.array(OUT_COLOR, dtype=float)*0.5 + CLOUD_COLOR*cloud_percent*0.25 + POLLUTION_COLOR*pollution_percent*0.25

    if np.max(OUT_COLOR) > 255:
        OUT_COLOR = OUT_COLOR/np.max(OUT_COLOR)*255
    #OUT_COLOR *= elevation_percent

    if OUT_COLOR[0] < 0 or OUT_COLOR[1] < 0 or OUT_COLOR[2] < 0:
        pass

    return OUT_COLOR


def update_cell(x,y, ca_engine):
    u = ca_engine.get_cell(x, y - 1)
    ul = ca_engine.get_cell(x - 1, y - 1)
    ur = ca_engine.get_cell(x + 1, y - 1)
    l = ca_engine.get_cell(x - 1, y)
    m = ca_engine.get_cell(x, y)
    r = ca_engine.get_cell(x + 1, y)
    d = ca_engine.get_cell(x, y + 1)
    dl = ca_engine.get_cell(x - 1, y + 1)
    dr = ca_engine.get_cell(x + 1, y + 1)
    neighbors = [u,d,l,r,ul,ur,dl,dr]
    neighbors_vecs = [ [0,-1],[0,1],[1,0],[-1,0],[1,-1],[-1,-1],[1,1],[-1,1] ]
    v = np.copy(m)

    # melt the ice
    if m[1] == 4 and m[5] > 0:
        v[1] = 1 # change cell type to water


    # calculate temperature, clouds reduce temperature and pollution increases temperature
    v[5] += 0.25 * (m[4]/POLLUTION_CLOUD_RANGE[1])
    if m[1] == 4: # make ice cool the temperature
        v[5] -=  0.25 * abs(m[5]/TEMP_RANGE[1])
    #if m[0] > ELEVATION_RANGE[1]*0.5 and m[5] > -10: # it's colder at higher elevation
        #v[5] -=  0.05 * abs(m[5]/TEMP_RANGE[1])

    # clouds regulate temperature by mitigating change
    v[5] = m[5] + ((v[5]-m[5])*10) * (m[6] / POLLUTION_CLOUD_RANGE[1]) * 0.5

    # create clouds from heating bodies of water
    if (m[1] == 4 or m[1] == 1) and v[5] > m[5]:
        v[6] = m[6] + 3*(v[5]-m[5])/TEMP_RANGE[1]
    # reduce clouds from cooling areas
    elif m[5] > v[5]:
        v[6] = m[6] - 2*(m[5]-v[5])/TEMP_RANGE[1]

    for i in range(len(neighbors)):
        n = neighbors[i]
        nv = neighbors_vecs[i]
        n_angle_to_m = angle_between_vectors(n[2:4], np.array(nv, dtype=float))
        m_angle_to_n = angle_between_vectors(m[2:4], np.array(nv, dtype=float)*-1)
        elevation_delta = m[0] - n[0]

        if n_angle_to_m >= 0 and n_angle_to_m <= 0.25:
            # apply wind from neighbor
            input_wind = n[2:4] * n_angle_to_m
            #if neighbor is higher than me, accelerate wind recived
            if elevation_delta < 0:
                input_wind *= 1 + abs(elevation_delta / ELEVATION_RANGE[1])
            v[2:4] += input_wind

            # apply pollution received from neighbor
            v[4] += max(n[4] * n_angle_to_m, 0)
            # apply clouds received from neighbor
            v[6] += max(n[6] * n_angle_to_m, 0)

        if m_angle_to_n >= 0 and m_angle_to_n <= 0.25:
            # blow away the wind
            input_wind = m[2:4] * m_angle_to_n
            # if neighbor is lower than me, decelerate wind received
            if elevation_delta > 0:
                input_wind *= 1 - abs(elevation_delta / ELEVATION_RANGE[1])
            v[2:4] -= input_wind
            # remove pollution given to neighbor
            v[4] -= min(m[4] * m_angle_to_n, v[4])
            # remove clouds given to neighbor
            v[6] -= min(m[6] * m_angle_to_n, v[6])

    # decelerate wind is you are city or forest
    if v[1] == 3 or v[1] == 2:
        v[2:4] *= 0.5

    # increase pollution if you are a city
    if v[1] == 3:
        v[4] = min(v[4]+4, POLLUTION_CLOUD_RANGE[1]) # it's harder to burn things at higher elevation
    # decrease pollution if you are a forest
    elif v[1] == 2:
        v[4] = max(v[4]-1,0)

    # random wind factor
    v[2:4] += np.random.uniform(low=-1,high=1) * 10
    # make sure wind does not exceed range
    if np.max(v[2:4]) != 0:
        v[2:4] = (v[2:4]/np.max(np.abs(v[2:4])))*WIND_SPEED_RANGE[1]

    return v


def post_update(ca_engine):
    global day
    global grid_cell_mean

    grid_cell_mean[day] = np.mean( ca_engine.grid.reshape((grid.shape[0]*grid.shape[1], grid.shape[2])), axis=0 )

    if day % 100 == 0:
        mean_world_temp = np.mean(ca_engine.grid[:, :, 5])
        mean_world_clouds = np.mean(ca_engine.grid[:, :, 6])
        mean_world_pollution = np.mean(ca_engine.grid[:, :, 4])
        std_world_temp = np.std(ca_engine.grid[:, :, 5])
        std_world_clouds = np.std(ca_engine.grid[:, :, 6])
        std_world_pollution = np.std(ca_engine.grid[:, :, 4])
        print("\nDay: ",day,
            "\nmean temp: ", format(mean_world_temp, '.4f'),
            ", mean clouds: ", format(mean_world_clouds, '.4f'),
            ", mean pollution: ", format(mean_world_pollution, '.4f'),
            "\nstd temp: ", format(std_world_temp, '.4f'),
            ", std clouds: ", format(std_world_clouds, '.4f'),
            ", std pollution: ", format(std_world_pollution, '.4f'))
    day += 1

grid = load_config(PATH_TO_GRID_FILE, RAND_SEED)
day = 0
grid_cell_mean = np.zeros((NUM_DAYS, grid.shape[2]))

engine = CAEngine(
    cell_size=30, fps=FRAMES_PER_SECOND,
    cell_zero=np.array([0,1,0,0,0,0,0]),
    state_color_fn=state_color,
    grid=grid,
    update_callback=update_cell,
    post_update_callback=post_update,
    is_circular=True,
    animate=True
)

engine.start(num_iter=NUM_DAYS)

# standardize data
grid_data_std = (grid_cell_mean - np.mean(grid_cell_mean, axis=0))/np.std(grid_cell_mean, axis=0)

temp_data = grid_data_std[:, 5]
poll_data = grid_data_std[:, 4]
elev_data = grid_data_std[:, 0]

x = np.array(list(range(NUM_DAYS)))

plt.plot(x, temp_data, label="Temperature")
plt.plot(x, poll_data, label="Pollution")
plt.plot(x, elev_data, label="Elevation")

plt.xlabel('Days')
plt.ylabel('Temperature/Pollution/Elevation')
plt.title('Simulation earth0 (stable)')
plt.legend()

#plt.savefig('stats.png')

plt.show()