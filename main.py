import noise
import numpy as np
import pygame

from colour import Color


# Function code from https://github.com/pvigier/perlin-numpy
def generate_perlin_noise_3d(shape, res):
	def f(t):
		return 6*t**5 - 15*t**4 + 10*t**3

	delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
	d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
	grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
	grid = grid.transpose(1, 2, 3, 0) % 1

	# Gradients
	theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
	phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
	gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
	gradients[-1] = gradients[0]
	g000 = gradients[0:-1,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
	g100 = gradients[1:  ,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
	g010 = gradients[0:-1,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
	g110 = gradients[1:  ,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
	g001 = gradients[0:-1,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
	g101 = gradients[1:  ,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
	g011 = gradients[0:-1,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
	g111 = gradients[1:  ,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)

	# Ramps
	n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
	n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
	n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
	n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
	n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
	n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
	n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
	n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)

	# Interpolation
	t = f(grid)
	n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
	n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
	n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
	n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
	n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
	n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11

	return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)


def get_state(cell):
	return cell[0] * 8 + cell[1] * 4 + cell[2] * 2 + cell[3]


def convert(value, threshold=0.2):
	if value > threshold:
		return 1
	else:
		return 0


def draw_cell(state=0, position=(0,0), color=(255, 255, 255), value=0):
	global view_x, view_y
	lines = {
		1: [
			((-1, 0), (0, 1))
		],
		2: [
			((0, 1), (1, 0))
		],
		3: [
			((-1, 0), (1, 0))
		],
		4: [
			((0, -1), (1, 0))
		],
		5: [
			((-1, 0), (0, -1)),
			((0, 1), (1, 0))
		],
		6: [
			((0, -1), (0, 1))
		],
		7: [
			((-1, 0), (0, -1))
		],
		8: [
			((-1, 0), (0, -1))
		],
		9: [
			((0, -1), (0, 1))
		],
		10: [
			((-1, 0), (0, 1)),
			((0, -1), (1, 0))
		],
		11: [
			((0, -1), (1, 0))
		],
		12: [
			((-1, 0), (1, 0))
		],
		13: [
			((0, 1), (1, 0))
		],
		14: [
			((-1, 0), (0, 1))
		],
	}.get(state)

	polygon = {
		0: ((-1, -1), (1, -1), (1, 1), (-1, 1)),
		1: ((-1, 0), (-1, 1), (0, 1)),
		2: ((1, 0), (0, 1), (1, 1)),
		3: ((-1, 0), (1, 0), (1, 1), (-1, 1)),
		4: ((0, -1), (1, -1), (1, 0)),
		5: ((-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)),
		6: ((0, -1), (1, -1), (1, 1), (0, 1)),
		7: ((-1, 0), (0, -1), (1, -1), (1, 1), (-1, 1)),
		8: ((-1, 0), (-1, -1), (0, -1)),
		9: ((-1, -1), (0, -1), (0, 1), (-1, 1)),
		10: ((-1, -1), (0, -1), (1, 0), (1, 1), (0, 1), (-1, 0)),
		11: ((-1, -1), (0, -1), (1, 0), (1, 1), (-1, 1)),
		12: ((-1, -1), (1, -1), (1, 0), (-1, 0)),
		13: ((-1, -1), (1, -1), (1, 0), (0, 1), (-1, 1)),
		14: ((-1, -1), (1, -1), (1, 1), (0, 1), (-1, 0)),
		15: ((-1, -1), (1, -1), (1, 1), (-1, 1)),
	}.get(state)

	# offset_x = -2*value/width/size_x
	# offset_y = -2*value/height/size_y

	if lines:
		for line in lines:
			start = [
				int(view_x + position[0] + line[0][0]*width/size_x/2),
				int(view_y + position[1] + line[0][1]*height/size_y/2)
			]
			end = [
				int(view_x + position[0] + line[1][0]*width/size_x/2),
				int(view_y + position[1] + line[1][1]*height/size_y/2)
			]
			pygame.draw.line(screen, (255, 255, 255), start, end, 4)

	if state == 0:
		return

	color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
	points = []
	for point in polygon:
		points.append((
			int(view_x + position[0] + point[0]*width/size_x/2),
			int(view_y + position[1] + point[1]*height/size_y/2)
		))

	pygame.draw.polygon(screen, color, points)


# Initialization
pygame.init()
width = 500
height = 500
screen = pygame.display.set_mode([width, height])

# Generate 3D Perlin Noise
size_x = 60
size_y = 60
size_z = 60
fields = generate_perlin_noise_3d((size_x, size_y, size_z), (5, 5, 5))

# Create a matrix of cells, each of which constitutes a list of 4 values representing its corners
all_cells = []
for field in fields:
	cells = []
	for y,row in enumerate(field):
		cells.append([])
		for x,value in enumerate(row):
			if x < len(row)-1 and y < len(field)-1:
				cells[-1].append([
					convert(row[x]),
					convert(row[x+1]),
					convert(field[y+1][x+1]),
					convert(field[y+1][x]),
				])

	all_cells.append(cells)

view_x = size_x / 16
view_y = size_y / 16
colors = list(Color("red").range_to(Color("blue"), size_z))
clock = pygame.time.Clock()
running = True
counter = 0
index = 0
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	clock.tick(30)
	screen.fill((0, 0, 0))

	for y,row in enumerate(all_cells[index]):
		for x,cell in enumerate(row):
			draw_cell(
				state=get_state(cell),
				position=(int(width/size_x/2 + x*width/size_x), int(height/size_y/2 + y*height/size_y)),
				color=colors[index].rgb,
				value=fields[index][y][x],
			)

	index += 1
	if index >= len(all_cells):
		index = 0

	pygame.display.flip()
