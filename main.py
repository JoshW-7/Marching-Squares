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


def draw_cell(cell=[0, 0, 0, 0], position=(0, 0), color=(255, 255, 255), threshold=0):
	global view_x, view_y

	state = get_state(cell)

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
	}.get(state, [])

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
	}.get(state, ())
	
	x_offset = width/size_x/2
	y_offset = height/size_y/2

	# Draw the border for this cell
	for line in lines:
		start = [
			int(view_x + position[0] + line[0][0]*x_offset),
			int(view_y + position[1] + line[0][1]*y_offset)
		]
		end = [
			int(view_x + position[0] + line[1][0]*x_offset),
			int(view_y + position[1] + line[1][1]*y_offset)
		]
		pygame.draw.line(screen, (255, 255, 255), start, end, 4)

	# Don't draw polygons for state 0
	if state == 0:
		return

	# Fill in the solid area of this cell
	points = []
	for point in polygon:
		points.append((
			int(view_x + position[0] + point[0]*x_offset),
			int(view_y + position[1] + point[1]*y_offset)
		))
	pygame.draw.polygon(screen, color, points)


# Initialization
pygame.init()
width = 500
height = 500
screen = pygame.display.set_mode([width, height])

# Options
size_x = 100
size_y = 100
size_z = 100
threshold = 0

# Generate 3D Perlin Noise
fields = generate_perlin_noise_3d((size_x, size_y, size_z), (5, 5, 5))

view_x = 0
view_y = 0
colors = list(Color("red").range_to(Color("blue"), size_z))
clock = pygame.time.Clock()
running = True
counter = 0
index = 0
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	# Limit fps
	clock.tick(60)

	# Draw a black background
	screen.fill((0, 0, 0))

	# Update the color based on the z index
	color = colors[index].rgb

	# Draw each cell in the current z index
	for y,row in enumerate(fields[index]):
		if y >= size_y - 1:
			continue
		for x,cell in enumerate(row):
			if x >= size_x - 1:
				continue
			draw_cell(
				cell=[
					convert(row[x], threshold=threshold),
					convert(row[x+1], threshold=threshold),
					convert(fields[index][y+1][x+1], threshold=threshold),
					convert(fields[index][y+1][x], threshold=threshold)
				],
				position=(int(width/size_x/2 + x*width/size_x), int(height/size_y/2 + y*height/size_y)),
				color=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
			)

	# Cycle through the z indices
	index += 1
	if index >= len(fields):
		index = 0

	pygame.display.flip()
