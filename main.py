import noise
import numpy as np
import pygame

from colour import Color

# Examples located in root of this repository


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


def draw_cell(cell=[0, 0, 0, 0], position=(0, 0), color=(255, 255, 255), threshold=0):
	global view_x, view_y

	# Get the state of this cell by passing in 1's or 0's for each corner
	state = get_state([
		1 if cell[0] > threshold else 0,
		1 if cell[1] > threshold else 0,
		1 if cell[2] > threshold else 0,
		1 if cell[3] > threshold else 0,
	])

	# Get the lines to be drawn for this cell based on state
	lines = {
		1: [
			[(-1, 0), (0, 1)],
		],
		2: [
			[(0, 1), (1, 0)],
		],
		3: [
			[(-1, 0), (1, 0)],
		],
		4: [
			[(0, -1), (1, 0)],
		],
		5: [
			[(-1, 0), (0, -1)],
			[(0, 1), (1, 0)],
		],
		6: [
			[(0, -1), (0, 1)],
		],
		7: [
			[(-1, 0), (0, -1)],
		],
		8: [
			[(-1, 0), (0, -1)],
		],
		9: [
			[(0, -1), (0, 1)],
		],
		10: [
			[(-1, 0), (0, 1)],
			[(0, -1), (1, 0)],
		],
		11: [
			[(0, -1), (1, 0)],
		],
		12: [
			[(-1, 0), (1, 0)],
		],
		13: [
			[(0, 1), (1, 0)],
		],
		14: [
			[(-1, 0), (0, 1)],
		],
	}.get(state, [])

	# Get the vertices for the polygon to be filled based on state
	polygon = {
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

	# Take the average value of this cell's corners to weight the fill color
	weight = (cell[0] + cell[1] + cell[2] + cell[3]) / 4

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
		# pygame.draw.line(screen, (255, 255, 255), start, end, 4)

	# Fill in the solid area of this cell
	points = [[
		int(view_x + position[0] + point[0]*x_offset),
		int(view_y + position[1] + point[1]*y_offset)
	] for point in polygon]
	if len(points) > 0:
		if weight > 0:
			# Determine color amplitude based on the weight
			color = [ int(1.8*color[0]*weight), int(1.8*color[1]*weight), int(1.8*color[2]*weight) ]
			r,g,b = color
			if r > 255:
				color[0] = 255
			if g > 255:
				color[1] = 255
			if b > 255:
				color[2] = 255
			pygame.draw.polygon(screen, color, points)


# Initialization
pygame.init()
width = 500
height = 500
screen = pygame.display.set_mode([width, height])

# Options
size_x = 90
size_y = 90
size_z = 90
threshold = 0

# Generate 3D Perlin Noise
fields = generate_perlin_noise_3d((size_x, size_y, size_z), (2, 2, 2))

view_x = 0
view_y = 0
colors = list(Color("red").range_to(Color("blue"), size_z))
clock = pygame.time.Clock()
running = True
counter = 0
index = 0
color_index = 0
player_x = 20
player_y = 20
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	# Limit fps
	clock.tick(60)

	# Draw a black background
	screen.fill((0, 0, 0))

	"""
	# Deform terrain on mouse over
	mouse_x, mouse_y = pygame.mouse.get_pos()
	grid_x = int(mouse_x / (width / size_x))
	grid_y = int(mouse_y / (height / size_y))
	if 0 <= grid_x < size_x and 0 <= grid_y < size_y:
		for y in range(grid_y - 10, grid_y + 10):
			for x in range(grid_x - 10, grid_x + 10):
				if x < size_x and y < size_y:
					fields[index][y][x] = 0

	# Move player
	if pygame.key.get_pressed()[pygame.K_RIGHT]:
		player_x += 1
	if pygame.key.get_pressed()[pygame.K_LEFT]:
		player_x -= 1
	if pygame.key.get_pressed()[pygame.K_UP]:
		player_y -= 1
	if pygame.key.get_pressed()[pygame.K_DOWN]:
		player_y += 1

	# Deform terrain under player
	if 0 <= player_x < size_x and 0 <= player_y < size_y:
		for y in range(player_y, player_y + 5):
			for x in range(player_x, player_x + 5):
				if x < size_x and y < size_y:
					fields[index][y][x] = 0
	"""

	# Draw each cell in the current z index
	for y,row in enumerate(fields[index]):
		color = colors[color_index].rgb
		color_index += 1
		if color_index >= size_z:
			color_index = 0
		if y >= size_y - 1:
			continue
		for x,cell in enumerate(row):
			if x >= size_x - 1:
				continue
			draw_cell(
				cell=[
					row[x],
					row[x+1],
					fields[index][y+1][x+1],
					fields[index][y+1][x]
				],
				position=(int(width/size_x/2 + x*width/size_x), int(height/size_y/2 + y*height/size_y)),
				color=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)),
			)

	# Draw the player
	# pygame.draw.rect(screen, (255, 255, 255), (player_x*(width/size_x), player_y*(height/size_y), 4*width/size_x, 4*height/size_y))
	
	# Cycle through the z indices
	index += 1
	if index >= len(fields):
		index = 0

	pygame.display.flip()
