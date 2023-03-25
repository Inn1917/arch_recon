import random
def print_cube(cube):
	n=len(cube)
	for i in range(n):
		for j in range(n):
			if cube[i][j]==0:
				print("_",end="")
			else:
				print("*",end="")
		print("\n",end="")
	print("\n")
def sand_fall(cube):
	n=len(cube)
	while cube[0][0]==1:
		cur_pos=[0,0]
		for i in range(2*n-1):
			x,y=cur_pos
			if x<n-1 and y<n-1:
				left_pos=[x+1,y]
				right_pos=[x,y+1]
				if cube[left_pos[0]][left_pos[1]]==1:
					if cube[right_pos[0]][right_pos[1]]==1:
						rand_=random.random()
						if rand_<0.5:
							cur_pos=left_pos
						else:
							cur_pos=right_pos
					else:
						cur_pos=left_pos
				else:
					if cube[right_pos[0]][right_pos[1]]==1:
						cur_pos=right_pos
					else:
						break	
			elif x<n-1:
				left_pos=[x+1,y]
				if cube[left_pos[0]][left_pos[1]]==1:
					cur_pos=left_pos
				else:
					break
			elif y<n-1:
				right_pos=[x,y+1]
				if cube[right_pos[0]][right_pos[1]]==1:
					cur_pos=right_pos
				else:
					break
			else:
				break
		cube[cur_pos[0]][cur_pos[1]]=0
		print_cube(cube)
if __name__=="__main__":
	cube=[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
	sand_fall(cube)
					