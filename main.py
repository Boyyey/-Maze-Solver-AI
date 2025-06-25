import heapq
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Maze representation: 0 = path, 1 = wall
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]

start = (0, 0)  # (row, col)
goal = (4, 4)

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze[neighbor[0]][neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
    return None

def draw_maze(maze, path=None, visited=None):
    maze_np = np.array(maze)
    fig, ax = plt.subplots()
    ax.imshow(maze_np, cmap='binary')
    if visited:
        vx, vy = zip(*visited)
        ax.plot(vy, vx, 'y.', alpha=0.3, label='Visited')
    if path:
        px, py = zip(*path)
        ax.plot(py, px, 'ro-', label='Path')
    ax.scatter(start[1], start[0], c='green', s=100, label='Start')
    ax.scatter(goal[1], goal[0], c='blue', s=100, label='Goal')
    ax.legend()
    ax.set_title('Maze and Fastest Path (A*)')
    ax.axis('off')
    plt.show()

def astar_visual(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()
    frames = []
    while open_set:
        _, current = heapq.heappop(open_set)
        visited.add(current)
        # For animation: store a copy of visited at each step
        frames.append(list(visited))
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, frames
        for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze[neighbor[0]][neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
    return None, frames

def animate_maze(maze, frames, path=None):
    import matplotlib.animation as animation
    maze_np = np.array(maze)
    fig, ax = plt.subplots()
    ax.imshow(maze_np, cmap='binary')
    scat_visited = ax.plot([], [], 'y.', alpha=0.3, label='Visited')[0]
    scat_path = ax.plot([], [], 'ro-', label='Path')[0]
    ax.scatter(start[1], start[0], c='green', s=100, label='Start')
    ax.scatter(goal[1], goal[0], c='blue', s=100, label='Goal')
    ax.legend()
    ax.set_title('Maze Solving Animation (A*)')
    ax.axis('off')
    def update(i):
        if i < len(frames):
            vx, vy = zip(*frames[i]) if frames[i] else ([],[])
            scat_visited.set_data(vy, vx)
        if path and i == len(frames)-1:
            px, py = zip(*path)
            scat_path.set_data(py, px)
        return scat_visited, scat_path
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True, repeat=False)
    plt.show()

def get_user_maze():
    print("Enter your maze row by row, using 0 for open and 1 for wall (comma separated, e.g. 0,1,0,0):")
    while True:
        try:
            rows = int(input("Number of rows: "))
            cols = int(input("Number of columns: "))
            break
        except ValueError:
            print("Please enter a valid integer for rows and columns.")
    maze = []
    for i in range(rows):
        while True:
            row = input(f"Row {i+1}: ").strip()
            try:
                row_vals = [int(x) for x in row.split(",") if x.strip() in ('0','1')]
                if len(row_vals) == cols:
                    maze.append(row_vals)
                    break
                else:
                    print(f"Row must have {cols} values.")
            except Exception:
                print("Invalid input. Please enter 0s and 1s, comma separated.")
    def get_point(name):
        while True:
            val = input(f"Enter {name} as row,col (0-based): ").strip()
            try:
                r, c = [int(x) for x in val.split(",")]
                if 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0:
                    return (r, c)
                else:
                    print("Point must be in bounds and on a walkable cell (0). Try again.")
            except Exception:
                print("Invalid input. Please enter as row,col.")
    start = get_point("start point")
    goal = get_point("goal point")
    return maze, start, goal

def run_gui():
    class MazeGUI:
        def __init__(self, master):
            self.master = master
            master.title("Maze Solver (A*)")
            master.configure(bg="#222831")
            self.rows = 15
            self.cols = 15
            self.cell_size = 32
            self.start = None
            self.goal = None
            self.drawing = 0  # 0: nothing, 1: wall, 2: erase, 3: start, 4: goal
            self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            self.canvas = tk.Canvas(master, width=self.cols*self.cell_size, height=self.rows*self.cell_size, bg="#393e46", highlightthickness=0)
            self.canvas.pack(padx=20, pady=20)
            self.canvas.bind('<Button-1>', self.on_click)
            self.canvas.bind('<B1-Motion>', self.on_drag)
            self.style_btn = {'font':('Segoe UI', 12, 'bold'), 'bg':'#00adb5', 'fg':'#eeeeee', 'activebackground':'#393e46', 'activeforeground':'#00adb5', 'bd':0, 'padx':10, 'pady':5, 'relief':tk.FLAT}
            btn_frame = tk.Frame(master, bg="#222831")
            btn_frame.pack(pady=(0,20))
            tk.Button(btn_frame, text="Draw Wall", command=lambda: self.set_draw_mode(1), **self.style_btn).pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Erase", command=lambda: self.set_draw_mode(2), **self.style_btn).pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Set Start", command=self.set_start_mode, **self.style_btn).pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Set Goal", command=self.set_goal_mode, **self.style_btn).pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Solve", command=self.solve, **self.style_btn).pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Clear", command=self.clear, **self.style_btn).pack(side=tk.LEFT, padx=5)
            self.status = tk.Label(master, text="Draw walls, set start/goal, then click Solve!", font=("Segoe UI", 12), bg="#222831", fg="#eeeeee", pady=10)
            self.status.pack()
            self.draw_grid()
        def set_draw_mode(self, mode):
            self.drawing = mode
            self.status.config(text="Drawing walls" if mode==1 else "Erasing walls")
        def set_start_mode(self):
            self.drawing = 3
            self.status.config(text="Click a cell to set the START")
        def set_goal_mode(self):
            self.drawing = 4
            self.status.config(text="Click a cell to set the GOAL")
        def on_click(self, event):
            r, c = event.y // self.cell_size, event.x // self.cell_size
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if self.drawing == 1:
                    self.grid[r][c] = 1
                elif self.drawing == 2:
                    self.grid[r][c] = 0
                elif self.drawing == 3:
                    if self.grid[r][c] == 0:
                        self.start = (r, c)
                        self.status.config(text=f"Start set at {self.start}")
                elif self.drawing == 4:
                    if self.grid[r][c] == 0:
                        self.goal = (r, c)
                        self.status.config(text=f"Goal set at {self.goal}")
                self.draw_grid()
        def on_drag(self, event):
            if self.drawing in (1,2):
                self.on_click(event)
        def draw_grid(self):
            self.canvas.delete("all")
            for r in range(self.rows):
                for c in range(self.cols):
                    x0, y0 = c*self.cell_size, r*self.cell_size
                    x1, y1 = x0+self.cell_size, y0+self.cell_size
                    color = '#eeeeee' if self.grid[r][c]==0 else '#222831'
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='#393e46', width=2)
            if self.start:
                x0, y0 = self.start[1]*self.cell_size, self.start[0]*self.cell_size
                x1, y1 = x0+self.cell_size, y0+self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill='#43e97b', outline='#393e46', width=2)
                self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text='S', fill='#222831', font=('Segoe UI', 14, 'bold'))
            if self.goal:
                x0, y0 = self.goal[1]*self.cell_size, self.goal[0]*self.cell_size
                x1, y1 = x0+self.cell_size, y0+self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill='#3a8dde', outline='#393e46', width=2)
                self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text='G', fill='#eeeeee', font=('Segoe UI', 14, 'bold'))
        def solve(self):
            if not self.start or not self.goal:
                messagebox.showerror("Error", "Set both start and goal!")
                return
            path, frames = astar_visual(self.grid, self.start, self.goal)
            if not path:
                messagebox.showinfo("Result", "No path found.")
                self.status.config(text="No path found.")
                return
            self.animate(frames, path)
        def animate(self, frames, path):
            self.draw_grid()
            for i, visited in enumerate(frames):
                self.draw_grid()
                for r, c in visited:
                    if (r, c) != self.start and (r, c) != self.goal:
                        x0, y0 = c*self.cell_size, r*self.cell_size
                        x1, y1 = x0+self.cell_size, y0+self.cell_size
                        self.canvas.create_rectangle(x0, y0, x1, y1, fill='#ffe066', outline='#393e46', width=2)
                self.master.update()
                self.master.after(10)
            for (r, c) in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    x0, y0 = c*self.cell_size, r*self.cell_size
                    x1, y1 = x0+self.cell_size, y0+self.cell_size
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='#ff1744', outline='#393e46', width=2)
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text='•', fill='#eeeeee', font=('Segoe UI', 14, 'bold'))
                self.master.update()
                self.master.after(30)
            self.status.config(text=f"Solved! Path length: {len(path)}")
        def clear(self):
            self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            self.start = None
            self.goal = None
            self.status.config(text="Draw walls, set start/goal, then click Solve!")
            self.draw_grid()
    root = tk.Tk()
    root.configure(bg="#222831")
    gui = MazeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    print("--- Maze Solver (A*) ---")
    use_gui = input("Would you like to use the GUI? (y/n): ").strip().lower() == 'y'
    if use_gui:
        run_gui()
    else:
        use_custom = input("Would you like to enter your own maze? (y/n): ").strip().lower() == 'y'
        if use_custom:
            maze, start, goal = get_user_maze()
        path, frames = astar_visual(maze, start, goal)
        if path:
            print("Path found:", path)
        else:
            print("No path found.")
        animate_maze(maze, frames, path)
