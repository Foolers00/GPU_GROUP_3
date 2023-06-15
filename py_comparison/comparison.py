import csv
import turtle as turtle
import tkinter as tkinter




canvwidth=10000
canvheight=10000


def reset_pos():
    t_point = transform_coord(Point(0, 0))
    turtle.setpos(t_point.x, t_point.y)


def move_turtle(p):
    t_point = transform_coord(p)
    turtle.setpos(t_point.x, t_point.y)


def transform_coord(p):
    return Point(p.x-(canvwidth/2)+100, p.y-(canvheight/2)+100)

def draw_point(p):
    move_turtle(p)
    p_str = str(p.x) + "/" + str(p.y)
    turtle.dot(10, "blue")
    #turtle.write(p_str, True)
    reset_pos()

def draw_line(p1, p2, color):
    turtle.color(color)
    move_turtle(p1)
    turtle.pendown()
    move_turtle(p2)
    turtle.penup()
    reset_pos()


def read_csv_points(filename):
    points = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extracting coordinates from CSV row
            x, y = map(float, row)
            p = Point(x, y)
            points.append(p)
    return points

def read_csv_lines(filename):
    lines = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extracting coordinates from CSV row
            x1, y1, x2, y2 = map(float, row)
            p1 = Point(x1, y1)
            p2 = Point(x2, y2)
            line = (p1, p2)
            lines.append(line)
    return lines

class Point:
    x = 0
    y = 0
    def __init__(self, X, Y):   
        self.x = X
        self.y = Y


turtle.tracer(0, 0)
#turtle.speed(1000)
turtle.penup()
turtle.screensize(canvwidth, canvheight, bg="white")


# Example usage
# filename = 'points.csv'  # Replace with the path to your CSV file
# points = read_csv_points(filename)


filename = 'cpu_hull.csv'  # Replace with the path to your CSV file
lines_cpu = read_csv_lines(filename)

filename = 'gpu_hull.csv'  # Replace with the path to your CSV file
lines_gpu = read_csv_lines(filename)

# Printing the stored lines
# for p in points:
#     draw_point(p)
#     print(f"Point: ({p.x}, {p.y})")


# Printing the stored lines cpu
for line in lines_cpu:
    p1, p2 = line
    draw_line(p1, p2, "red")
    print(f"Line segment: ({p1.x}, {p1.y}) to ({p2.x}, {p2.y})")
    

# Printing the stored lines gpu
for line in lines_gpu:
    p1, p2 = line
    draw_line(p1, p2, "green")
    print(f"Line segment: ({p1.x}, {p1.y}) to ({p2.x}, {p2.y})")

turtle.update()

ts = turtle.getscreen()
ts.getcanvas().postscript(file="duck.eps")

turtle.done()


