import ctypes as ctypes
import turtle as turtle

canvwidth=500
canvheight=500



def reset_pos():
    t_point = transform_coord(Point(0, 0))
    turtle.setpos(t_point.x, t_point.y)


def move_turtle(point):
    t_point = transform_coord(point)
    turtle.setpos(t_point.x, t_point.y)


def transform_coord(point):
    return Point(point.x-(canvwidth/2), point.y-(canvheight/2))

def draw_point(point):
    move_turtle(point)
    turtle.dot(10, "blue")
    reset_pos()

def draw_line(line):
    move_turtle(line.p)
    turtle.pendown()
    move_turtle(line.q)
    turtle.penup()
    reset_pos()



class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double),
                ("y", ctypes.c_double)]

class Point_array(ctypes.Structure):
    _fields_ = [("array", ctypes.POINTER(Point)),
                ("curr_size", ctypes.c_size_t),
                ("max_size", ctypes.c_size_t)]
    

class Line(ctypes.Structure):
    _fields_ = [("p", Point),
                ("q", Point)]
    

class Hull(ctypes.Structure):
    _fields_ = [("array", ctypes.POINTER(Line)),
                ("curr_size", ctypes.c_size_t),
                ("max_size", ctypes.c_size_t)]




turtle.speed(1)
turtle.penup()
turtle.screensize(canvwidth, canvheight, bg="white")

so_file = "./libtest.so"
my_functions = ctypes.CDLL(so_file)

my_functions.test_sequence_2.restype = ctypes.POINTER(Point_array)

my_functions.test_sequence_3.restype = ctypes.POINTER(Hull)

my_functions.test_sequence_4_1.restype = ctypes.POINTER(Point_array)
my_functions.test_sequence_4_2.argtypes = [ctypes.POINTER(Point_array)]
my_functions.test_sequence_4_2.restype = ctypes.POINTER(Hull)


### test_sequence_2 ###
# points = my_functions.test_sequence_2()
# for i in range (points.contents.curr_size):
#     draw_point(points.contents.array[i])


### test_sequence_3 ###
# hull = my_functions.test_sequence_3()
# for i in range (hull.contents.curr_size):
#     draw_line(hull.contents.array[i])

### test_sequence_4 ###
points = my_functions.test_sequence_4_1()
for i in range (points.contents.curr_size):
    draw_point(points.contents.array[i])
hull = my_functions.test_sequence_4_2(points)
for i in range (hull.contents.curr_size):
    draw_line(hull.contents.array[i])

turtle.done()