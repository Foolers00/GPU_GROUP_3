import ctypes as ctypes
import turtle as turtle

canvwidth=500
canvheight=500



def reset_pos():
    turtle.penup()
    t_point = transform_coord(0, 0)
    turtle.setpos(t_point.x, t_point.y)
    turtle.pendown()


def move_turtle(point):
    t_point = transform_coord(point)
    turtle.penup()
    turtle.setpos(t_point.x, t_point.y)
    turtle.pendown()


def transform_coord(point):
    return Point(point.x-(canvwidth/2), point.y-(canvheight/2))

def draw_point(point):
    move_turtle(point)
    turtle.dot(10, "blue")


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double),
                ("y", ctypes.c_double)]

class Point_array(ctypes.Structure):
    _fields_ = [("array", ctypes.POINTER(Point)),
                ("curr_size", ctypes.c_size_t),
                ("max_size", ctypes.c_size_t),
                ("index", ctypes.c_int)]


turtle.speed(0)
turtle.screensize(canvwidth, canvheight, bg="white")

so_file = "./libtest.so"
my_functions = ctypes.CDLL(so_file)

my_functions.test_sequence_1.restype = ctypes.POINTER(Point_array)
points = my_functions.test_sequence_1()


for i in range (points.contents.curr_size):
    draw_point(points.contents.array[i])


turtle.done()