import taichi as ti
import math
import numpy as np
import os
from PIL import Image

ti.init(arch=ti.vulkan)
vec = ti.math.vec2

source = Image.open('source.png')
main_tex = ti.Texture(ti.u8, 4, arr_shape=source.size)
main_tex.from_image(source)

window_size = 1024  # Number of pixels of the window
max_n = 1024 * 16  # Maximum number of grains

density = 100.0
stiffness = 8
restitution_coef = 0.001
gravity = -9.81
dt = 0.0001  # Larger dt might lead to unstable results.
substeps = 60


def arg_ndarray(name: str, dtype, elem_shape):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                             name,
                             dtype,
                             field_dim=1,
                             element_shape=elem_shape)

_p = arg_ndarray("p", ti.f32, (2, )) # Position
_m = arg_ndarray("m", ti.f32, ()) # Mass
_r = arg_ndarray("r", ti.f32, ()) # Radius
_v = arg_ndarray("v", ti.f32, (2, )) # Velocity
_a = arg_ndarray("a", ti.f32, (2, )) # Acceleration
_f = arg_ndarray("f", ti.f32, (2, )) # Force
_c = arg_ndarray("c", ti.u32, ()) # Color
_num_grains = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                             "num_grains",
                             ti.i32,
                             shape=(1,),
                             element_shape=())
_img = ti.graph.Arg(ti.graph.ArgKind.TEXTURE,
                    "img",
                    channel_format=ti.u8,
                    shape=source.size,
                    num_channels=4)

grid_n = 64
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

grain_r_min = 0.003
grain_r_max = 0.004

assert grain_r_max * 2 < grid_size

collider_radius = 0.05
num_colliders = 9
colliders = ti.Vector.field(2, dtype=ti.f32, shape=num_colliders)

for i in range(4):
    colliders[i] = [0.2 + 0.2 * i, 0.42]

for i in range(4, 9):
    colliders[i] = [0.1 + 0.2 * (i - 4), 0.27]


@ti.kernel
def init(
    F_num_grains: ti.types.ndarray(),
    F_p: ti.types.ndarray(field_dim=1),
    F_r: ti.types.ndarray(field_dim=1),
    F_m: ti.types.ndarray(field_dim=1),
    F_c: ti.types.ndarray(field_dim=1),
    F_img: ti.types.texture(num_dimensions=2)):

    sample_res = 128
    for x in range(sample_res):
        for y in range(sample_res // 2, sample_res):
            # Spread grains in a restricted area.
            p, q = x * 8 + 4, y * 8 + 4
            uv = vec(p / 1024, q / 1024)
            i = ti.atomic_add(F_num_grains[None], 1)
            F_p[i] = uv
            F_r[i] = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
            F_m[i] = density * math.pi * F_r[i]**2
            sample = F_img.sample_lod(1.0 - uv, 0.0)
            r = ti.cast(sample.r * 255.0, ti.u32)
            g = ti.cast(sample.g * 255.0, ti.u32)
            b = ti.cast(sample.b * 255.0, ti.u32)
            F_c[i] = b * 65536 + g * 256 + r
            #gf[i].c = 256 * ti.cast(255.0 * uv.y, ti.u32) + ti.cast(255.0 * uv.x, ti.u32)
    #print(F_num_grains[None])


@ti.kernel
def update(
    F_f: ti.types.ndarray(field_dim=1),
    F_m: ti.types.ndarray(field_dim=1),
    F_v: ti.types.ndarray(field_dim=1),
    F_p: ti.types.ndarray(field_dim=1),
    F_a: ti.types.ndarray(field_dim=1)):

    for i in F_f:
        a = F_f[i] / F_m[i]
        F_v[i] += (F_a[i] + a) * dt / 2.0
        F_p[i] += F_v[i] * dt + 0.5 * a * dt**2
        F_a[i] = a


@ti.kernel
def apply_bc(
    F_p: ti.types.ndarray(field_dim=1),
    F_r: ti.types.ndarray(field_dim=1),
    F_v: ti.types.ndarray(field_dim=1)):

    bounce_coef = 0.3  # Velocity damping
    for i in F_p:
        x = F_p[i][0]
        y = F_p[i][1]

        if y - F_r[i] < 0:
            F_p[i][1] = F_r[i]
            F_v[i][1] *= -bounce_coef

        elif y + F_r[i] > 1.0:
            F_p[i][1] = 1.0 - F_r[i]
            F_v[i][1] *= -bounce_coef

        if x - F_r[i] < 0:
            F_p[i][0] = F_r[i]
            F_v[i][0] *= -bounce_coef

        elif x + F_r[i] > 1.0:
            F_p[i][0] = 1.0 - F_r[i]
            F_v[i][0] *= -bounce_coef

        for j in range(num_colliders):
            delta = (F_p[i] - colliders[j]).norm() - (F_r[i] +
                                                       collider_radius)
            if delta < 0:
                normal = (F_p[i] - colliders[j]).normalized()
                F_v[i] -= normal * min(normal.dot(F_v[i]), 0)
                F_p[i] -= delta * normal


@ti.func
def resolve(i, j,
    F_p: ti.types.ndarray(field_dim=1),
    F_r: ti.types.ndarray(field_dim=1),
    F_v: ti.types.ndarray(field_dim=1),
    F_m: ti.types.ndarray(field_dim=1),
    F_f: ti.types.ndarray(field_dim=1),
):
    rel_pos = F_p[j] - F_p[i]
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    delta = -dist + F_r[i] + F_r[j]  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness * 1000
        # Damping force
        M = (F_m[i] * F_m[j]) / (F_m[i] + F_m[j])
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (F_v[j] - F_v[i]) * normal
        f2 = C * V * normal
        F_f[i] += f2 - f1
        F_f[j] -= f2 - f1


list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=max_n, name="particle_id")


@ti.kernel
def contact(
    F_num_grains: ti.types.ndarray(),
    F_f: ti.types.ndarray(field_dim=1),
    F_p: ti.types.ndarray(field_dim=1),
    F_r: ti.types.ndarray(field_dim=1),
    F_v: ti.types.ndarray(field_dim=1),
    F_m: ti.types.ndarray(field_dim=1),
):
    '''
    Handle the collision between grains.
    '''
    for i in range(F_num_grains[None]):
        F_f[i] = vec(0., gravity * F_m[i])  # Apply gravity.

    grain_count.fill(0)

    for i in range(F_num_grains[None]):
        grid_idx = ti.floor(F_p[i] * grid_n, int)
        grain_count[grid_idx] += 1

    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += grain_count[i, j]
        column_sum[i] = sum

    prefix_sum[0, 0] = 0

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                prefix_sum[i, j] += grain_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]

            linear_idx = i * grid_n + j

            list_head[linear_idx] = prefix_sum[i, j] - grain_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    for i in range(F_num_grains[None]):
        grid_idx = ti.floor(F_p[i] * grid_n, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

    # Brute-force collision detection
    '''
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    # Fast collision detection
    for i in range(F_num_grains[None]):
        grid_idx = ti.floor(F_p[i] * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                   list_tail[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if i < j:
                        resolve(i, j, F_p, F_r, F_v, F_m, F_f)


gb_init = ti.graph.GraphBuilder()
gb_init.dispatch(init, _num_grains, _p, _r, _m, _c, _img)
g_init = gb_init.compile()


gb_step = ti.graph.GraphBuilder()
gb_step.dispatch(update, _f, _m, _v, _p, _a)
gb_step.dispatch(apply_bc, _p, _r, _v)
gb_step.dispatch(contact, _num_grains, _f, _p, _r, _v, _m)
g_step = gb_step.compile()


D_p = ti.Vector.ndarray(2, ti.f32, (max_n,)) # Position
D_m = ti.ndarray(ti.f32, (max_n,)) # Mass
D_r = ti.ndarray(ti.f32, (max_n,)) # Radius
D_v = ti.Vector.ndarray(2, ti.f32, (max_n,)) # Velocity
D_a = ti.Vector.ndarray(2, ti.f32, (max_n,)) # Acceleration
D_f = ti.Vector.ndarray(2, ti.f32, (max_n,)) # Force
D_c = ti.ndarray(ti.u32, (max_n,)) # Color
D_num_grains = ti.ndarray(ti.i32, ())

gui = ti.GUI('Taichi DEM', (window_size, window_size),
             background_color=0x000022)
step = 0



module = ti.aot.Module(ti.vulkan)
module.add_graph("g_init", g_init)
module.add_graph("g_step", g_step)
module.save("aot/dem.cgraph", "")


g_init.run({
    "p": D_p,
    "m": D_m,
    "r": D_r,
    "c": D_c,
    "num_grains": D_num_grains,
    "img": main_tex,
})
while gui.running:
    for s in range(substeps):
        g_step.run({
            "p": D_p,
            "m": D_m,
            "r": D_r,
            "v": D_v,
            "a": D_a,
            "f": D_f,
            "num_grains": D_num_grains,
        })
    pos = D_p.to_numpy()
    r = D_r.to_numpy() * window_size
    gui.circles(pos, radius=r, color=D_c.to_numpy())
    gui.circles(colliders.to_numpy(),
                radius=collider_radius * window_size,
                color=0xffff55)
    gui.show()
    step += 1
