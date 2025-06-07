import torch 
import ttnn

import math
import time
import sys

def get_shade(v):
    """Map [0,1] value to an ASCII character."""
    if v > 0.8: return '#'
    if v > 0.6: return '*'
    if v > 0.4: return '+'
    if v > 0.2: return '.'
    return ' '

def render_frame(mat):
    """Clears the terminal and prints the ASCII frame."""
    sys.stdout.write('\x1b[2J\x1b[H')  # clear & home
    for row in mat:
        sys.stdout.write(''.join(get_shade(v) for v in row) + '\n')
    sys.stdout.flush()


def convert_torch_to_ttnn_tensor(
    torch_tensors: tuple,
    device,
    tt_dtype,
    layout,
    mem_config,
):
    return [
        ttnn.to_device(
            ttnn.from_torch(
                tensor,
                layout=layout,
                dtype=tt_dtype,
                memory_config=mem_config,
                device=device,
            ),
            device,
        )
        for tensor in torch_tensors
    ]

# Configuration
width, height = 64, 64
aspect = width / height
radius = 0.1
fps = 30.0
scale = 10.0
coef =  -10


shape = (width,height)
torch_dtype = torch.float16
ttnn_dtype = ttnn.bfloat16


xs = [
    [ (i/width*2.0 - 1.0)*aspect for i in range(width) ]
    for _ in range(height)
]
ys = [
    [ (j/height*2.0 - 1.0)        for _ in range(width) ]
    for j in range(height)
]

def sdf_circle(x, y, cx, cy, r):
    """Signed distance from (x,y) to circle centered at (cx,cy)."""
    return math.hypot(x - cx, y - cy) - r

def generate_frame(t):
    # animated center
    cx = 0.5 * math.sin(t)
    cy = 0.5 * math.cos(t)

    # 1) raw SDF in one pass over [height][width]
    mat = [
        [ sdf_circle(xs[y][x], ys[y][x], cx, cy, radius)
          for x in range(width) ]
        for y in range(height)
    ]

    # 2) global post-process
    return [
        [ math.exp(-abs(d)*scale) for d in row ]
        for row in mat
    ]


tens_radius = torch.full(shape, radius)
tens_coef = torch.full(shape, coef)
# xs: from –aspect to +aspect, width elements
xxs = torch.tensor(xs, dtype=torch.float16) #torch.linspace(-aspect, aspect, steps=width, dtype=torch_dtype)
# ys: from –1 to +1, height elements
yys = torch.tensor(ys, dtype=torch.float16) #torch.linspace(-1.0, 1.0, steps=height, dtype=torch_dtype)


def generate_ttnn_frame(t,device):
    ntens_radius,ntens_coef,nxxs,nyys = convert_torch_to_ttnn_tensor((tens_radius,tens_coef,xxs,yys),device, ttnn.bfloat16 ,ttnn.TILE_LAYOUT, mem_config=None )

    # loop over t;
    cx = torch.full(shape,  0.5 * math.sin(t))
    cy = torch.full(shape,  0.5 * math.cos(t))

    ncx,ncy = convert_torch_to_ttnn_tensor((cx,cy),device, ttnn.bfloat16 ,ttnn.TILE_LAYOUT, mem_config=None )

    x1 = ttnn.sub(nxxs,ncx)
    x2 = ttnn.pow(x1,2)

    y1 = ttnn.sub(nyys,ncy)
    y2 = ttnn.pow(y1,2)

    z1 = ttnn.add(x2,y2)
    z1 = ttnn.sqrt(z1)
    z1 = ttnn.sub(z1,ntens_radius)

    z1 = ttnn.abs(z1)
    z2 = ttnn.mul(z1,ntens_coef)
    z2 = ttnn.exp(z2)

    t += 0.05
    tensor = ttnn.from_device(z2)
    tensor = ttnn.to_torch(tensor)
    return tensor.tolist()




def tt_main():
    device = ttnn.open_device(device_id=0)

    t = 0.0
    delay = 1.0 / fps
    while True:
        frame = generate_ttnn_frame(t,device)
        render_frame(frame)
        t += 0.05
        time.sleep(delay)

    ttnn.close_device(device)

def main():
    t = 0.0
    delay = 1.0 / fps
    while True:
        frame = generate_frame(t)
        render_frame(frame)
        t += 0.05
        time.sleep(delay)



if __name__ == '__main__':
    tt_main()
    #main()

