# Ray Marching with the SFPU

A basic Signed Distance Field (SDF) renderer implemented via ray-marching on Tenstorrent’s Special Function Processing Unit (SFPU). This challenge demonstrates how Tenstorrent hardware can accelerate nonlinear math workloads and simple graphics–style rendering, proving it’s not just for transformers and CNNs.

## Overview

Signed Distance Fields (SDFs) represent shapes implicitly by encoding, at each point in space, the shortest distance to the surface. Ray marching steps along a ray by querying the SDF until it “hits” a surface. In this project, you’ll:

1. Define an SDF for a simple shape (circle, sphere, or custom implicit surface).  
2. March rays through a 2D or 3D grid, using the SFPU to evaluate the SDF at each step.  
3. Shade ASCII or image output based on hit distance or normal estimation.

By leveraging the SFPU’s built-in support for nonlinear functions (√, sin, cos, exp, etc.), you can achieve compact, high-performance kernels that run directly on Tenstorrent accelerators.


