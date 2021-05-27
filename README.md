# ray-tracer

## Description 
Recursive ray tracer with soft shadows/caustics and global illumination. 

Fun photo of the result: <br>
![rendering result](https://user-images.githubusercontent.com/54962990/118700874-f616db00-b7c7-11eb-9e78-35122d0eca74.PNG)

## Background
Ray tracing simulates the path of light rays as they emerge from a light source, 
bounce around a scene, and eventually into the camera. 

Visualize it!

Instead of simulating all light rays emerging from a light source, which is computationally infeasible, 
we only trace the rays that end up hitting the camera-- that is, we cast 
a ray from each pixel of the camera plane, and on every intersection with an 
object (such as a plane or sphere), we calculate the direction the light ray 
must have came from geometrically. At this intersection, we also collect information 
about the magnitude of light which we compound at every light bounce.
After the tracing function is finished, terminating after a fixed depth, 
this aggregated light information is used to color each pixel the simulated ray originated from. 

To describe ray tracing in more detail we need some terminology:
- World Space - 3D Cartesian coordinate system in which the scene is defined.
- BRDF - Describes how a surface interacts with light at the point of intersection (either reflecting, scattering, or transmitting). 
Matematically, BRDF's are functions that map an input direction to an output direction depending on the physical properties of the surface material.
There are three general BRDF's to describe three general surfaces:
  - Specular - reflect light perfectly, think a laser bouncing off a mirror. 
  - Diffuse - scatters light all about, which are a majority of surfaces. An object often looks the same color viewed from a different perspective.
  - Glossy - in between specular and diffuse. Scatters rays in reflected input direction. Think chrome. 
 
- Direct Illumination - Magnitude of light originating from the light source at the current intersection/bounce.
- Indirect Illumination - Magnitude of light coming from other objects at the current bounce. 

Great! We will finish up this explanation with how the direction of light is really determined and light is really measured and aggregated. 
Starting at camera plane, we send multiple light rays which are representative of the infinite light rays that would be really intersecting this pixel.

The ray tracing equation for each pixel, for each ray is the same (which is why we can parallelize this computation). 

Tracing the ray out in world space, we check if there is an intersection with another object in the scene, also defined in terms of world space.
If an intersection is found, we mark this intersection point and calculate the next direction that this ray will trace according to the BRDF. 
At this intersection, we record the magnitude of light which is a *combination* of direct and indirect illumination, in which indirect illumination
is only known when the recursive function telescopes backward. That is, the combination of direct and indirection illumination at the current bounce
is the indirect illumination of the previous bounce in the ray tracing function. Neato!

All light information from a set of representative light rays are finally averaged to color the originated pixel.

The matematical details of the camera transform, local coordinate projection, and BRDF are left to the enthusiatic reader. 

## Code Walkthrough 
Start at the main function and "trace" the description above!
