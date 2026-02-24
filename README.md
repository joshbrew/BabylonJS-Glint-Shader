## BabylonJS Glint Shader Implementation + Opalescence

Live demo: https://codepen.io/joshbrew/pen/QwKLpyR

Video demo: https://www.youtube.com/watch?v=R8Ca9FvAY0Q

Adapted implementation of: https://www.shadertoy.com/view/tcdGDl (featured in two minute papers)

It's meant for sparkly snow but you can do a lot with it. Customizable colors and scaling. This specifically uses BabylonJS's WebGL2-styled WebGPU shaders which have generic UV access which underpins the shader.

I added a voronoi tiling based opalescence shader on the same gaussian integration foundations, and the results are pretty damn interesting, and it opens a lot of doors. We had to fix a lot of things from the shadertoy sample to make it more customizable. The results speak for themselves. 

<img width="400" alt="Screenshot 2026-02-23 015503" src="https://github.com/user-attachments/assets/ba53a4f8-67ba-4d25-a17d-b257172bff95" />
<img width="400" alt="Screenshot 2026-02-23 104054" src="https://github.com/user-attachments/assets/07a4984f-0354-4017-84fa-1522d52f4f35" />
<img width="400" alt="Screenshot 2026-02-22 212550" src="https://github.com/user-attachments/assets/01c21c98-b9f7-4c5a-af29-0ae0a497ead7" />
<img width="400" alt="Screenshot 2026-02-22 213753" src="https://github.com/user-attachments/assets/33fb761a-0fa4-498e-bfbd-ff1054d0bf75" />
<img width="400" alt="Screenshot 2026-02-23 015122" src="https://github.com/user-attachments/assets/957392f6-5152-42a6-a851-1aaa664b2525" />
