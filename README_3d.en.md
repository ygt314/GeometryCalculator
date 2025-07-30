As one of the three prohibitions in high school is encouraged(byd) (bushi [/doge] > <p align = " right "> Editor's note </p>
## Geometry Calculator Ver 2

## Other Languages

* [English (US)](README.en.md)
* [简体中文](README.md)

Take advantage of your PC’s raw horsepower—brute‑force your geometry problems with analytic geometry!

* [User Guide](frontend/src/pages/docs.md)
* [About Geometry Calculator Ver 2](frontend/src/pages/about.md)
# Change # here illustrates the changes made to accommodate 3d mode
The backend `backend/src/` parent directory: see the directory [change_src.md file](backend/src/change\_src.md)
[The new triangle writing]: `trABC` # This is to distinguish the flat writing 
[And add the flat writing]: `spABC` # The actual representation plane normal vector
# About Three-dimensional (3d) Pattern Support

1. The basic usage remains the same, and the flat writing only supports three points (such as `spABC`(plane ABC))
2. Three dimensional cross product(**cross**):The result is a three-dimensional vector (which can be regarded as a plane normal vector)
Such as _AB_×_AC_ writed by `vecAB cross vecAC`，but you have to be careful.:
- Cross product can only be used for vectors in three-dimensional space.
- The result of cross multiplication of two parallel vectors is zero vector, which cannot be used as normal vector.