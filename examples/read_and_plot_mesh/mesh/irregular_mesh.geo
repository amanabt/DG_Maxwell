// Gmsh project created on Sat Dec  2 17:08:15 2017
Point(1) = {-1, 1, 0, 1.0};
Point(2) = {-1, -1, 0, 1.0};
Point(3) = {-1, 0, 0, 1.0};
Point(4) = {1, -1, 0, 1.0};
Point(5) = {1, 0, 0, 1.0};
Point(6) = {1, 1, 0, 1.0};
Point(7) = {-0, 0.3, 0, 1.0};
Line(1) = {1, 3};
Line(2) = {2, 3};
Line(3) = {2, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Spline(7) = {3, 7, 5};
Line Loop(8) = {6, 1, 7, 5};
Plane Surface(9) = {8};
Line Loop(10) = {2, 7, -4, -3};
Plane Surface(11) = {10};
Transfinite Line {6, 7, 1, 5, 2, 3, 4} = 10 Using Progression 1;
Transfinite Surface {9};
Transfinite Surface {11};
Recombine Surface {9, 11};