function out = Fixed_Topology(R,C, N)
%// Set parameters
%R = 200;   %// radius 200 meters
%C = [100 100]; %// center [x y]
%N = 2;    %// number of points inside circle

%// generate circle boundary
t = linspace(0, 2*pi, 100);
x = R*cos(t) + C(1);
y = R*sin(t) + C(2);

%// generate fixed coordinate of N users inside the Macro cell
%xR = [250;350;350;450;550];
%yR = [100;200;100;100;100];
xR = [300;400;500];
yR = [100;200;100];
coordinate = horzcat(xR,yR);

%// Plot everything
figure(1), clf, hold on
plot(x,y,'b')
%plot(C(1),C(2),'r.', 'MarkerSize', 100)
plot(C(1),C(2),'r*','MarkerSize', 10)
plot(xR,yR,'go')
axis equal
out = coordinate;
end