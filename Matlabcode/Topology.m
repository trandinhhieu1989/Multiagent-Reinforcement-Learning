function out = Topology(R,C, N)
%// Set parameters
%R = 200;   %// radius 200 meters
%C = [100 100]; %// center [x y]
%N = 2;    %// number of points inside circle

%// generate circle boundary
t = linspace(0, 2*pi, 100);
x = R*cos(t) + C(1);
y = R*sin(t) + C(2);

%// generate random points inside it
th = 2*pi*rand(N,1);
r  = R*rand(N,1);

xR = r.*cos(th) + C(1);
yR = r.*sin(th) + C(2);
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