% Simscape(TM) Multibody(TM) version: 7.1

% This is a model data file derived from a Simscape Multibody Import XML file using the smimport function.
% The data in this file sets the block parameter values in an imported Simscape Multibody model.
% For more information on this file, see the smimport function help page in the Simscape Multibody documentation.
% You can modify numerical values, but avoid any other changes to this file.
% Do not add code to this file. Do not edit the physical units shown in comments.

%%%VariableName:smiData


%============= RigidTransform =============%

%Initialize the RigidTransform structure array by filling in null values.
smiData.RigidTransform(5).translation = [0.0 0.0 0.0];
smiData.RigidTransform(5).angle = 0.0;
smiData.RigidTransform(5).axis = [0.0 0.0 0.0];
smiData.RigidTransform(5).ID = '';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(1).translation = [0 0 50];  % mm
smiData.RigidTransform(1).angle = 3.1415926535897931;  % rad
smiData.RigidTransform(1).axis = [1 0 0];
smiData.RigidTransform(1).ID = 'B[link1-1:-:link2-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(2).translation = [-6.6613381477509392e-15 -5.0000000000000249 -2.5000000000000284];  % mm
smiData.RigidTransform(2).angle = 3.1415926535897931;  % rad
smiData.RigidTransform(2).axis = [1 0 2.2204460492503131e-16];
smiData.RigidTransform(2).ID = 'F[link1-1:-:link2-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(3).translation = [0 10 0];  % mm
smiData.RigidTransform(3).angle = 2.0943951023931953;  % rad
smiData.RigidTransform(3).axis = [-0.57735026918962584 -0.57735026918962584 -0.57735026918962584];
smiData.RigidTransform(3).ID = 'B[shaft-1:-:link1-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(4).translation = [1.7763568394002505e-15 -2.5000000000000009 2.4999999999999991];  % mm
smiData.RigidTransform(4).angle = 2.0943951023931953;  % rad
smiData.RigidTransform(4).axis = [-0.57735026918962584 -0.57735026918962584 -0.57735026918962584];
smiData.RigidTransform(4).ID = 'F[shaft-1:-:link1-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(5).translation = [-2.6675687500471135 -6.4033731250247135 10];  % mm
smiData.RigidTransform(5).angle = 0;  % rad
smiData.RigidTransform(5).axis = [0 0 0];
smiData.RigidTransform(5).ID = 'RootGround[shaft-1]';


%============= Solid =============%
%Center of Mass (CoM) %Moments of Inertia (MoI) %Product of Inertia (PoI)

%Initialize the Solid structure array by filling in null values.
smiData.Solid(3).mass = 0.0;
smiData.Solid(3).CoM = [0.0 0.0 0.0];
smiData.Solid(3).MoI = [0.0 0.0 0.0];
smiData.Solid(3).PoI = [0.0 0.0 0.0];
smiData.Solid(3).color = [0.0 0.0 0.0];
smiData.Solid(3).opacity = 0.0;
smiData.Solid(3).ID = '';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(1).mass = 0.01624234569796066;  % kg
smiData.Solid(1).CoM = [-1.2668784181790654e-05 -52.329692952302317 -0.0074077968532969204];  % mm
smiData.Solid(1).MoI = [15.025676407688273 0.051025273899613116 15.025613615918486];  % kg*mm^2
smiData.Solid(1).PoI = [0.0056953831544085658 -1.7157639838747039e-06 -1.1106906264045729e-06];  % kg*mm^2
smiData.Solid(1).color = [0.65098039215686276 0.61960784313725492 0.58823529411764708];
smiData.Solid(1).opacity = 1;
smiData.Solid(1).ID = 'link2*:*기본';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(2).mass = 0.0077649542717894574;  % kg
smiData.Solid(2).CoM = [-1.2590884508001884e-05 -0.01549526154824152 24.831254820188903];  % mm
smiData.Solid(2).MoI = [1.6471813205502108 1.647119501865532 0.024532452610473314];  % kg*mm^2
smiData.Solid(2).PoI = [-0.00268721607943463 1.5342656749273739e-07 -8.1426866394655224e-07];  % kg*mm^2
smiData.Solid(2).color = [0.65098039215686276 0.61960784313725492 0.58823529411764708];
smiData.Solid(2).opacity = 1;
smiData.Solid(2).ID = 'link1*:*기본';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(3).mass = 0.0021283208027941506;  % kg
smiData.Solid(3).CoM = [0 5 0];  % mm
smiData.Solid(3).MoI = [0.031038011707414697 0.026604010034926882 0.031038011707414697];  % kg*mm^2
smiData.Solid(3).PoI = [0 0 0];  % kg*mm^2
smiData.Solid(3).color = [0.89803921568627454 0.91764705882352937 0.92941176470588238];
smiData.Solid(3).opacity = 1;
smiData.Solid(3).ID = 'shaft*:*기본';


%============= Joint =============%
%X Revolute Primitive (Rx) %Y Revolute Primitive (Ry) %Z Revolute Primitive (Rz)
%X Prismatic Primitive (Px) %Y Prismatic Primitive (Py) %Z Prismatic Primitive (Pz) %Spherical Primitive (S)
%Constant Velocity Primitive (CV) %Lead Screw Primitive (LS)
%Position Target (Pos)

%Initialize the RevoluteJoint structure array by filling in null values.
smiData.RevoluteJoint(2).Rz.Pos = 0.0;
smiData.RevoluteJoint(2).ID = '';

smiData.RevoluteJoint(1).Rz.Pos = 180;  % deg
smiData.RevoluteJoint(1).ID = '[link1-1:-:link2-1]';

smiData.RevoluteJoint(2).Rz.Pos = 90.000000000000014;  % deg
smiData.RevoluteJoint(2).ID = '[shaft-1:-:link1-1]';

