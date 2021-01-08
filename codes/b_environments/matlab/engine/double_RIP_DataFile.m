% Simscape(TM) Multibody(TM) version: 7.2

% This is a model data file derived from a Simscape Multibody Import XML file using the smimport function.
% The data in this file sets the block parameter values in an imported Simscape Multibody model.
% For more information on this file, see the smimport function help page in the Simscape Multibody documentation.
% You can modify numerical values, but avoid any other changes to this file.
% Do not add code to this file. Do not edit the physical units shown in comments.

%%%VariableName:smiData


%============= RigidTransform =============%

%Initialize the RigidTransform structure array by filling in null values.
smiData.RigidTransform(7).translation = [0.0 0.0 0.0];
smiData.RigidTransform(7).angle = 0.0;
smiData.RigidTransform(7).axis = [0.0 0.0 0.0];
smiData.RigidTransform(7).ID = '';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(1).translation = [0 152.5 -2.5000000000000022];  % mm
smiData.RigidTransform(1).angle = 0;  % rad
smiData.RigidTransform(1).axis = [0 0 0];
smiData.RigidTransform(1).ID = 'B[Link2-1:-:Link1-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(2).translation = [0 2.5 2.5000000000000284];  % mm
smiData.RigidTransform(2).angle = 0;  % rad
smiData.RigidTransform(2).axis = [0 0 0];
smiData.RigidTransform(2).ID = 'F[Link2-1:-:Link1-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(3).translation = [0 0 140];  % mm
smiData.RigidTransform(3).angle = 0;  % rad
smiData.RigidTransform(3).axis = [0 0 0];
smiData.RigidTransform(3).ID = 'B[Arm-1:-:Link1-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(4).translation = [0 162.5 -2.5];  % mm
smiData.RigidTransform(4).angle = 0;  % rad
smiData.RigidTransform(4).axis = [0 0 0];
smiData.RigidTransform(4).ID = 'F[Arm-1:-:Link1-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(5).translation = [0 10 0];  % mm
smiData.RigidTransform(5).angle = 2.0943951023931953;  % rad
smiData.RigidTransform(5).axis = [-0.57735026918962584 -0.57735026918962584 -0.57735026918962584];
smiData.RigidTransform(5).ID = 'B[base-1:-:Arm-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(6).translation = [0 -2.5 2.5000000000000067];  % mm
smiData.RigidTransform(6).angle = 2.0943951023931953;  % rad
smiData.RigidTransform(6).axis = [-0.57735026918962584 -0.57735026918962584 -0.57735026918962584];
smiData.RigidTransform(6).ID = 'F[base-1:-:Arm-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(7).translation = [0 0 0];  % mm
smiData.RigidTransform(7).angle = 0;  % rad
smiData.RigidTransform(7).axis = [0 0 0];
smiData.RigidTransform(7).ID = 'RootGround[base-1]';


%============= Solid =============%
%Center of Mass (CoM) %Moments of Inertia (MoI) %Product of Inertia (PoI)

%Initialize the Solid structure array by filling in null values.
smiData.Solid(4).mass = 0.0;
smiData.Solid(4).CoM = [0.0 0.0 0.0];
smiData.Solid(4).MoI = [0.0 0.0 0.0];
smiData.Solid(4).PoI = [0.0 0.0 0.0];
smiData.Solid(4).color = [0.0 0.0 0.0];
smiData.Solid(4).opacity = 0.0;
smiData.Solid(4).ID = '';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(1).mass = 0.19527625884395622;  % kg
smiData.Solid(1).CoM = [-1.1491130215647148e-05 82.5 0];  % mm
smiData.Solid(1).MoI = [447.01934497060626 0.61434788698423759 447.01837147137115];  % kg*mm^2
smiData.Solid(1).PoI = [0.14716029548827675 2.2344366095490191e-05 0];  % kg*mm^2
smiData.Solid(1).color = [0.69803921568627447 0.69803921568627447 0.69803921568627447];
smiData.Solid(1).opacity = 1;
smiData.Solid(1).ID = 'Link1*:*기본';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(2).mass = 0.330757442030016;  % kg
smiData.Solid(2).CoM = [-1.3033496877605982e-05 -0.0055608320478861368 69.818325698943823];  % mm
smiData.Solid(2).MoI = [543.34854422308115 543.34758095177949 1.0377163564464935];  % kg*mm^2
smiData.Solid(2).PoI = [-0.12383234821369314 2.7005163418298363e-05 -3.6033555593217705e-05];  % kg*mm^2
smiData.Solid(2).color = [0.69803921568627447 0.69803921568627447 0.69803921568627447];
smiData.Solid(2).opacity = 1;
smiData.Solid(2).ID = 'Arm*:*기본';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(3).mass = 0.045762544922862664;  % kg
smiData.Solid(3).CoM = [1.5649465996980077e-06 77.682372964619205 -0.0050239955700061316];  % mm
smiData.Solid(3).MoI = [92.093993638111655 0.14352049507821019 92.093873105781327];  % kg*mm^2
smiData.Solid(3).PoI = [0.017203417609149532 2.4859492862555878e-07 -4.0228268904054465e-06];  % kg*mm^2
smiData.Solid(3).color = [0.69803921568627447 0.69803921568627447 0.69803921568627447];
smiData.Solid(3).opacity = 1;
smiData.Solid(3).ID = 'Link2*:*기본';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(4).mass = 7.8539816339744796e-13;  % kg
smiData.Solid(4).CoM = [0 5 0];  % mm
smiData.Solid(4).MoI = [0 0 0];  % kg*mm^2
smiData.Solid(4).PoI = [0 0 0];  % kg*mm^2
smiData.Solid(4).color = [0.52549019607843139 0.52549019607843139 0.52549019607843139];
smiData.Solid(4).opacity = 1;
smiData.Solid(4).ID = 'base*:*기본';


%============= Joint =============%
%X Revolute Primitive (Rx) %Y Revolute Primitive (Ry) %Z Revolute Primitive (Rz)
%X Prismatic Primitive (Px) %Y Prismatic Primitive (Py) %Z Prismatic Primitive (Pz) %Spherical Primitive (S)
%Constant Velocity Primitive (CV) %Lead Screw Primitive (LS)
%Position Target (Pos)

%Initialize the RevoluteJoint structure array by filling in null values.
smiData.RevoluteJoint(3).Rz.Pos = 0.0;
smiData.RevoluteJoint(3).ID = '';

smiData.RevoluteJoint(1).Rz.Pos = 0;  % deg
smiData.RevoluteJoint(1).ID = '[Link2-1:-:Link1-1]';

smiData.RevoluteJoint(2).Rz.Pos = 0;  % deg
smiData.RevoluteJoint(2).ID = '[Arm-1:-:Link1-1]';

smiData.RevoluteJoint(3).Rz.Pos = 0;  % deg
smiData.RevoluteJoint(3).ID = '[base-1:-:Arm-1]';

