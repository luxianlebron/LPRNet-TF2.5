??8
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??-
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
?
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_6/gamma
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_6/beta
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_6/moving_variance
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
?
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
?
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
?
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
?
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
?
container/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@B*!
shared_namecontainer/kernel
}
$container/kernel/Read/ReadVariableOpReadVariableOpcontainer/kernel*&
_output_shapes
:@B*
dtype0
t
container/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:B*
shared_namecontainer/bias
m
"container/bias/Read/ReadVariableOpReadVariableOpcontainer/bias*
_output_shapes
:B*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:  *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
?
!depthwise_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!depthwise_conv2d/depthwise_kernel
?
5depthwise_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!depthwise_conv2d/depthwise_kernel*&
_output_shapes
: *
dtype0
?
depthwise_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namedepthwise_conv2d/bias
{
)depthwise_conv2d/bias/Read/ReadVariableOpReadVariableOpdepthwise_conv2d/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
?
#depthwise_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#depthwise_conv2d_1/depthwise_kernel
?
7depthwise_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_1/depthwise_kernel*&
_output_shapes
: *
dtype0
?
depthwise_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namedepthwise_conv2d_1/bias

+depthwise_conv2d_1/bias/Read/ReadVariableOpReadVariableOpdepthwise_conv2d_1/bias*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
?
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
?
#depthwise_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#depthwise_conv2d_2/depthwise_kernel
?
7depthwise_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_2/depthwise_kernel*&
_output_shapes
: *
dtype0
?
depthwise_conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namedepthwise_conv2d_2/bias

+depthwise_conv2d_2/bias/Read/ReadVariableOpReadVariableOpdepthwise_conv2d_2/bias*
_output_shapes
: *
dtype0
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_4/moving_variance
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
?
#depthwise_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#depthwise_conv2d_3/depthwise_kernel
?
7depthwise_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_3/depthwise_kernel*&
_output_shapes
: *
dtype0
?
depthwise_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namedepthwise_conv2d_3/bias

+depthwise_conv2d_3/bias/Read/ReadVariableOpReadVariableOpdepthwise_conv2d_3/bias*
_output_shapes
: *
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:`@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
?
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
?
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
?
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
?
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0

NoOpNoOp
М
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
regularization_losses
	variables
trainable_variables
	keras_api

signatures

_init_input_shape
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
?
(layer-0
)layer_with_weights-0
)layer-1
*layer_with_weights-1
*layer-2
+layer-3
,layer_with_weights-2
,layer-4
-layer_with_weights-3
-layer-5
.layer-6
/layer_with_weights-4
/layer-7
0layer-8
1layer-9
2layer_with_weights-5
2layer-10
3layer_with_weights-6
3layer-11
4layer-12
5regularization_losses
6	variables
7trainable_variables
8	keras_api
?
9layer-0
:layer_with_weights-0
:layer-1
;layer_with_weights-1
;layer-2
<layer-3
=layer_with_weights-2
=layer-4
>layer_with_weights-3
>layer-5
?layer-6
@layer_with_weights-4
@layer-7
Alayer-8
Blayer-9
Clayer_with_weights-5
Clayer-10
Dlayer_with_weights-6
Dlayer-11
Elayer-12
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
R
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
?
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
R
]regularization_losses
^	variables
_trainable_variables
`	keras_api
R
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
h

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api

k	keras_api

l	keras_api
 
?
0
1
2
3
4
5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
J46
K47
U48
V49
W50
X51
e52
f53
?
0
1
2
3
m4
n5
o6
p7
s8
t9
u10
v11
y12
z13
{14
|15
}16
~17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
J32
K33
U34
V35
e36
f37
?
?layer_metrics
regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
	variables
trainable_variables
 
 
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
?layer_metrics
regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
	variables
?layers
trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
?
?layer_metrics
 regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
!	variables
?layers
"trainable_variables
 
 
 
?
?layer_metrics
$regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
%	variables
?layers
&trainable_variables
 
l

mkernel
nbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis
	ogamma
pbeta
qmoving_mean
rmoving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
v
sdepthwise_kernel
tbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis
	ugamma
vbeta
wmoving_mean
xmoving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
v
ydepthwise_kernel
zbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

{kernel
|bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis
	}gamma
~beta
moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
?
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13
{14
|15
}16
~17
18
?19
f
m0
n1
o2
p3
s4
t5
u6
v7
y8
z9
{10
|11
}12
~13
?
?layer_metrics
5regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
6	variables
7trainable_variables
 
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
x
?depthwise_kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
x
?depthwise_kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
t
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?
?layer_metrics
Fregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
G	variables
Htrainable_variables
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
?
?layer_metrics
Lregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
M	variables
?layers
Ntrainable_variables
 
 
 
?
?layer_metrics
Pregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
Q	variables
?layers
Rtrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1
W2
X3

U0
V1
?
?layer_metrics
Yregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
Z	variables
?layers
[trainable_variables
 
 
 
?
?layer_metrics
]regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
^	variables
?layers
_trainable_variables
 
 
 
?
?layer_metrics
aregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
b	variables
?layers
ctrainable_variables
\Z
VARIABLE_VALUEcontainer/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEcontainer/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
?
?layer_metrics
gregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
h	variables
?layers
itrainable_variables
 
 
IG
VARIABLE_VALUEconv2d/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEbatch_normalization/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#batch_normalization/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!depthwise_conv2d/depthwise_kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdepthwise_conv2d/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_1/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#depthwise_conv2d_1/depthwise_kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdepthwise_conv2d_1/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/24/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/25/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_2/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_2/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_3/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_3/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/30/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/31/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#depthwise_conv2d_2/depthwise_kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdepthwise_conv2d_2/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_4/gamma'variables/34/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_4/beta'variables/35/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_4/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_4/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#depthwise_conv2d_3/depthwise_kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdepthwise_conv2d_3/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_3/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_3/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_5/gamma'variables/42/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_5/beta'variables/43/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_5/moving_mean'variables/44/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_5/moving_variance'variables/45/.ATTRIBUTES/VARIABLE_VALUE
 
}
0
1
q2
r3
w4
x5
6
?7
?8
?9
?10
?11
?12
?13
W14
X15
 
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

m0
n1

m0
n1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 

o0
p1
q2
r3

o0
p1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 

s0
t1

s0
t1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 

u0
v1
w2
x3

u0
v1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 

y0
z1

y0
z1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 

{0
|1

{0
|1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 

}0
~1
2
?3

}0
~1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
+
q0
r1
w2
x3
4
?5
 
 
^
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
 

?0
?1

?0
?1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?0
?1
?2
?3

?0
?1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 

?0
?1

?0
?1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?0
?1
?2
?3

?0
?1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 

?0
?1

?0
?1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 

?0
?1

?0
?1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?0
?1
?2
?3

?0
?1
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
 
0
?0
?1
?2
?3
?4
?5
 
 
^
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
 
 
 
 
 
 
 
 
 
 
 

W0
X1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

q0
r1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

w0
x1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
?
serving_default_inputPlaceholder*/
_output_shapes
:?????????^*
dtype0*$
shape:?????????^
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv2d_4/kernelconv2d_4/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!depthwise_conv2d/depthwise_kerneldepthwise_conv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance#depthwise_conv2d_1/depthwise_kerneldepthwise_conv2d_1/biasconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#depthwise_conv2d_2/depthwise_kerneldepthwise_conv2d_2/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#depthwise_conv2d_3/depthwise_kerneldepthwise_conv2d_3/biasconv2d_3/kernelconv2d_3/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancecontainer/kernelcontainer/bias*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_511313485
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp$container/kernel/Read/ReadVariableOp"container/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5depthwise_conv2d/depthwise_kernel/Read/ReadVariableOp)depthwise_conv2d/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp7depthwise_conv2d_1/depthwise_kernel/Read/ReadVariableOp+depthwise_conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp7depthwise_conv2d_2/depthwise_kernel/Read/ReadVariableOp+depthwise_conv2d_2/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp7depthwise_conv2d_3/depthwise_kernel/Read/ReadVariableOp+depthwise_conv2d_3/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOpConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_save_511316229
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancecontainer/kernelcontainer/biasconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!depthwise_conv2d/depthwise_kerneldepthwise_conv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance#depthwise_conv2d_1/depthwise_kerneldepthwise_conv2d_1/biasconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#depthwise_conv2d_2/depthwise_kerneldepthwise_conv2d_2/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#depthwise_conv2d_3/depthwise_kerneldepthwise_conv2d_3/biasconv2d_3/kernelconv2d_3/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference__traced_restore_511316401??*
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511309600

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?C
?

F__inference_model_1_layer_call_and_return_conditional_losses_511311530
input_2,
conv2d_2_511311477:@  
conv2d_2_511311479: -
batch_normalization_3_511311482: -
batch_normalization_3_511311484: -
batch_normalization_3_511311486: -
batch_normalization_3_511311488: 6
depthwise_conv2d_2_511311492: *
depthwise_conv2d_2_511311494: -
batch_normalization_4_511311497: -
batch_normalization_4_511311499: -
batch_normalization_4_511311501: -
batch_normalization_4_511311503: 6
depthwise_conv2d_3_511311507: *
depthwise_conv2d_3_511311509: ,
conv2d_3_511311514:`@ 
conv2d_3_511311516:@-
batch_normalization_5_511311519:@-
batch_normalization_5_511311521:@-
batch_normalization_5_511311523:@-
batch_normalization_5_511311525:@
identity??-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?*depthwise_conv2d_2/StatefulPartitionedCall?*depthwise_conv2d_3/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_511311477conv2d_2_511311479*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_5113110362"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_511311482batch_normalization_3_511311484batch_normalization_3_511311486batch_normalization_3_511311488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5113106612/
-batch_normalization_3/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_5113110562
re_lu_4/PartitionedCall?
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0depthwise_conv2d_2_511311492depthwise_conv2d_2_511311494*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_5113107342,
*depthwise_conv2d_2/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_4_511311497batch_normalization_4_511311499batch_normalization_4_511311501batch_normalization_4_511311503*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5113108102/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_5113110772
re_lu_5/PartitionedCall?
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0depthwise_conv2d_3_511311507depthwise_conv2d_3_511311509*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_5113108832,
*depthwise_conv2d_3/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_5113110892
re_lu_6/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_concatenate_1_layer_call_and_return_conditional_losses_5113110982
concatenate_1/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_3_511311514conv2d_3_511311516*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_5113111102"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_5_511311519batch_normalization_5_511311521batch_normalization_5_511311523batch_normalization_5_511311525*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5113109592/
-batch_normalization_5/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_5113111302
re_lu_7/PartitionedCall?
IdentityIdentity re_lu_7/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????@
!
_user_specified_name	input_2
??
?:
$__inference__wrapped_model_511309534	
inputI
/model_2_conv2d_4_conv2d_readvariableop_resource: >
0model_2_conv2d_4_biasadd_readvariableop_resource: C
5model_2_batch_normalization_6_readvariableop_resource: E
7model_2_batch_normalization_6_readvariableop_1_resource: T
Fmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: V
Hmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: M
3model_2_model_conv2d_conv2d_readvariableop_resource:  B
4model_2_model_conv2d_biasadd_readvariableop_resource: G
9model_2_model_batch_normalization_readvariableop_resource: I
;model_2_model_batch_normalization_readvariableop_1_resource: X
Jmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_resource: Z
Lmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: Z
@model_2_model_depthwise_conv2d_depthwise_readvariableop_resource: L
>model_2_model_depthwise_conv2d_biasadd_readvariableop_resource: I
;model_2_model_batch_normalization_1_readvariableop_resource: K
=model_2_model_batch_normalization_1_readvariableop_1_resource: Z
Lmodel_2_model_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: \
Nmodel_2_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: \
Bmodel_2_model_depthwise_conv2d_1_depthwise_readvariableop_resource: N
@model_2_model_depthwise_conv2d_1_biasadd_readvariableop_resource: O
5model_2_model_conv2d_1_conv2d_readvariableop_resource:@@D
6model_2_model_conv2d_1_biasadd_readvariableop_resource:@I
;model_2_model_batch_normalization_2_readvariableop_resource:@K
=model_2_model_batch_normalization_2_readvariableop_1_resource:@Z
Lmodel_2_model_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@\
Nmodel_2_model_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@Q
7model_2_model_1_conv2d_2_conv2d_readvariableop_resource:@ F
8model_2_model_1_conv2d_2_biasadd_readvariableop_resource: K
=model_2_model_1_batch_normalization_3_readvariableop_resource: M
?model_2_model_1_batch_normalization_3_readvariableop_1_resource: \
Nmodel_2_model_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: ^
Pmodel_2_model_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: ^
Dmodel_2_model_1_depthwise_conv2d_2_depthwise_readvariableop_resource: P
Bmodel_2_model_1_depthwise_conv2d_2_biasadd_readvariableop_resource: K
=model_2_model_1_batch_normalization_4_readvariableop_resource: M
?model_2_model_1_batch_normalization_4_readvariableop_1_resource: \
Nmodel_2_model_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: ^
Pmodel_2_model_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: ^
Dmodel_2_model_1_depthwise_conv2d_3_depthwise_readvariableop_resource: P
Bmodel_2_model_1_depthwise_conv2d_3_biasadd_readvariableop_resource: Q
7model_2_model_1_conv2d_3_conv2d_readvariableop_resource:`@F
8model_2_model_1_conv2d_3_biasadd_readvariableop_resource:@K
=model_2_model_1_batch_normalization_5_readvariableop_resource:@M
?model_2_model_1_batch_normalization_5_readvariableop_1_resource:@\
Nmodel_2_model_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@^
Pmodel_2_model_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@I
/model_2_conv2d_5_conv2d_readvariableop_resource:@@>
0model_2_conv2d_5_biasadd_readvariableop_resource:@C
5model_2_batch_normalization_7_readvariableop_resource:@E
7model_2_batch_normalization_7_readvariableop_1_resource:@T
Fmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@J
0model_2_container_conv2d_readvariableop_resource:@B?
1model_2_container_biasadd_readvariableop_resource:B
identity??=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?,model_2/batch_normalization_6/ReadVariableOp?.model_2/batch_normalization_6/ReadVariableOp_1?=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?,model_2/batch_normalization_7/ReadVariableOp?.model_2/batch_normalization_7/ReadVariableOp_1?(model_2/container/BiasAdd/ReadVariableOp?'model_2/container/Conv2D/ReadVariableOp?'model_2/conv2d_4/BiasAdd/ReadVariableOp?&model_2/conv2d_4/Conv2D/ReadVariableOp?'model_2/conv2d_5/BiasAdd/ReadVariableOp?&model_2/conv2d_5/Conv2D/ReadVariableOp?Amodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp?Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?0model_2/model/batch_normalization/ReadVariableOp?2model_2/model/batch_normalization/ReadVariableOp_1?Cmodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Emodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?2model_2/model/batch_normalization_1/ReadVariableOp?4model_2/model/batch_normalization_1/ReadVariableOp_1?Cmodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Emodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?2model_2/model/batch_normalization_2/ReadVariableOp?4model_2/model/batch_normalization_2/ReadVariableOp_1?+model_2/model/conv2d/BiasAdd/ReadVariableOp?*model_2/model/conv2d/Conv2D/ReadVariableOp?-model_2/model/conv2d_1/BiasAdd/ReadVariableOp?,model_2/model/conv2d_1/Conv2D/ReadVariableOp?5model_2/model/depthwise_conv2d/BiasAdd/ReadVariableOp?7model_2/model/depthwise_conv2d/depthwise/ReadVariableOp?7model_2/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp?9model_2/model/depthwise_conv2d_1/depthwise/ReadVariableOp?Emodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Gmodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?4model_2/model_1/batch_normalization_3/ReadVariableOp?6model_2/model_1/batch_normalization_3/ReadVariableOp_1?Emodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Gmodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?4model_2/model_1/batch_normalization_4/ReadVariableOp?6model_2/model_1/batch_normalization_4/ReadVariableOp_1?Emodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Gmodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?4model_2/model_1/batch_normalization_5/ReadVariableOp?6model_2/model_1/batch_normalization_5/ReadVariableOp_1?/model_2/model_1/conv2d_2/BiasAdd/ReadVariableOp?.model_2/model_1/conv2d_2/Conv2D/ReadVariableOp?/model_2/model_1/conv2d_3/BiasAdd/ReadVariableOp?.model_2/model_1/conv2d_3/Conv2D/ReadVariableOp?9model_2/model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp?;model_2/model_1/depthwise_conv2d_2/depthwise/ReadVariableOp?9model_2/model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp?;model_2/model_1/depthwise_conv2d_3/depthwise/ReadVariableOp?
&model_2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&model_2/conv2d_4/Conv2D/ReadVariableOp?
model_2/conv2d_4/Conv2DConv2Dinput.model_2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingVALID*
strides
2
model_2/conv2d_4/Conv2D?
'model_2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/conv2d_4/BiasAdd/ReadVariableOp?
model_2/conv2d_4/BiasAddBiasAdd model_2/conv2d_4/Conv2D:output:0/model_2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
model_2/conv2d_4/BiasAdd?
,model_2/batch_normalization_6/ReadVariableOpReadVariableOp5model_2_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_2/batch_normalization_6/ReadVariableOp?
.model_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_2/batch_normalization_6/ReadVariableOp_1?
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3!model_2/conv2d_4/BiasAdd:output:04model_2/batch_normalization_6/ReadVariableOp:value:06model_2/batch_normalization_6/ReadVariableOp_1:value:0Emodel_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 20
.model_2/batch_normalization_6/FusedBatchNormV3?
model_2/re_lu_8/ReluRelu2model_2/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
model_2/re_lu_8/Relu?
*model_2/model/conv2d/Conv2D/ReadVariableOpReadVariableOp3model_2_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*model_2/model/conv2d/Conv2D/ReadVariableOp?
model_2/model/conv2d/Conv2DConv2D"model_2/re_lu_8/Relu:activations:02model_2/model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
model_2/model/conv2d/Conv2D?
+model_2/model/conv2d/BiasAdd/ReadVariableOpReadVariableOp4model_2_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_2/model/conv2d/BiasAdd/ReadVariableOp?
model_2/model/conv2d/BiasAddBiasAdd$model_2/model/conv2d/Conv2D:output:03model_2/model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
model_2/model/conv2d/BiasAdd?
0model_2/model/batch_normalization/ReadVariableOpReadVariableOp9model_2_model_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype022
0model_2/model/batch_normalization/ReadVariableOp?
2model_2/model/batch_normalization/ReadVariableOp_1ReadVariableOp;model_2_model_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype024
2model_2/model/batch_normalization/ReadVariableOp_1?
Amodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpJmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Amodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
2model_2/model/batch_normalization/FusedBatchNormV3FusedBatchNormV3%model_2/model/conv2d/BiasAdd:output:08model_2/model/batch_normalization/ReadVariableOp:value:0:model_2/model/batch_normalization/ReadVariableOp_1:value:0Imodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Kmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 24
2model_2/model/batch_normalization/FusedBatchNormV3?
model_2/model/re_lu/ReluRelu6model_2/model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
model_2/model/re_lu/Relu?
7model_2/model/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp@model_2_model_depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype029
7model_2/model/depthwise_conv2d/depthwise/ReadVariableOp?
.model_2/model/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             20
.model_2/model/depthwise_conv2d/depthwise/Shape?
6model_2/model/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      28
6model_2/model/depthwise_conv2d/depthwise/dilation_rate?
(model_2/model/depthwise_conv2d/depthwiseDepthwiseConv2dNative&model_2/model/re_lu/Relu:activations:0?model_2/model/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2*
(model_2/model/depthwise_conv2d/depthwise?
5model_2/model/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp>model_2_model_depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5model_2/model/depthwise_conv2d/BiasAdd/ReadVariableOp?
&model_2/model/depthwise_conv2d/BiasAddBiasAdd1model_2/model/depthwise_conv2d/depthwise:output:0=model_2/model/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2(
&model_2/model/depthwise_conv2d/BiasAdd?
2model_2/model/batch_normalization_1/ReadVariableOpReadVariableOp;model_2_model_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype024
2model_2/model/batch_normalization_1/ReadVariableOp?
4model_2/model/batch_normalization_1/ReadVariableOp_1ReadVariableOp=model_2_model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype026
4model_2/model/batch_normalization_1/ReadVariableOp_1?
Cmodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpLmodel_2_model_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Cmodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Emodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNmodel_2_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Emodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
4model_2/model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/model_2/model/depthwise_conv2d/BiasAdd:output:0:model_2/model/batch_normalization_1/ReadVariableOp:value:0<model_2/model/batch_normalization_1/ReadVariableOp_1:value:0Kmodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Mmodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 26
4model_2/model/batch_normalization_1/FusedBatchNormV3?
model_2/model/re_lu_1/ReluRelu8model_2/model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
model_2/model/re_lu_1/Relu?
9model_2/model/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpBmodel_2_model_depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02;
9model_2/model/depthwise_conv2d_1/depthwise/ReadVariableOp?
0model_2/model/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             22
0model_2/model/depthwise_conv2d_1/depthwise/Shape?
8model_2/model/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8model_2/model/depthwise_conv2d_1/depthwise/dilation_rate?
*model_2/model/depthwise_conv2d_1/depthwiseDepthwiseConv2dNative(model_2/model/re_lu_1/Relu:activations:0Amodel_2/model/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2,
*model_2/model/depthwise_conv2d_1/depthwise?
7model_2/model/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@model_2_model_depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7model_2/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp?
(model_2/model/depthwise_conv2d_1/BiasAddBiasAdd3model_2/model/depthwise_conv2d_1/depthwise:output:0?model_2/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2*
(model_2/model/depthwise_conv2d_1/BiasAdd?
model_2/model/re_lu_2/ReluRelu1model_2/model/depthwise_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\ 2
model_2/model/re_lu_2/Relu?
%model_2/model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_2/model/concatenate/concat/axis?
 model_2/model/concatenate/concatConcatV2(model_2/model/re_lu_2/Relu:activations:0"model_2/re_lu_8/Relu:activations:0.model_2/model/concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????\@2"
 model_2/model/concatenate/concat?
,model_2/model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5model_2_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,model_2/model/conv2d_1/Conv2D/ReadVariableOp?
model_2/model/conv2d_1/Conv2DConv2D)model_2/model/concatenate/concat:output:04model_2/model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
model_2/model/conv2d_1/Conv2D?
-model_2/model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6model_2_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_2/model/conv2d_1/BiasAdd/ReadVariableOp?
model_2/model/conv2d_1/BiasAddBiasAdd&model_2/model/conv2d_1/Conv2D:output:05model_2/model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@2 
model_2/model/conv2d_1/BiasAdd?
2model_2/model/batch_normalization_2/ReadVariableOpReadVariableOp;model_2_model_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_2/model/batch_normalization_2/ReadVariableOp?
4model_2/model/batch_normalization_2/ReadVariableOp_1ReadVariableOp=model_2_model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4model_2/model/batch_normalization_2/ReadVariableOp_1?
Cmodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpLmodel_2_model_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Cmodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Emodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNmodel_2_model_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Emodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
4model_2/model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3'model_2/model/conv2d_1/BiasAdd:output:0:model_2/model/batch_normalization_2/ReadVariableOp:value:0<model_2/model/batch_normalization_2/ReadVariableOp_1:value:0Kmodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Mmodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z@:@:@:@:@:*
epsilon%o?:*
is_training( 26
4model_2/model/batch_normalization_2/FusedBatchNormV3?
model_2/model/re_lu_3/ReluRelu8model_2/model/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z@2
model_2/model/re_lu_3/Relu?
.model_2/model_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp7model_2_model_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype020
.model_2/model_1/conv2d_2/Conv2D/ReadVariableOp?
model_2/model_1/conv2d_2/Conv2DConv2D(model_2/model/re_lu_3/Relu:activations:06model_2/model_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2!
model_2/model_1/conv2d_2/Conv2D?
/model_2/model_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp8model_2_model_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/model_2/model_1/conv2d_2/BiasAdd/ReadVariableOp?
 model_2/model_1/conv2d_2/BiasAddBiasAdd(model_2/model_1/conv2d_2/Conv2D:output:07model_2/model_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2"
 model_2/model_1/conv2d_2/BiasAdd?
4model_2/model_1/batch_normalization_3/ReadVariableOpReadVariableOp=model_2_model_1_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype026
4model_2/model_1/batch_normalization_3/ReadVariableOp?
6model_2/model_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp?model_2_model_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6model_2/model_1/batch_normalization_3/ReadVariableOp_1?
Emodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_2_model_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02G
Emodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Gmodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_2_model_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gmodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
6model_2/model_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3)model_2/model_1/conv2d_2/BiasAdd:output:0<model_2/model_1/batch_normalization_3/ReadVariableOp:value:0>model_2/model_1/batch_normalization_3/ReadVariableOp_1:value:0Mmodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Omodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
is_training( 28
6model_2/model_1/batch_normalization_3/FusedBatchNormV3?
model_2/model_1/re_lu_4/ReluRelu:model_2/model_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
model_2/model_1/re_lu_4/Relu?
;model_2/model_1/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpDmodel_2_model_1_depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02=
;model_2/model_1/depthwise_conv2d_2/depthwise/ReadVariableOp?
2model_2/model_1/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             24
2model_2/model_1/depthwise_conv2d_2/depthwise/Shape?
:model_2/model_1/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/model_1/depthwise_conv2d_2/depthwise/dilation_rate?
,model_2/model_1/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative*model_2/model_1/re_lu_4/Relu:activations:0Cmodel_2/model_1/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2.
,model_2/model_1/depthwise_conv2d_2/depthwise?
9model_2/model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9model_2/model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp?
*model_2/model_1/depthwise_conv2d_2/BiasAddBiasAdd5model_2/model_1/depthwise_conv2d_2/depthwise:output:0Amodel_2/model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2,
*model_2/model_1/depthwise_conv2d_2/BiasAdd?
4model_2/model_1/batch_normalization_4/ReadVariableOpReadVariableOp=model_2_model_1_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype026
4model_2/model_1/batch_normalization_4/ReadVariableOp?
6model_2/model_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp?model_2_model_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype028
6model_2/model_1/batch_normalization_4/ReadVariableOp_1?
Emodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_2_model_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02G
Emodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Gmodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_2_model_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gmodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
6model_2/model_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV33model_2/model_1/depthwise_conv2d_2/BiasAdd:output:0<model_2/model_1/batch_normalization_4/ReadVariableOp:value:0>model_2/model_1/batch_normalization_4/ReadVariableOp_1:value:0Mmodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Omodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
is_training( 28
6model_2/model_1/batch_normalization_4/FusedBatchNormV3?
model_2/model_1/re_lu_5/ReluRelu:model_2/model_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
model_2/model_1/re_lu_5/Relu?
;model_2/model_1/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpDmodel_2_model_1_depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02=
;model_2/model_1/depthwise_conv2d_3/depthwise/ReadVariableOp?
2model_2/model_1/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             24
2model_2/model_1/depthwise_conv2d_3/depthwise/Shape?
:model_2/model_1/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/model_1/depthwise_conv2d_3/depthwise/dilation_rate?
,model_2/model_1/depthwise_conv2d_3/depthwiseDepthwiseConv2dNative*model_2/model_1/re_lu_5/Relu:activations:0Cmodel_2/model_1/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2.
,model_2/model_1/depthwise_conv2d_3/depthwise?
9model_2/model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9model_2/model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp?
*model_2/model_1/depthwise_conv2d_3/BiasAddBiasAdd5model_2/model_1/depthwise_conv2d_3/depthwise:output:0Amodel_2/model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2,
*model_2/model_1/depthwise_conv2d_3/BiasAdd?
model_2/model_1/re_lu_6/ReluRelu3model_2/model_1/depthwise_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Z 2
model_2/model_1/re_lu_6/Relu?
)model_2/model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_2/model_1/concatenate_1/concat/axis?
$model_2/model_1/concatenate_1/concatConcatV2*model_2/model_1/re_lu_6/Relu:activations:0(model_2/model/re_lu_3/Relu:activations:02model_2/model_1/concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????Z`2&
$model_2/model_1/concatenate_1/concat?
.model_2/model_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp7model_2_model_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype020
.model_2/model_1/conv2d_3/Conv2D/ReadVariableOp?
model_2/model_1/conv2d_3/Conv2DConv2D-model_2/model_1/concatenate_1/concat:output:06model_2/model_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@*
paddingVALID*
strides
2!
model_2/model_1/conv2d_3/Conv2D?
/model_2/model_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp8model_2_model_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_2/model_1/conv2d_3/BiasAdd/ReadVariableOp?
 model_2/model_1/conv2d_3/BiasAddBiasAdd(model_2/model_1/conv2d_3/Conv2D:output:07model_2/model_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@2"
 model_2/model_1/conv2d_3/BiasAdd?
4model_2/model_1/batch_normalization_5/ReadVariableOpReadVariableOp=model_2_model_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype026
4model_2/model_1/batch_normalization_5/ReadVariableOp?
6model_2/model_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp?model_2_model_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6model_2/model_1/batch_normalization_5/ReadVariableOp_1?
Emodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_2_model_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Emodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Gmodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_2_model_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gmodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
6model_2/model_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3)model_2/model_1/conv2d_3/BiasAdd:output:0<model_2/model_1/batch_normalization_5/ReadVariableOp:value:0>model_2/model_1/batch_normalization_5/ReadVariableOp_1:value:0Mmodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Omodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????X@:@:@:@:@:*
epsilon%o?:*
is_training( 28
6model_2/model_1/batch_normalization_5/FusedBatchNormV3?
model_2/model_1/re_lu_7/ReluRelu:model_2/model_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????X@2
model_2/model_1/re_lu_7/Relu?
&model_2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&model_2/conv2d_5/Conv2D/ReadVariableOp?
model_2/conv2d_5/Conv2DConv2D*model_2/model_1/re_lu_7/Relu:activations:0.model_2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@*
paddingVALID*
strides
2
model_2/conv2d_5/Conv2D?
'model_2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/conv2d_5/BiasAdd/ReadVariableOp?
model_2/conv2d_5/BiasAddBiasAdd model_2/conv2d_5/Conv2D:output:0/model_2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@2
model_2/conv2d_5/BiasAdd?
model_2/dropout/IdentityIdentity!model_2/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????+@2
model_2/dropout/Identity?
,model_2/batch_normalization_7/ReadVariableOpReadVariableOp5model_2_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model_2/batch_normalization_7/ReadVariableOp?
.model_2/batch_normalization_7/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.model_2/batch_normalization_7/ReadVariableOp_1?
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02?
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02A
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3!model_2/dropout/Identity:output:04model_2/batch_normalization_7/ReadVariableOp:value:06model_2/batch_normalization_7/ReadVariableOp_1:value:0Emodel_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????+@:@:@:@:@:*
epsilon%o?:*
is_training( 20
.model_2/batch_normalization_7/FusedBatchNormV3?
model_2/re_lu_9/ReluRelu2model_2/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????+@2
model_2/re_lu_9/Relu?
!model_2/average_pooling2d/AvgPoolAvgPool"model_2/re_lu_9/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2#
!model_2/average_pooling2d/AvgPool?
'model_2/container/Conv2D/ReadVariableOpReadVariableOp0model_2_container_conv2d_readvariableop_resource*&
_output_shapes
:@B*
dtype02)
'model_2/container/Conv2D/ReadVariableOp?
model_2/container/Conv2DConv2D*model_2/average_pooling2d/AvgPool:output:0/model_2/container/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B*
paddingVALID*
strides
2
model_2/container/Conv2D?
(model_2/container/BiasAdd/ReadVariableOpReadVariableOp1model_2_container_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype02*
(model_2/container/BiasAdd/ReadVariableOp?
model_2/container/BiasAddBiasAdd!model_2/container/Conv2D:output:00model_2/container/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B2
model_2/container/BiasAdd?
-model_2/tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-model_2/tf.compat.v1.transpose/transpose/perm?
(model_2/tf.compat.v1.transpose/transpose	Transpose"model_2/container/BiasAdd:output:06model_2/tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????B2*
(model_2/tf.compat.v1.transpose/transpose?
2model_2/tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/tf.math.reduce_mean/Mean/reduction_indices?
 model_2/tf.math.reduce_mean/MeanMean,model_2/tf.compat.v1.transpose/transpose:y:0;model_2/tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????B2"
 model_2/tf.math.reduce_mean/Mean?
IdentityIdentity)model_2/tf.math.reduce_mean/Mean:output:0>^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_6/ReadVariableOp/^model_2/batch_normalization_6/ReadVariableOp_1>^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_7/ReadVariableOp/^model_2/batch_normalization_7/ReadVariableOp_1)^model_2/container/BiasAdd/ReadVariableOp(^model_2/container/Conv2D/ReadVariableOp(^model_2/conv2d_4/BiasAdd/ReadVariableOp'^model_2/conv2d_4/Conv2D/ReadVariableOp(^model_2/conv2d_5/BiasAdd/ReadVariableOp'^model_2/conv2d_5/Conv2D/ReadVariableOpB^model_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOpD^model_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_11^model_2/model/batch_normalization/ReadVariableOp3^model_2/model/batch_normalization/ReadVariableOp_1D^model_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpF^model_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_13^model_2/model/batch_normalization_1/ReadVariableOp5^model_2/model/batch_normalization_1/ReadVariableOp_1D^model_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpF^model_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_13^model_2/model/batch_normalization_2/ReadVariableOp5^model_2/model/batch_normalization_2/ReadVariableOp_1,^model_2/model/conv2d/BiasAdd/ReadVariableOp+^model_2/model/conv2d/Conv2D/ReadVariableOp.^model_2/model/conv2d_1/BiasAdd/ReadVariableOp-^model_2/model/conv2d_1/Conv2D/ReadVariableOp6^model_2/model/depthwise_conv2d/BiasAdd/ReadVariableOp8^model_2/model/depthwise_conv2d/depthwise/ReadVariableOp8^model_2/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp:^model_2/model/depthwise_conv2d_1/depthwise/ReadVariableOpF^model_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpH^model_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_15^model_2/model_1/batch_normalization_3/ReadVariableOp7^model_2/model_1/batch_normalization_3/ReadVariableOp_1F^model_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpH^model_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_15^model_2/model_1/batch_normalization_4/ReadVariableOp7^model_2/model_1/batch_normalization_4/ReadVariableOp_1F^model_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpH^model_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_15^model_2/model_1/batch_normalization_5/ReadVariableOp7^model_2/model_1/batch_normalization_5/ReadVariableOp_10^model_2/model_1/conv2d_2/BiasAdd/ReadVariableOp/^model_2/model_1/conv2d_2/Conv2D/ReadVariableOp0^model_2/model_1/conv2d_3/BiasAdd/ReadVariableOp/^model_2/model_1/conv2d_3/Conv2D/ReadVariableOp:^model_2/model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp<^model_2/model_1/depthwise_conv2d_2/depthwise/ReadVariableOp:^model_2/model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp<^model_2/model_1/depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2~
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12\
,model_2/batch_normalization_6/ReadVariableOp,model_2/batch_normalization_6/ReadVariableOp2`
.model_2/batch_normalization_6/ReadVariableOp_1.model_2/batch_normalization_6/ReadVariableOp_12~
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12\
,model_2/batch_normalization_7/ReadVariableOp,model_2/batch_normalization_7/ReadVariableOp2`
.model_2/batch_normalization_7/ReadVariableOp_1.model_2/batch_normalization_7/ReadVariableOp_12T
(model_2/container/BiasAdd/ReadVariableOp(model_2/container/BiasAdd/ReadVariableOp2R
'model_2/container/Conv2D/ReadVariableOp'model_2/container/Conv2D/ReadVariableOp2R
'model_2/conv2d_4/BiasAdd/ReadVariableOp'model_2/conv2d_4/BiasAdd/ReadVariableOp2P
&model_2/conv2d_4/Conv2D/ReadVariableOp&model_2/conv2d_4/Conv2D/ReadVariableOp2R
'model_2/conv2d_5/BiasAdd/ReadVariableOp'model_2/conv2d_5/BiasAdd/ReadVariableOp2P
&model_2/conv2d_5/Conv2D/ReadVariableOp&model_2/conv2d_5/Conv2D/ReadVariableOp2?
Amodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOpAmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12d
0model_2/model/batch_normalization/ReadVariableOp0model_2/model/batch_normalization/ReadVariableOp2h
2model_2/model/batch_normalization/ReadVariableOp_12model_2/model/batch_normalization/ReadVariableOp_12?
Cmodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpCmodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Emodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Emodel_2/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12h
2model_2/model/batch_normalization_1/ReadVariableOp2model_2/model/batch_normalization_1/ReadVariableOp2l
4model_2/model/batch_normalization_1/ReadVariableOp_14model_2/model/batch_normalization_1/ReadVariableOp_12?
Cmodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpCmodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Emodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Emodel_2/model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12h
2model_2/model/batch_normalization_2/ReadVariableOp2model_2/model/batch_normalization_2/ReadVariableOp2l
4model_2/model/batch_normalization_2/ReadVariableOp_14model_2/model/batch_normalization_2/ReadVariableOp_12Z
+model_2/model/conv2d/BiasAdd/ReadVariableOp+model_2/model/conv2d/BiasAdd/ReadVariableOp2X
*model_2/model/conv2d/Conv2D/ReadVariableOp*model_2/model/conv2d/Conv2D/ReadVariableOp2^
-model_2/model/conv2d_1/BiasAdd/ReadVariableOp-model_2/model/conv2d_1/BiasAdd/ReadVariableOp2\
,model_2/model/conv2d_1/Conv2D/ReadVariableOp,model_2/model/conv2d_1/Conv2D/ReadVariableOp2n
5model_2/model/depthwise_conv2d/BiasAdd/ReadVariableOp5model_2/model/depthwise_conv2d/BiasAdd/ReadVariableOp2r
7model_2/model/depthwise_conv2d/depthwise/ReadVariableOp7model_2/model/depthwise_conv2d/depthwise/ReadVariableOp2r
7model_2/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp7model_2/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp2v
9model_2/model/depthwise_conv2d_1/depthwise/ReadVariableOp9model_2/model/depthwise_conv2d_1/depthwise/ReadVariableOp2?
Emodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpEmodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Gmodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Gmodel_2/model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12l
4model_2/model_1/batch_normalization_3/ReadVariableOp4model_2/model_1/batch_normalization_3/ReadVariableOp2p
6model_2/model_1/batch_normalization_3/ReadVariableOp_16model_2/model_1/batch_normalization_3/ReadVariableOp_12?
Emodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpEmodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Gmodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Gmodel_2/model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12l
4model_2/model_1/batch_normalization_4/ReadVariableOp4model_2/model_1/batch_normalization_4/ReadVariableOp2p
6model_2/model_1/batch_normalization_4/ReadVariableOp_16model_2/model_1/batch_normalization_4/ReadVariableOp_12?
Emodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpEmodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Gmodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Gmodel_2/model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12l
4model_2/model_1/batch_normalization_5/ReadVariableOp4model_2/model_1/batch_normalization_5/ReadVariableOp2p
6model_2/model_1/batch_normalization_5/ReadVariableOp_16model_2/model_1/batch_normalization_5/ReadVariableOp_12b
/model_2/model_1/conv2d_2/BiasAdd/ReadVariableOp/model_2/model_1/conv2d_2/BiasAdd/ReadVariableOp2`
.model_2/model_1/conv2d_2/Conv2D/ReadVariableOp.model_2/model_1/conv2d_2/Conv2D/ReadVariableOp2b
/model_2/model_1/conv2d_3/BiasAdd/ReadVariableOp/model_2/model_1/conv2d_3/BiasAdd/ReadVariableOp2`
.model_2/model_1/conv2d_3/Conv2D/ReadVariableOp.model_2/model_1/conv2d_3/Conv2D/ReadVariableOp2v
9model_2/model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp9model_2/model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp2z
;model_2/model_1/depthwise_conv2d_2/depthwise/ReadVariableOp;model_2/model_1/depthwise_conv2d_2/depthwise/ReadVariableOp2v
9model_2/model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp9model_2/model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp2z
;model_2/model_1/depthwise_conv2d_3/depthwise/ReadVariableOp;model_2/model_1/depthwise_conv2d_3/depthwise/ReadVariableOp:V R
/
_output_shapes
:?????????^

_user_specified_nameinput
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_511315553

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_511310959

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_8_layer_call_and_return_conditional_losses_511314291

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????\ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????\ :W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_layer_call_fn_511311668

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_5113116622
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_layer_call_fn_511315499

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_5113101012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314245

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
+__inference_model_1_layer_call_fn_511311176
input_2!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:`@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113111332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????@
!
_user_specified_name	input_2
?
l
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_511311662

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_511315830

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_511310810

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
H__inference_container_layer_call_and_return_conditional_losses_511315490

inputs8
conv2d_readvariableop_resource:@B-
biasadd_readvariableop_resource:B
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@B*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:B*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_511315902

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
+__inference_model_1_layer_call_fn_511314971

inputs!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:`@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????X@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113123772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????X@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????Z@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?

?
H__inference_container_layer_call_and_return_conditional_losses_511312035

inputs8
conv2d_readvariableop_resource:@B-
biasadd_readvariableop_resource:B
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@B*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:B*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_511315786

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?m
?
D__inference_model_layer_call_and_return_conditional_losses_511311804

inputs?
%conv2d_conv2d_readvariableop_resource:  4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: L
2depthwise_conv2d_depthwise_readvariableop_resource: >
0depthwise_conv2d_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_1_depthwise_readvariableop_resource: @
2depthwise_conv2d_1_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'depthwise_conv2d/BiasAdd/ReadVariableOp?)depthwise_conv2d/depthwise/ReadVariableOp?)depthwise_conv2d_1/BiasAdd/ReadVariableOp?+depthwise_conv2d_1/depthwise/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2

re_lu/Relu?
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp2depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp?
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape?
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate?
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
depthwise_conv2d/depthwise?
'depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp0depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'depthwise_conv2d/BiasAdd/ReadVariableOp?
depthwise_conv2d/BiasAddBiasAdd#depthwise_conv2d/depthwise:output:0/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
depthwise_conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!depthwise_conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_1/Relu?
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp?
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape?
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate?
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_1/Relu:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
depthwise_conv2d_1/depthwise?
)depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_1/BiasAdd/ReadVariableOp?
depthwise_conv2d_1/BiasAddBiasAdd%depthwise_conv2d_1/depthwise:output:01depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
depthwise_conv2d_1/BiasAdd?
re_lu_2/ReluRelu#depthwise_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_2/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2re_lu_2/Relu:activations:0inputs concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????\@2
concatenate/concat?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconcatenate/concat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@2
conv2d_1/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
re_lu_3/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z@2
re_lu_3/Relu?
IdentityIdentityre_lu_3/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^depthwise_conv2d/BiasAdd/ReadVariableOp*^depthwise_conv2d/depthwise/ReadVariableOp*^depthwise_conv2d_1/BiasAdd/ReadVariableOp,^depthwise_conv2d_1/depthwise/ReadVariableOp*
T0*/
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????\ : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'depthwise_conv2d/BiasAdd/ReadVariableOp'depthwise_conv2d/BiasAdd/ReadVariableOp2V
)depthwise_conv2d/depthwise/ReadVariableOp)depthwise_conv2d/depthwise/ReadVariableOp2V
)depthwise_conv2d_1/BiasAdd/ReadVariableOp)depthwise_conv2d_1/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_1/depthwise/ReadVariableOp+depthwise_conv2d_1/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_7_layer_call_fn_511315363

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5113115962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_511309831

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314263

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????\ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315407

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_5_layer_call_and_return_conditional_losses_511311077

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?P
?
F__inference_model_2_layer_call_and_return_conditional_losses_511313370	
input,
conv2d_4_511313246:  
conv2d_4_511313248: -
batch_normalization_6_511313251: -
batch_normalization_6_511313253: -
batch_normalization_6_511313255: -
batch_normalization_6_511313257: )
model_511313261:  
model_511313263: 
model_511313265: 
model_511313267: 
model_511313269: 
model_511313271: )
model_511313273: 
model_511313275: 
model_511313277: 
model_511313279: 
model_511313281: 
model_511313283: )
model_511313285: 
model_511313287: )
model_511313289:@@
model_511313291:@
model_511313293:@
model_511313295:@
model_511313297:@
model_511313299:@+
model_1_511313302:@ 
model_1_511313304: 
model_1_511313306: 
model_1_511313308: 
model_1_511313310: 
model_1_511313312: +
model_1_511313314: 
model_1_511313316: 
model_1_511313318: 
model_1_511313320: 
model_1_511313322: 
model_1_511313324: +
model_1_511313326: 
model_1_511313328: +
model_1_511313330:`@
model_1_511313332:@
model_1_511313334:@
model_1_511313336:@
model_1_511313338:@
model_1_511313340:@,
conv2d_5_511313343:@@ 
conv2d_5_511313345:@-
batch_normalization_7_511313349:@-
batch_normalization_7_511313351:@-
batch_normalization_7_511313353:@-
batch_normalization_7_511313355:@-
container_511313360:@B!
container_511313362:B
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?!container/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dropout/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_4_511313246conv2d_4_511313248*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_5113116852"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_6_511313251batch_normalization_6_511313253batch_normalization_6_511313255batch_normalization_6_511313257*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5113126292/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_5113117232
re_lu_8/PartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0model_511313261model_511313263model_511313265model_511313267model_511313269model_511313271model_511313273model_511313275model_511313277model_511313279model_511313281model_511313283model_511313285model_511313287model_511313289model_511313291model_511313293model_511313295model_511313297model_511313299* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????Z@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113125472
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0model_1_511313302model_1_511313304model_1_511313306model_1_511313308model_1_511313310model_1_511313312model_1_511313314model_1_511313316model_1_511313318model_1_511313320model_1_511313322model_1_511313324model_1_511313326model_1_511313328model_1_511313330model_1_511313332model_1_511313334model_1_511313336model_1_511313338model_1_511313340* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????X@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113123772!
model_1/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(model_1/StatefulPartitionedCall:output:0conv2d_5_511313343conv2d_5_511313345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_5113119772"
 conv2d_5/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_5113122372!
dropout/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0batch_normalization_7_511313349batch_normalization_7_511313351batch_normalization_7_511313353batch_normalization_7_511313355*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5113122062/
-batch_normalization_7/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_9_layer_call_and_return_conditional_losses_5113120222
re_lu_9/PartitionedCall?
!average_pooling2d/PartitionedCallPartitionedCall re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_5113116622#
!average_pooling2d/PartitionedCall?
!container/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0container_511313360container_511313362*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_container_layer_call_and_return_conditional_losses_5113120352#
!container/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose*container/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????B2"
 tf.compat.v1.transpose/transpose?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean$tf.compat.v1.transpose/transpose:y:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????B2
tf.math.reduce_mean/Mean?
IdentityIdentity!tf.math.reduce_mean/Mean:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall"^container/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!container/StatefulPartitionedCall!container/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????^

_user_specified_nameinput
?
?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_511310175

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_6_layer_call_and_return_conditional_losses_511311089

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ވ
?
F__inference_model_1_layer_call_and_return_conditional_losses_511315131

inputsA
'conv2d_2_conv2d_readvariableop_resource:@ 6
(conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_2_depthwise_readvariableop_resource: @
2depthwise_conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_3_depthwise_readvariableop_resource: @
2depthwise_conv2d_3_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:`@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity??$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?)depthwise_conv2d_2/BiasAdd/ReadVariableOp?+depthwise_conv2d_2/depthwise/ReadVariableOp?)depthwise_conv2d_3/BiasAdd/ReadVariableOp?+depthwise_conv2d_3/depthwise/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
conv2d_2/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
re_lu_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_4/Relu?
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp?
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_2/depthwise/Shape?
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate?
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise?
)depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_2/BiasAdd/ReadVariableOp?
depthwise_conv2d_2/BiasAddBiasAdd%depthwise_conv2d_2/depthwise:output:01depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
depthwise_conv2d_2/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d_2/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
re_lu_5/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_5/Relu?
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp?
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_3/depthwise/Shape?
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate?
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_5/Relu:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise?
)depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_3/BiasAdd/ReadVariableOp?
depthwise_conv2d_3/BiasAddBiasAdd%depthwise_conv2d_3/depthwise:output:01depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
depthwise_conv2d_3/BiasAdd?
re_lu_6/ReluRelu#depthwise_conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_6/Relux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2re_lu_6/Relu:activations:0inputs"concatenate_1/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????`2
concatenate_1/concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcatenate_1/concat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
conv2d_3/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
re_lu_7/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
re_lu_7/Relu?	
IdentityIdentityre_lu_7/Relu:activations:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*^depthwise_conv2d_2/BiasAdd/ReadVariableOp,^depthwise_conv2d_2/depthwise/ReadVariableOp*^depthwise_conv2d_3/BiasAdd/ReadVariableOp,^depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2V
)depthwise_conv2d_2/BiasAdd/ReadVariableOp)depthwise_conv2d_2/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_2/depthwise/ReadVariableOp+depthwise_conv2d_2/depthwise/ReadVariableOp2V
)depthwise_conv2d_3/BiasAdd/ReadVariableOp)depthwise_conv2d_3/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_3/depthwise/ReadVariableOp+depthwise_conv2d_3/depthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_511309682

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_511314426

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????Z@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113118042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????\ : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?B
?

D__inference_model_layer_call_and_return_conditional_losses_511310595
input_1*
conv2d_511310542:  
conv2d_511310544: +
batch_normalization_511310547: +
batch_normalization_511310549: +
batch_normalization_511310551: +
batch_normalization_511310553: 4
depthwise_conv2d_511310557: (
depthwise_conv2d_511310559: -
batch_normalization_1_511310562: -
batch_normalization_1_511310564: -
batch_normalization_1_511310566: -
batch_normalization_1_511310568: 6
depthwise_conv2d_1_511310572: *
depthwise_conv2d_1_511310574: ,
conv2d_1_511310579:@@ 
conv2d_1_511310581:@-
batch_normalization_2_511310584:@-
batch_normalization_2_511310586:@-
batch_normalization_2_511310588:@-
batch_normalization_2_511310590:@
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(depthwise_conv2d/StatefulPartitionedCall?*depthwise_conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_511310542conv2d_511310544*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_5113101012 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_511310547batch_normalization_511310549batch_normalization_511310551batch_normalization_511310553*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_5113097262-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_5113101212
re_lu/PartitionedCall?
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0depthwise_conv2d_511310557depthwise_conv2d_511310559*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_5113097992*
(depthwise_conv2d/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:0batch_normalization_1_511310562batch_normalization_1_511310564batch_normalization_1_511310566batch_normalization_1_511310568*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5113098752/
-batch_normalization_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_5113101422
re_lu_1/PartitionedCall?
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0depthwise_conv2d_1_511310572depthwise_conv2d_1_511310574*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_5113099482,
*depthwise_conv2d_1/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_5113101542
re_lu_2/PartitionedCall?
concatenate/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_5113101632
concatenate/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_1_511310579conv2d_1_511310581*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_5113101752"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_511310584batch_normalization_2_511310586batch_normalization_2_511310588batch_normalization_2_511310590*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5113100242/
-batch_normalization_2/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_5113101952
re_lu_3/PartitionedCall?
IdentityIdentity re_lu_3/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall:j f
A
_output_shapes/
-:+??????????????????????????? 
!
_user_specified_name	input_1
?O
?
F__inference_model_2_layer_call_and_return_conditional_losses_511313243	
input,
conv2d_4_511313119:  
conv2d_4_511313121: -
batch_normalization_6_511313124: -
batch_normalization_6_511313126: -
batch_normalization_6_511313128: -
batch_normalization_6_511313130: )
model_511313134:  
model_511313136: 
model_511313138: 
model_511313140: 
model_511313142: 
model_511313144: )
model_511313146: 
model_511313148: 
model_511313150: 
model_511313152: 
model_511313154: 
model_511313156: )
model_511313158: 
model_511313160: )
model_511313162:@@
model_511313164:@
model_511313166:@
model_511313168:@
model_511313170:@
model_511313172:@+
model_1_511313175:@ 
model_1_511313177: 
model_1_511313179: 
model_1_511313181: 
model_1_511313183: 
model_1_511313185: +
model_1_511313187: 
model_1_511313189: 
model_1_511313191: 
model_1_511313193: 
model_1_511313195: 
model_1_511313197: +
model_1_511313199: 
model_1_511313201: +
model_1_511313203:`@
model_1_511313205:@
model_1_511313207:@
model_1_511313209:@
model_1_511313211:@
model_1_511313213:@,
conv2d_5_511313216:@@ 
conv2d_5_511313218:@-
batch_normalization_7_511313222:@-
batch_normalization_7_511313224:@-
batch_normalization_7_511313226:@-
batch_normalization_7_511313228:@-
container_511313233:@B!
container_511313235:B
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?!container/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_4_511313119conv2d_4_511313121*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_5113116852"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_6_511313124batch_normalization_6_511313126batch_normalization_6_511313128batch_normalization_6_511313130*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5113117082/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_5113117232
re_lu_8/PartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0model_511313134model_511313136model_511313138model_511313140model_511313142model_511313144model_511313146model_511313148model_511313150model_511313152model_511313154model_511313156model_511313158model_511313160model_511313162model_511313164model_511313166model_511313168model_511313170model_511313172* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????Z@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113118042
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0model_1_511313175model_1_511313177model_1_511313179model_1_511313181model_1_511313183model_1_511313185model_1_511313187model_1_511313189model_1_511313191model_1_511313193model_1_511313195model_1_511313197model_1_511313199model_1_511313201model_1_511313203model_1_511313205model_1_511313207model_1_511313209model_1_511313211model_1_511313213* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????X@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113119252!
model_1/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(model_1/StatefulPartitionedCall:output:0conv2d_5_511313216conv2d_5_511313218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_5113119772"
 conv2d_5/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_5113119882
dropout/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0batch_normalization_7_511313222batch_normalization_7_511313224batch_normalization_7_511313226batch_normalization_7_511313228*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5113120072/
-batch_normalization_7/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_9_layer_call_and_return_conditional_losses_5113120222
re_lu_9/PartitionedCall?
!average_pooling2d/PartitionedCallPartitionedCall re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_5113116622#
!average_pooling2d/PartitionedCall?
!container/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0container_511313233container_511313235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_container_layer_call_and_return_conditional_losses_5113120352#
!container/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose*container/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????B2"
 tf.compat.v1.transpose/transpose?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean$tf.compat.v1.transpose/transpose:y:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????B2
tf.math.reduce_mean/Mean?
IdentityIdentity!tf.math.reduce_mean/Mean:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall"^container/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!container/StatefulPartitionedCall!container/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????^

_user_specified_nameinput
?
v
L__inference_concatenate_1_layer_call_and_return_conditional_losses_511311098

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????`2
concat}
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+??????????????????????????? :+???????????????????????????@:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_511311110

inputs8
conv2d_readvariableop_resource:`@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?
G
+__inference_re_lu_5_layer_call_fn_511315925

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_5113110772
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_3_layer_call_fn_511315799

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5113106172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_511310024

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_4_layer_call_and_return_conditional_losses_511315858

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
G
+__inference_re_lu_3_layer_call_fn_511315762

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_5113101952
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_2_layer_call_and_return_conditional_losses_511310154

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_6_layer_call_fn_511314183

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5113096002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?:
F__inference_model_2_layer_call_and_return_conditional_losses_511314138

inputsA
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: E
+model_conv2d_conv2d_readvariableop_resource:  :
,model_conv2d_biasadd_readvariableop_resource: ?
1model_batch_normalization_readvariableop_resource: A
3model_batch_normalization_readvariableop_1_resource: P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource: R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: R
8model_depthwise_conv2d_depthwise_readvariableop_resource: D
6model_depthwise_conv2d_biasadd_readvariableop_resource: A
3model_batch_normalization_1_readvariableop_resource: C
5model_batch_normalization_1_readvariableop_1_resource: R
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: T
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: T
:model_depthwise_conv2d_1_depthwise_readvariableop_resource: F
8model_depthwise_conv2d_1_biasadd_readvariableop_resource: G
-model_conv2d_1_conv2d_readvariableop_resource:@@<
.model_conv2d_1_biasadd_readvariableop_resource:@A
3model_batch_normalization_2_readvariableop_resource:@C
5model_batch_normalization_2_readvariableop_1_resource:@R
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@I
/model_1_conv2d_2_conv2d_readvariableop_resource:@ >
0model_1_conv2d_2_biasadd_readvariableop_resource: C
5model_1_batch_normalization_3_readvariableop_resource: E
7model_1_batch_normalization_3_readvariableop_1_resource: T
Fmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: V
Hmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: V
<model_1_depthwise_conv2d_2_depthwise_readvariableop_resource: H
:model_1_depthwise_conv2d_2_biasadd_readvariableop_resource: C
5model_1_batch_normalization_4_readvariableop_resource: E
7model_1_batch_normalization_4_readvariableop_1_resource: T
Fmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: V
Hmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: V
<model_1_depthwise_conv2d_3_depthwise_readvariableop_resource: H
:model_1_depthwise_conv2d_3_biasadd_readvariableop_resource: I
/model_1_conv2d_3_conv2d_readvariableop_resource:`@>
0model_1_conv2d_3_biasadd_readvariableop_resource:@C
5model_1_batch_normalization_5_readvariableop_resource:@E
7model_1_batch_normalization_5_readvariableop_1_resource:@T
Fmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_7_readvariableop_resource:@=
/batch_normalization_7_readvariableop_1_resource:@L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@B
(container_conv2d_readvariableop_resource:@B7
)container_biasadd_readvariableop_resource:B
identity??$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?$batch_normalization_7/AssignNewValue?&batch_normalization_7/AssignNewValue_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1? container/BiasAdd/ReadVariableOp?container/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?(model/batch_normalization/AssignNewValue?*model/batch_normalization/AssignNewValue_1?9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?*model/batch_normalization_1/AssignNewValue?,model/batch_normalization_1/AssignNewValue_1?;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_1/ReadVariableOp?,model/batch_normalization_1/ReadVariableOp_1?*model/batch_normalization_2/AssignNewValue?,model/batch_normalization_2/AssignNewValue_1?;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_2/ReadVariableOp?,model/batch_normalization_2/ReadVariableOp_1?#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?-model/depthwise_conv2d/BiasAdd/ReadVariableOp?/model/depthwise_conv2d/depthwise/ReadVariableOp?/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp?1model/depthwise_conv2d_1/depthwise/ReadVariableOp?,model_1/batch_normalization_3/AssignNewValue?.model_1/batch_normalization_3/AssignNewValue_1?=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_3/ReadVariableOp?.model_1/batch_normalization_3/ReadVariableOp_1?,model_1/batch_normalization_4/AssignNewValue?.model_1/batch_normalization_4/AssignNewValue_1?=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_4/ReadVariableOp?.model_1/batch_normalization_4/ReadVariableOp_1?,model_1/batch_normalization_5/AssignNewValue?.model_1/batch_normalization_5/AssignNewValue_1?=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_5/ReadVariableOp?.model_1/batch_normalization_5/ReadVariableOp_1?'model_1/conv2d_2/BiasAdd/ReadVariableOp?&model_1/conv2d_2/Conv2D/ReadVariableOp?'model_1/conv2d_3/BiasAdd/ReadVariableOp?&model_1/conv2d_3/Conv2D/ReadVariableOp?1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp?3model_1/depthwise_conv2d_2/depthwise/ReadVariableOp?1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp?3model_1/depthwise_conv2d_3/depthwise/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
conv2d_4/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1?
re_lu_8/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_8/Relu?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dre_lu_8/Relu:activations:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
model/conv2d/BiasAdd?
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02*
(model/batch_normalization/ReadVariableOp?
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization/ReadVariableOp_1?
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02;
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/BiasAdd:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2,
*model/batch_normalization/FusedBatchNormV3?
(model/batch_normalization/AssignNewValueAssignVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource7model/batch_normalization/FusedBatchNormV3:batch_mean:0:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02*
(model/batch_normalization/AssignNewValue?
*model/batch_normalization/AssignNewValue_1AssignVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource;model/batch_normalization/FusedBatchNormV3:batch_variance:0<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02,
*model/batch_normalization/AssignNewValue_1?
model/re_lu/ReluRelu.model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
model/re_lu/Relu?
/model/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp8model_depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype021
/model/depthwise_conv2d/depthwise/ReadVariableOp?
&model/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&model/depthwise_conv2d/depthwise/Shape?
.model/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.model/depthwise_conv2d/depthwise/dilation_rate?
 model/depthwise_conv2d/depthwiseDepthwiseConv2dNativemodel/re_lu/Relu:activations:07model/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2"
 model/depthwise_conv2d/depthwise?
-model/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp6model_depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/depthwise_conv2d/BiasAdd/ReadVariableOp?
model/depthwise_conv2d/BiasAddBiasAdd)model/depthwise_conv2d/depthwise:output:05model/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2 
model/depthwise_conv2d/BiasAdd?
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization_1/ReadVariableOp?
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,model/batch_normalization_1/ReadVariableOp_1?
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3'model/depthwise_conv2d/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2.
,model/batch_normalization_1/FusedBatchNormV3?
*model/batch_normalization_1/AssignNewValueAssignVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource9model/batch_normalization_1/FusedBatchNormV3:batch_mean:0<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02,
*model/batch_normalization_1/AssignNewValue?
,model/batch_normalization_1/AssignNewValue_1AssignVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource=model/batch_normalization_1/FusedBatchNormV3:batch_variance:0>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02.
,model/batch_normalization_1/AssignNewValue_1?
model/re_lu_1/ReluRelu0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
model/re_lu_1/Relu?
1model/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp:model_depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype023
1model/depthwise_conv2d_1/depthwise/ReadVariableOp?
(model/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(model/depthwise_conv2d_1/depthwise/Shape?
0model/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      22
0model/depthwise_conv2d_1/depthwise/dilation_rate?
"model/depthwise_conv2d_1/depthwiseDepthwiseConv2dNative model/re_lu_1/Relu:activations:09model/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2$
"model/depthwise_conv2d_1/depthwise?
/model/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp8model_depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp?
 model/depthwise_conv2d_1/BiasAddBiasAdd+model/depthwise_conv2d_1/depthwise:output:07model/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2"
 model/depthwise_conv2d_1/BiasAdd?
model/re_lu_2/ReluRelu)model/depthwise_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\ 2
model/re_lu_2/Relu?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2 model/re_lu_2/Relu:activations:0re_lu_8/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????\@2
model/concatenate/concat?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2D!model/concatenate/concat:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@2
model/conv2d_1/BiasAdd?
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_2/ReadVariableOp?
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_2/ReadVariableOp_1?
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3model/conv2d_1/BiasAdd:output:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2.
,model/batch_normalization_2/FusedBatchNormV3?
*model/batch_normalization_2/AssignNewValueAssignVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource9model/batch_normalization_2/FusedBatchNormV3:batch_mean:0<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02,
*model/batch_normalization_2/AssignNewValue?
,model/batch_normalization_2/AssignNewValue_1AssignVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource=model/batch_normalization_2/FusedBatchNormV3:batch_variance:0>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02.
,model/batch_normalization_2/AssignNewValue_1?
model/re_lu_3/ReluRelu0model/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z@2
model/re_lu_3/Relu?
&model_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02(
&model_1/conv2d_2/Conv2D/ReadVariableOp?
model_1/conv2d_2/Conv2DConv2D model/re_lu_3/Relu:activations:0.model_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
model_1/conv2d_2/Conv2D?
'model_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv2d_2/BiasAdd/ReadVariableOp?
model_1/conv2d_2/BiasAddBiasAdd model_1/conv2d_2/Conv2D:output:0/model_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
model_1/conv2d_2/BiasAdd?
,model_1/batch_normalization_3/ReadVariableOpReadVariableOp5model_1_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_3/ReadVariableOp?
.model_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_3/ReadVariableOp_1?
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3!model_1/conv2d_2/BiasAdd:output:04model_1/batch_normalization_3/ReadVariableOp:value:06model_1/batch_normalization_3/ReadVariableOp_1:value:0Emodel_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<20
.model_1/batch_normalization_3/FusedBatchNormV3?
,model_1/batch_normalization_3/AssignNewValueAssignVariableOpFmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource;model_1/batch_normalization_3/FusedBatchNormV3:batch_mean:0>^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,model_1/batch_normalization_3/AssignNewValue?
.model_1/batch_normalization_3/AssignNewValue_1AssignVariableOpHmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource?model_1/batch_normalization_3/FusedBatchNormV3:batch_variance:0@^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_1/batch_normalization_3/AssignNewValue_1?
model_1/re_lu_4/ReluRelu2model_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
model_1/re_lu_4/Relu?
3model_1/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp<model_1_depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype025
3model_1/depthwise_conv2d_2/depthwise/ReadVariableOp?
*model_1/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*model_1/depthwise_conv2d_2/depthwise/Shape?
2model_1/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2model_1/depthwise_conv2d_2/depthwise/dilation_rate?
$model_1/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative"model_1/re_lu_4/Relu:activations:0;model_1/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2&
$model_1/depthwise_conv2d_2/depthwise?
1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:model_1_depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp?
"model_1/depthwise_conv2d_2/BiasAddBiasAdd-model_1/depthwise_conv2d_2/depthwise:output:09model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2$
"model_1/depthwise_conv2d_2/BiasAdd?
,model_1/batch_normalization_4/ReadVariableOpReadVariableOp5model_1_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_4/ReadVariableOp?
.model_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_4/ReadVariableOp_1?
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+model_1/depthwise_conv2d_2/BiasAdd:output:04model_1/batch_normalization_4/ReadVariableOp:value:06model_1/batch_normalization_4/ReadVariableOp_1:value:0Emodel_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<20
.model_1/batch_normalization_4/FusedBatchNormV3?
,model_1/batch_normalization_4/AssignNewValueAssignVariableOpFmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource;model_1/batch_normalization_4/FusedBatchNormV3:batch_mean:0>^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,model_1/batch_normalization_4/AssignNewValue?
.model_1/batch_normalization_4/AssignNewValue_1AssignVariableOpHmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource?model_1/batch_normalization_4/FusedBatchNormV3:batch_variance:0@^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_1/batch_normalization_4/AssignNewValue_1?
model_1/re_lu_5/ReluRelu2model_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
model_1/re_lu_5/Relu?
3model_1/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp<model_1_depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype025
3model_1/depthwise_conv2d_3/depthwise/ReadVariableOp?
*model_1/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*model_1/depthwise_conv2d_3/depthwise/Shape?
2model_1/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2model_1/depthwise_conv2d_3/depthwise/dilation_rate?
$model_1/depthwise_conv2d_3/depthwiseDepthwiseConv2dNative"model_1/re_lu_5/Relu:activations:0;model_1/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2&
$model_1/depthwise_conv2d_3/depthwise?
1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:model_1_depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp?
"model_1/depthwise_conv2d_3/BiasAddBiasAdd-model_1/depthwise_conv2d_3/depthwise:output:09model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2$
"model_1/depthwise_conv2d_3/BiasAdd?
model_1/re_lu_6/ReluRelu+model_1/depthwise_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Z 2
model_1/re_lu_6/Relu?
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axis?
model_1/concatenate_1/concatConcatV2"model_1/re_lu_6/Relu:activations:0 model/re_lu_3/Relu:activations:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????Z`2
model_1/concatenate_1/concat?
&model_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02(
&model_1/conv2d_3/Conv2D/ReadVariableOp?
model_1/conv2d_3/Conv2DConv2D%model_1/concatenate_1/concat:output:0.model_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@*
paddingVALID*
strides
2
model_1/conv2d_3/Conv2D?
'model_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv2d_3/BiasAdd/ReadVariableOp?
model_1/conv2d_3/BiasAddBiasAdd model_1/conv2d_3/Conv2D:output:0/model_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@2
model_1/conv2d_3/BiasAdd?
,model_1/batch_normalization_5/ReadVariableOpReadVariableOp5model_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model_1/batch_normalization_5/ReadVariableOp?
.model_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.model_1/batch_normalization_5/ReadVariableOp_1?
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02?
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02A
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3!model_1/conv2d_3/BiasAdd:output:04model_1/batch_normalization_5/ReadVariableOp:value:06model_1/batch_normalization_5/ReadVariableOp_1:value:0Emodel_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????X@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<20
.model_1/batch_normalization_5/FusedBatchNormV3?
,model_1/batch_normalization_5/AssignNewValueAssignVariableOpFmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource;model_1/batch_normalization_5/FusedBatchNormV3:batch_mean:0>^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,model_1/batch_normalization_5/AssignNewValue?
.model_1/batch_normalization_5/AssignNewValue_1AssignVariableOpHmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource?model_1/batch_normalization_5/FusedBatchNormV3:batch_variance:0@^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_1/batch_normalization_5/AssignNewValue_1?
model_1/re_lu_7/ReluRelu2model_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????X@2
model_1/re_lu_7/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D"model_1/re_lu_7/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@*
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@2
conv2d_5/BiasAdds
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulconv2d_5/BiasAdd:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????+@2
dropout/dropout/Mulw
dropout/dropout/ShapeShapeconv2d_5/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????+@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????+@2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????+@2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????+@2
dropout/dropout/Mul_1?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3dropout/dropout/Mul_1:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????+@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_7/FusedBatchNormV3?
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue?
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1?
re_lu_9/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????+@2
re_lu_9/Relu?
average_pooling2d/AvgPoolAvgPoolre_lu_9/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool?
container/Conv2D/ReadVariableOpReadVariableOp(container_conv2d_readvariableop_resource*&
_output_shapes
:@B*
dtype02!
container/Conv2D/ReadVariableOp?
container/Conv2DConv2D"average_pooling2d/AvgPool:output:0'container/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B*
paddingVALID*
strides
2
container/Conv2D?
 container/BiasAdd/ReadVariableOpReadVariableOp)container_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype02"
 container/BiasAdd/ReadVariableOp?
container/BiasAddBiasAddcontainer/Conv2D:output:0(container/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B2
container/BiasAdd?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transposecontainer/BiasAdd:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????B2"
 tf.compat.v1.transpose/transpose?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean$tf.compat.v1.transpose/transpose:y:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????B2
tf.math.reduce_mean/Mean?
IdentityIdentity!tf.math.reduce_mean/Mean:output:0%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^container/BiasAdd/ReadVariableOp ^container/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp)^model/batch_normalization/AssignNewValue+^model/batch_normalization/AssignNewValue_1:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1+^model/batch_normalization_1/AssignNewValue-^model/batch_normalization_1/AssignNewValue_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1+^model/batch_normalization_2/AssignNewValue-^model/batch_normalization_2/AssignNewValue_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp.^model/depthwise_conv2d/BiasAdd/ReadVariableOp0^model/depthwise_conv2d/depthwise/ReadVariableOp0^model/depthwise_conv2d_1/BiasAdd/ReadVariableOp2^model/depthwise_conv2d_1/depthwise/ReadVariableOp-^model_1/batch_normalization_3/AssignNewValue/^model_1/batch_normalization_3/AssignNewValue_1>^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_3/ReadVariableOp/^model_1/batch_normalization_3/ReadVariableOp_1-^model_1/batch_normalization_4/AssignNewValue/^model_1/batch_normalization_4/AssignNewValue_1>^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_4/ReadVariableOp/^model_1/batch_normalization_4/ReadVariableOp_1-^model_1/batch_normalization_5/AssignNewValue/^model_1/batch_normalization_5/AssignNewValue_1>^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_5/ReadVariableOp/^model_1/batch_normalization_5/ReadVariableOp_1(^model_1/conv2d_2/BiasAdd/ReadVariableOp'^model_1/conv2d_2/Conv2D/ReadVariableOp(^model_1/conv2d_3/BiasAdd/ReadVariableOp'^model_1/conv2d_3/Conv2D/ReadVariableOp2^model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp4^model_1/depthwise_conv2d_2/depthwise/ReadVariableOp2^model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp4^model_1/depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 container/BiasAdd/ReadVariableOp container/BiasAdd/ReadVariableOp2B
container/Conv2D/ReadVariableOpcontainer/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2T
(model/batch_normalization/AssignNewValue(model/batch_normalization/AssignNewValue2X
*model/batch_normalization/AssignNewValue_1*model/batch_normalization/AssignNewValue_12v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12X
*model/batch_normalization_1/AssignNewValue*model/batch_normalization_1/AssignNewValue2\
,model/batch_normalization_1/AssignNewValue_1,model/batch_normalization_1/AssignNewValue_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12X
*model/batch_normalization_2/AssignNewValue*model/batch_normalization_2/AssignNewValue2\
,model/batch_normalization_2/AssignNewValue_1,model/batch_normalization_2/AssignNewValue_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2^
-model/depthwise_conv2d/BiasAdd/ReadVariableOp-model/depthwise_conv2d/BiasAdd/ReadVariableOp2b
/model/depthwise_conv2d/depthwise/ReadVariableOp/model/depthwise_conv2d/depthwise/ReadVariableOp2b
/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp2f
1model/depthwise_conv2d_1/depthwise/ReadVariableOp1model/depthwise_conv2d_1/depthwise/ReadVariableOp2\
,model_1/batch_normalization_3/AssignNewValue,model_1/batch_normalization_3/AssignNewValue2`
.model_1/batch_normalization_3/AssignNewValue_1.model_1/batch_normalization_3/AssignNewValue_12~
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_3/ReadVariableOp,model_1/batch_normalization_3/ReadVariableOp2`
.model_1/batch_normalization_3/ReadVariableOp_1.model_1/batch_normalization_3/ReadVariableOp_12\
,model_1/batch_normalization_4/AssignNewValue,model_1/batch_normalization_4/AssignNewValue2`
.model_1/batch_normalization_4/AssignNewValue_1.model_1/batch_normalization_4/AssignNewValue_12~
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_4/ReadVariableOp,model_1/batch_normalization_4/ReadVariableOp2`
.model_1/batch_normalization_4/ReadVariableOp_1.model_1/batch_normalization_4/ReadVariableOp_12\
,model_1/batch_normalization_5/AssignNewValue,model_1/batch_normalization_5/AssignNewValue2`
.model_1/batch_normalization_5/AssignNewValue_1.model_1/batch_normalization_5/AssignNewValue_12~
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_5/ReadVariableOp,model_1/batch_normalization_5/ReadVariableOp2`
.model_1/batch_normalization_5/ReadVariableOp_1.model_1/batch_normalization_5/ReadVariableOp_12R
'model_1/conv2d_2/BiasAdd/ReadVariableOp'model_1/conv2d_2/BiasAdd/ReadVariableOp2P
&model_1/conv2d_2/Conv2D/ReadVariableOp&model_1/conv2d_2/Conv2D/ReadVariableOp2R
'model_1/conv2d_3/BiasAdd/ReadVariableOp'model_1/conv2d_3/BiasAdd/ReadVariableOp2P
&model_1/conv2d_3/Conv2D/ReadVariableOp&model_1/conv2d_3/Conv2D/ReadVariableOp2f
1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp2j
3model_1/depthwise_conv2d_2/depthwise/ReadVariableOp3model_1/depthwise_conv2d_2/depthwise/ReadVariableOp2f
1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp2j
3model_1/depthwise_conv2d_3/depthwise/ReadVariableOp3model_1/depthwise_conv2d_3/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_511314336

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113101982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_6_layer_call_fn_511314170

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5113095562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_re_lu_3_layer_call_and_return_conditional_losses_511310195

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315425

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?C
?

F__inference_model_1_layer_call_and_return_conditional_losses_511311133

inputs,
conv2d_2_511311037:@  
conv2d_2_511311039: -
batch_normalization_3_511311042: -
batch_normalization_3_511311044: -
batch_normalization_3_511311046: -
batch_normalization_3_511311048: 6
depthwise_conv2d_2_511311058: *
depthwise_conv2d_2_511311060: -
batch_normalization_4_511311063: -
batch_normalization_4_511311065: -
batch_normalization_4_511311067: -
batch_normalization_4_511311069: 6
depthwise_conv2d_3_511311079: *
depthwise_conv2d_3_511311081: ,
conv2d_3_511311111:`@ 
conv2d_3_511311113:@-
batch_normalization_5_511311116:@-
batch_normalization_5_511311118:@-
batch_normalization_5_511311120:@-
batch_normalization_5_511311122:@
identity??-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?*depthwise_conv2d_2/StatefulPartitionedCall?*depthwise_conv2d_3/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_511311037conv2d_2_511311039*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_5113110362"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_511311042batch_normalization_3_511311044batch_normalization_3_511311046batch_normalization_3_511311048*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5113106172/
-batch_normalization_3/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_5113110562
re_lu_4/PartitionedCall?
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0depthwise_conv2d_2_511311058depthwise_conv2d_2_511311060*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_5113107342,
*depthwise_conv2d_2/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_4_511311063batch_normalization_4_511311065batch_normalization_4_511311067batch_normalization_4_511311069*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5113107662/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_5113110772
re_lu_5/PartitionedCall?
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0depthwise_conv2d_3_511311079depthwise_conv2d_3_511311081*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_5113108832,
*depthwise_conv2d_3/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_5113110892
re_lu_6/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_concatenate_1_layer_call_and_return_conditional_losses_5113110982
concatenate_1/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_3_511311111conv2d_3_511311113*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_5113111102"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_5_511311116batch_normalization_5_511311118batch_normalization_5_511311120batch_normalization_5_511311122*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5113109152/
-batch_normalization_5/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_5113111302
re_lu_7/PartitionedCall?
IdentityIdentity re_lu_7/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_511311977

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????X@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????X@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_layer_call_fn_511315522

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_5113096822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_511311685

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
,__inference_conv2d_3_layer_call_fn_511315962

inputs!
unknown:`@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_5113111102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????`: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_511316034

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511309556

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511311596

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314281

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????\ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
??
?4
F__inference_model_2_layer_call_and_return_conditional_losses_511313921

inputsA
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: E
+model_conv2d_conv2d_readvariableop_resource:  :
,model_conv2d_biasadd_readvariableop_resource: ?
1model_batch_normalization_readvariableop_resource: A
3model_batch_normalization_readvariableop_1_resource: P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource: R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: R
8model_depthwise_conv2d_depthwise_readvariableop_resource: D
6model_depthwise_conv2d_biasadd_readvariableop_resource: A
3model_batch_normalization_1_readvariableop_resource: C
5model_batch_normalization_1_readvariableop_1_resource: R
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: T
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: T
:model_depthwise_conv2d_1_depthwise_readvariableop_resource: F
8model_depthwise_conv2d_1_biasadd_readvariableop_resource: G
-model_conv2d_1_conv2d_readvariableop_resource:@@<
.model_conv2d_1_biasadd_readvariableop_resource:@A
3model_batch_normalization_2_readvariableop_resource:@C
5model_batch_normalization_2_readvariableop_1_resource:@R
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@I
/model_1_conv2d_2_conv2d_readvariableop_resource:@ >
0model_1_conv2d_2_biasadd_readvariableop_resource: C
5model_1_batch_normalization_3_readvariableop_resource: E
7model_1_batch_normalization_3_readvariableop_1_resource: T
Fmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: V
Hmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: V
<model_1_depthwise_conv2d_2_depthwise_readvariableop_resource: H
:model_1_depthwise_conv2d_2_biasadd_readvariableop_resource: C
5model_1_batch_normalization_4_readvariableop_resource: E
7model_1_batch_normalization_4_readvariableop_1_resource: T
Fmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: V
Hmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: V
<model_1_depthwise_conv2d_3_depthwise_readvariableop_resource: H
:model_1_depthwise_conv2d_3_biasadd_readvariableop_resource: I
/model_1_conv2d_3_conv2d_readvariableop_resource:`@>
0model_1_conv2d_3_biasadd_readvariableop_resource:@C
5model_1_batch_normalization_5_readvariableop_resource:@E
7model_1_batch_normalization_5_readvariableop_1_resource:@T
Fmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_7_readvariableop_resource:@=
/batch_normalization_7_readvariableop_1_resource:@L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@B
(container_conv2d_readvariableop_resource:@B7
)container_biasadd_readvariableop_resource:B
identity??5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1? container/BiasAdd/ReadVariableOp?container/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_1/ReadVariableOp?,model/batch_normalization_1/ReadVariableOp_1?;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_2/ReadVariableOp?,model/batch_normalization_2/ReadVariableOp_1?#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?-model/depthwise_conv2d/BiasAdd/ReadVariableOp?/model/depthwise_conv2d/depthwise/ReadVariableOp?/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp?1model/depthwise_conv2d_1/depthwise/ReadVariableOp?=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_3/ReadVariableOp?.model_1/batch_normalization_3/ReadVariableOp_1?=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_4/ReadVariableOp?.model_1/batch_normalization_4/ReadVariableOp_1?=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_5/ReadVariableOp?.model_1/batch_normalization_5/ReadVariableOp_1?'model_1/conv2d_2/BiasAdd/ReadVariableOp?&model_1/conv2d_2/Conv2D/ReadVariableOp?'model_1/conv2d_3/BiasAdd/ReadVariableOp?&model_1/conv2d_3/Conv2D/ReadVariableOp?1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp?3model_1/depthwise_conv2d_2/depthwise/ReadVariableOp?1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp?3model_1/depthwise_conv2d_3/depthwise/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
conv2d_4/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
re_lu_8/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_8/Relu?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dre_lu_8/Relu:activations:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
model/conv2d/BiasAdd?
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02*
(model/batch_normalization/ReadVariableOp?
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization/ReadVariableOp_1?
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02;
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/BiasAdd:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2,
*model/batch_normalization/FusedBatchNormV3?
model/re_lu/ReluRelu.model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
model/re_lu/Relu?
/model/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp8model_depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype021
/model/depthwise_conv2d/depthwise/ReadVariableOp?
&model/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&model/depthwise_conv2d/depthwise/Shape?
.model/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.model/depthwise_conv2d/depthwise/dilation_rate?
 model/depthwise_conv2d/depthwiseDepthwiseConv2dNativemodel/re_lu/Relu:activations:07model/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2"
 model/depthwise_conv2d/depthwise?
-model/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp6model_depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/depthwise_conv2d/BiasAdd/ReadVariableOp?
model/depthwise_conv2d/BiasAddBiasAdd)model/depthwise_conv2d/depthwise:output:05model/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2 
model/depthwise_conv2d/BiasAdd?
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization_1/ReadVariableOp?
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,model/batch_normalization_1/ReadVariableOp_1?
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3'model/depthwise_conv2d/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_1/FusedBatchNormV3?
model/re_lu_1/ReluRelu0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
model/re_lu_1/Relu?
1model/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp:model_depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype023
1model/depthwise_conv2d_1/depthwise/ReadVariableOp?
(model/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(model/depthwise_conv2d_1/depthwise/Shape?
0model/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      22
0model/depthwise_conv2d_1/depthwise/dilation_rate?
"model/depthwise_conv2d_1/depthwiseDepthwiseConv2dNative model/re_lu_1/Relu:activations:09model/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2$
"model/depthwise_conv2d_1/depthwise?
/model/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp8model_depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp?
 model/depthwise_conv2d_1/BiasAddBiasAdd+model/depthwise_conv2d_1/depthwise:output:07model/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2"
 model/depthwise_conv2d_1/BiasAdd?
model/re_lu_2/ReluRelu)model/depthwise_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\ 2
model/re_lu_2/Relu?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2 model/re_lu_2/Relu:activations:0re_lu_8/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????\@2
model/concatenate/concat?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2D!model/concatenate/concat:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@2
model/conv2d_1/BiasAdd?
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_2/ReadVariableOp?
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_2/ReadVariableOp_1?
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3model/conv2d_1/BiasAdd:output:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z@:@:@:@:@:*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_2/FusedBatchNormV3?
model/re_lu_3/ReluRelu0model/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z@2
model/re_lu_3/Relu?
&model_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02(
&model_1/conv2d_2/Conv2D/ReadVariableOp?
model_1/conv2d_2/Conv2DConv2D model/re_lu_3/Relu:activations:0.model_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
model_1/conv2d_2/Conv2D?
'model_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv2d_2/BiasAdd/ReadVariableOp?
model_1/conv2d_2/BiasAddBiasAdd model_1/conv2d_2/Conv2D:output:0/model_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
model_1/conv2d_2/BiasAdd?
,model_1/batch_normalization_3/ReadVariableOpReadVariableOp5model_1_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_3/ReadVariableOp?
.model_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_3/ReadVariableOp_1?
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3!model_1/conv2d_2/BiasAdd:output:04model_1/batch_normalization_3/ReadVariableOp:value:06model_1/batch_normalization_3/ReadVariableOp_1:value:0Emodel_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
is_training( 20
.model_1/batch_normalization_3/FusedBatchNormV3?
model_1/re_lu_4/ReluRelu2model_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
model_1/re_lu_4/Relu?
3model_1/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp<model_1_depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype025
3model_1/depthwise_conv2d_2/depthwise/ReadVariableOp?
*model_1/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*model_1/depthwise_conv2d_2/depthwise/Shape?
2model_1/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2model_1/depthwise_conv2d_2/depthwise/dilation_rate?
$model_1/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative"model_1/re_lu_4/Relu:activations:0;model_1/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2&
$model_1/depthwise_conv2d_2/depthwise?
1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:model_1_depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp?
"model_1/depthwise_conv2d_2/BiasAddBiasAdd-model_1/depthwise_conv2d_2/depthwise:output:09model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2$
"model_1/depthwise_conv2d_2/BiasAdd?
,model_1/batch_normalization_4/ReadVariableOpReadVariableOp5model_1_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_4/ReadVariableOp?
.model_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_4/ReadVariableOp_1?
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+model_1/depthwise_conv2d_2/BiasAdd:output:04model_1/batch_normalization_4/ReadVariableOp:value:06model_1/batch_normalization_4/ReadVariableOp_1:value:0Emodel_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
is_training( 20
.model_1/batch_normalization_4/FusedBatchNormV3?
model_1/re_lu_5/ReluRelu2model_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
model_1/re_lu_5/Relu?
3model_1/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp<model_1_depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype025
3model_1/depthwise_conv2d_3/depthwise/ReadVariableOp?
*model_1/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*model_1/depthwise_conv2d_3/depthwise/Shape?
2model_1/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2model_1/depthwise_conv2d_3/depthwise/dilation_rate?
$model_1/depthwise_conv2d_3/depthwiseDepthwiseConv2dNative"model_1/re_lu_5/Relu:activations:0;model_1/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2&
$model_1/depthwise_conv2d_3/depthwise?
1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:model_1_depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp?
"model_1/depthwise_conv2d_3/BiasAddBiasAdd-model_1/depthwise_conv2d_3/depthwise:output:09model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2$
"model_1/depthwise_conv2d_3/BiasAdd?
model_1/re_lu_6/ReluRelu+model_1/depthwise_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Z 2
model_1/re_lu_6/Relu?
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axis?
model_1/concatenate_1/concatConcatV2"model_1/re_lu_6/Relu:activations:0 model/re_lu_3/Relu:activations:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????Z`2
model_1/concatenate_1/concat?
&model_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02(
&model_1/conv2d_3/Conv2D/ReadVariableOp?
model_1/conv2d_3/Conv2DConv2D%model_1/concatenate_1/concat:output:0.model_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@*
paddingVALID*
strides
2
model_1/conv2d_3/Conv2D?
'model_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv2d_3/BiasAdd/ReadVariableOp?
model_1/conv2d_3/BiasAddBiasAdd model_1/conv2d_3/Conv2D:output:0/model_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@2
model_1/conv2d_3/BiasAdd?
,model_1/batch_normalization_5/ReadVariableOpReadVariableOp5model_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model_1/batch_normalization_5/ReadVariableOp?
.model_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.model_1/batch_normalization_5/ReadVariableOp_1?
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02?
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02A
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3!model_1/conv2d_3/BiasAdd:output:04model_1/batch_normalization_5/ReadVariableOp:value:06model_1/batch_normalization_5/ReadVariableOp_1:value:0Emodel_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????X@:@:@:@:@:*
epsilon%o?:*
is_training( 20
.model_1/batch_normalization_5/FusedBatchNormV3?
model_1/re_lu_7/ReluRelu2model_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????X@2
model_1/re_lu_7/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D"model_1/re_lu_7/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@*
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@2
conv2d_5/BiasAdd?
dropout/IdentityIdentityconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????+@2
dropout/Identity?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3dropout/Identity:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????+@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
re_lu_9/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????+@2
re_lu_9/Relu?
average_pooling2d/AvgPoolAvgPoolre_lu_9/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool?
container/Conv2D/ReadVariableOpReadVariableOp(container_conv2d_readvariableop_resource*&
_output_shapes
:@B*
dtype02!
container/Conv2D/ReadVariableOp?
container/Conv2DConv2D"average_pooling2d/AvgPool:output:0'container/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B*
paddingVALID*
strides
2
container/Conv2D?
 container/BiasAdd/ReadVariableOpReadVariableOp)container_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype02"
 container/BiasAdd/ReadVariableOp?
container/BiasAddBiasAddcontainer/Conv2D:output:0(container/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????B2
container/BiasAdd?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transposecontainer/BiasAdd:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????B2"
 tf.compat.v1.transpose/transpose?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean$tf.compat.v1.transpose/transpose:y:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????B2
tf.math.reduce_mean/Mean?
IdentityIdentity!tf.math.reduce_mean/Mean:output:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^container/BiasAdd/ReadVariableOp ^container/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp.^model/depthwise_conv2d/BiasAdd/ReadVariableOp0^model/depthwise_conv2d/depthwise/ReadVariableOp0^model/depthwise_conv2d_1/BiasAdd/ReadVariableOp2^model/depthwise_conv2d_1/depthwise/ReadVariableOp>^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_3/ReadVariableOp/^model_1/batch_normalization_3/ReadVariableOp_1>^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_4/ReadVariableOp/^model_1/batch_normalization_4/ReadVariableOp_1>^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_5/ReadVariableOp/^model_1/batch_normalization_5/ReadVariableOp_1(^model_1/conv2d_2/BiasAdd/ReadVariableOp'^model_1/conv2d_2/Conv2D/ReadVariableOp(^model_1/conv2d_3/BiasAdd/ReadVariableOp'^model_1/conv2d_3/Conv2D/ReadVariableOp2^model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp4^model_1/depthwise_conv2d_2/depthwise/ReadVariableOp2^model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp4^model_1/depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 container/BiasAdd/ReadVariableOp container/BiasAdd/ReadVariableOp2B
container/Conv2D/ReadVariableOpcontainer/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2^
-model/depthwise_conv2d/BiasAdd/ReadVariableOp-model/depthwise_conv2d/BiasAdd/ReadVariableOp2b
/model/depthwise_conv2d/depthwise/ReadVariableOp/model/depthwise_conv2d/depthwise/ReadVariableOp2b
/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp/model/depthwise_conv2d_1/BiasAdd/ReadVariableOp2f
1model/depthwise_conv2d_1/depthwise/ReadVariableOp1model/depthwise_conv2d_1/depthwise/ReadVariableOp2~
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_3/ReadVariableOp,model_1/batch_normalization_3/ReadVariableOp2`
.model_1/batch_normalization_3/ReadVariableOp_1.model_1/batch_normalization_3/ReadVariableOp_12~
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_4/ReadVariableOp,model_1/batch_normalization_4/ReadVariableOp2`
.model_1/batch_normalization_4/ReadVariableOp_1.model_1/batch_normalization_4/ReadVariableOp_12~
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_5/ReadVariableOp,model_1/batch_normalization_5/ReadVariableOp2`
.model_1/batch_normalization_5/ReadVariableOp_1.model_1/batch_normalization_5/ReadVariableOp_12R
'model_1/conv2d_2/BiasAdd/ReadVariableOp'model_1/conv2d_2/BiasAdd/ReadVariableOp2P
&model_1/conv2d_2/Conv2D/ReadVariableOp&model_1/conv2d_2/Conv2D/ReadVariableOp2R
'model_1/conv2d_3/BiasAdd/ReadVariableOp'model_1/conv2d_3/BiasAdd/ReadVariableOp2P
&model_1/conv2d_3/Conv2D/ReadVariableOp&model_1/conv2d_3/Conv2D/ReadVariableOp2f
1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp1model_1/depthwise_conv2d_2/BiasAdd/ReadVariableOp2j
3model_1/depthwise_conv2d_2/depthwise/ReadVariableOp3model_1/depthwise_conv2d_2/depthwise/ReadVariableOp2f
1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp1model_1/depthwise_conv2d_3/BiasAdd/ReadVariableOp2j
3model_1/depthwise_conv2d_3/depthwise/ReadVariableOp3model_1/depthwise_conv2d_3/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
t
J__inference_concatenate_layer_call_and_return_conditional_losses_511310163

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????@2
concat}
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+??????????????????????????? :+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_re_lu_9_layer_call_and_return_conditional_losses_511315471

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????+@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
`
D__inference_re_lu_layer_call_and_return_conditional_losses_511315581

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
G
+__inference_re_lu_9_layer_call_fn_511315466

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_9_layer_call_and_return_conditional_losses_5113120222
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
`
D__inference_re_lu_layer_call_and_return_conditional_losses_511310121

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
x
L__inference_concatenate_1_layer_call_and_return_conditional_losses_511315953
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????`2
concat}
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+??????????????????????????? :+???????????????????????????@:k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/1
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_511309980

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?i
?
"__inference__traced_save_511316229
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop/
+savev2_container_kernel_read_readvariableop-
)savev2_container_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_depthwise_conv2d_depthwise_kernel_read_readvariableop4
0savev2_depthwise_conv2d_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_1_depthwise_kernel_read_readvariableop6
2savev2_depthwise_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_2_depthwise_kernel_read_readvariableop6
2savev2_depthwise_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_3_depthwise_kernel_read_readvariableop6
2savev2_depthwise_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop+savev2_container_kernel_read_readvariableop)savev2_container_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_depthwise_conv2d_depthwise_kernel_read_readvariableop0savev2_depthwise_conv2d_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop>savev2_depthwise_conv2d_1_depthwise_kernel_read_readvariableop2savev2_depthwise_conv2d_1_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop>savev2_depthwise_conv2d_2_depthwise_kernel_read_readvariableop2savev2_depthwise_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop>savev2_depthwise_conv2d_3_depthwise_kernel_read_readvariableop2savev2_depthwise_conv2d_3_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
9272
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : :@@:@:@:@:@:@:@B:B:  : : : : : : : : : : : : : :@@:@:@:@:@:@:@ : : : : : : : : : : : : : :`@:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:@@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@B: 

_output_shapes
:B:,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@:,#(
&
_output_shapes
:@ : $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: :,)(
&
_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: : ,

_output_shapes
: : -

_output_shapes
: : .

_output_shapes
: :,/(
&
_output_shapes
: : 0

_output_shapes
: :,1(
&
_output_shapes
:`@: 2

_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@:7

_output_shapes
: 
?
?
)__inference_model_layer_call_fn_511310483
input_1!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113103952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+??????????????????????????? 
!
_user_specified_name	input_1
?
?
9__inference_batch_normalization_2_layer_call_fn_511315721

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5113100242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_511314157

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?B
?

D__inference_model_layer_call_and_return_conditional_losses_511310395

inputs*
conv2d_511310342:  
conv2d_511310344: +
batch_normalization_511310347: +
batch_normalization_511310349: +
batch_normalization_511310351: +
batch_normalization_511310353: 4
depthwise_conv2d_511310357: (
depthwise_conv2d_511310359: -
batch_normalization_1_511310362: -
batch_normalization_1_511310364: -
batch_normalization_1_511310366: -
batch_normalization_1_511310368: 6
depthwise_conv2d_1_511310372: *
depthwise_conv2d_1_511310374: ,
conv2d_1_511310379:@@ 
conv2d_1_511310381:@-
batch_normalization_2_511310384:@-
batch_normalization_2_511310386:@-
batch_normalization_2_511310388:@-
batch_normalization_2_511310390:@
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(depthwise_conv2d/StatefulPartitionedCall?*depthwise_conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_511310342conv2d_511310344*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_5113101012 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_511310347batch_normalization_511310349batch_normalization_511310351batch_normalization_511310353*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_5113097262-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_5113101212
re_lu/PartitionedCall?
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0depthwise_conv2d_511310357depthwise_conv2d_511310359*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_5113097992*
(depthwise_conv2d/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:0batch_normalization_1_511310362batch_normalization_1_511310364batch_normalization_1_511310366batch_normalization_1_511310368*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5113098752/
-batch_normalization_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_5113101422
re_lu_1/PartitionedCall?
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0depthwise_conv2d_1_511310372depthwise_conv2d_1_511310374*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_5113099482,
*depthwise_conv2d_1/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_5113101542
re_lu_2/PartitionedCall?
concatenate/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_5113101632
concatenate/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_1_511310379conv2d_1_511310381*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_5113101752"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_511310384batch_normalization_2_511310386batch_normalization_2_511310388batch_normalization_2_511310390*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5113100242/
-batch_normalization_2/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_5113101952
re_lu_3/PartitionedCall?
IdentityIdentity re_lu_3/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
[
/__inference_concatenate_layer_call_fn_511315669
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_5113101632
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+??????????????????????????? :+??????????????????????????? :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/1
?
?
Q__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_511310734

inputs;
!depthwise_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
	depthwise?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
+__inference_model_1_layer_call_fn_511314881

inputs!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:`@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113113302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_model_1_layer_call_fn_511311418
input_2!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:`@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113113302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????@
!
_user_specified_name	input_2
?
?
-__inference_container_layer_call_fn_511315480

inputs!
unknown:@B
	unknown_0:B
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_container_layer_call_and_return_conditional_losses_5113120352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_2_layer_call_fn_511315708

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5113099802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_model_2_layer_call_fn_511313598

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: $

unknown_37: 

unknown_38: $

unknown_39:`@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45:@@

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@B

unknown_52:B
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_5113120462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
,__inference_conv2d_5_layer_call_fn_511315300

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_5113119772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????X@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????X@
 
_user_specified_nameinputs
?

?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_511315310

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????+@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????X@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????X@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_1_layer_call_and_return_conditional_losses_511315653

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511311708

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????\ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_511315643

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_4_layer_call_fn_511315871

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5113107662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?C
?

F__inference_model_1_layer_call_and_return_conditional_losses_511311330

inputs,
conv2d_2_511311277:@  
conv2d_2_511311279: -
batch_normalization_3_511311282: -
batch_normalization_3_511311284: -
batch_normalization_3_511311286: -
batch_normalization_3_511311288: 6
depthwise_conv2d_2_511311292: *
depthwise_conv2d_2_511311294: -
batch_normalization_4_511311297: -
batch_normalization_4_511311299: -
batch_normalization_4_511311301: -
batch_normalization_4_511311303: 6
depthwise_conv2d_3_511311307: *
depthwise_conv2d_3_511311309: ,
conv2d_3_511311314:`@ 
conv2d_3_511311316:@-
batch_normalization_5_511311319:@-
batch_normalization_5_511311321:@-
batch_normalization_5_511311323:@-
batch_normalization_5_511311325:@
identity??-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?*depthwise_conv2d_2/StatefulPartitionedCall?*depthwise_conv2d_3/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_511311277conv2d_2_511311279*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_5113110362"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_511311282batch_normalization_3_511311284batch_normalization_3_511311286batch_normalization_3_511311288*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5113106612/
-batch_normalization_3/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_5113110562
re_lu_4/PartitionedCall?
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0depthwise_conv2d_2_511311292depthwise_conv2d_2_511311294*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_5113107342,
*depthwise_conv2d_2/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_4_511311297batch_normalization_4_511311299batch_normalization_4_511311301batch_normalization_4_511311303*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5113108102/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_5113110772
re_lu_5/PartitionedCall?
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0depthwise_conv2d_3_511311307depthwise_conv2d_3_511311309*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_5113108832,
*depthwise_conv2d_3/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_5113110892
re_lu_6/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_concatenate_1_layer_call_and_return_conditional_losses_5113110982
concatenate_1/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_3_511311314conv2d_3_511311316*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_5113111102"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_5_511311319batch_normalization_5_511311321batch_normalization_5_511311323batch_normalization_5_511311325*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5113109592/
-batch_normalization_5/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_5113111302
re_lu_7/PartitionedCall?
IdentityIdentity re_lu_7/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_511316016

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_511315739

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
4__inference_depthwise_conv2d_layer_call_fn_511309809

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_5113097992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?P
?
F__inference_model_2_layer_call_and_return_conditional_losses_511312892

inputs,
conv2d_4_511312768:  
conv2d_4_511312770: -
batch_normalization_6_511312773: -
batch_normalization_6_511312775: -
batch_normalization_6_511312777: -
batch_normalization_6_511312779: )
model_511312783:  
model_511312785: 
model_511312787: 
model_511312789: 
model_511312791: 
model_511312793: )
model_511312795: 
model_511312797: 
model_511312799: 
model_511312801: 
model_511312803: 
model_511312805: )
model_511312807: 
model_511312809: )
model_511312811:@@
model_511312813:@
model_511312815:@
model_511312817:@
model_511312819:@
model_511312821:@+
model_1_511312824:@ 
model_1_511312826: 
model_1_511312828: 
model_1_511312830: 
model_1_511312832: 
model_1_511312834: +
model_1_511312836: 
model_1_511312838: 
model_1_511312840: 
model_1_511312842: 
model_1_511312844: 
model_1_511312846: +
model_1_511312848: 
model_1_511312850: +
model_1_511312852:`@
model_1_511312854:@
model_1_511312856:@
model_1_511312858:@
model_1_511312860:@
model_1_511312862:@,
conv2d_5_511312865:@@ 
conv2d_5_511312867:@-
batch_normalization_7_511312871:@-
batch_normalization_7_511312873:@-
batch_normalization_7_511312875:@-
batch_normalization_7_511312877:@-
container_511312882:@B!
container_511312884:B
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?!container/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dropout/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_511312768conv2d_4_511312770*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_5113116852"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_6_511312773batch_normalization_6_511312775batch_normalization_6_511312777batch_normalization_6_511312779*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5113126292/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_5113117232
re_lu_8/PartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0model_511312783model_511312785model_511312787model_511312789model_511312791model_511312793model_511312795model_511312797model_511312799model_511312801model_511312803model_511312805model_511312807model_511312809model_511312811model_511312813model_511312815model_511312817model_511312819model_511312821* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????Z@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113125472
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0model_1_511312824model_1_511312826model_1_511312828model_1_511312830model_1_511312832model_1_511312834model_1_511312836model_1_511312838model_1_511312840model_1_511312842model_1_511312844model_1_511312846model_1_511312848model_1_511312850model_1_511312852model_1_511312854model_1_511312856model_1_511312858model_1_511312860model_1_511312862* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????X@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113123772!
model_1/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(model_1/StatefulPartitionedCall:output:0conv2d_5_511312865conv2d_5_511312867*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_5113119772"
 conv2d_5/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_5113122372!
dropout/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0batch_normalization_7_511312871batch_normalization_7_511312873batch_normalization_7_511312875batch_normalization_7_511312877*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5113122062/
-batch_normalization_7/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_9_layer_call_and_return_conditional_losses_5113120222
re_lu_9/PartitionedCall?
!average_pooling2d/PartitionedCallPartitionedCall re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_5113116622#
!average_pooling2d/PartitionedCall?
!container/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0container_511312882container_511312884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_container_layer_call_and_return_conditional_losses_5113120352#
!container/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose*container/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????B2"
 tf.compat.v1.transpose/transpose?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean$tf.compat.v1.transpose/transpose:y:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????B2
tf.math.reduce_mean/Mean?
IdentityIdentity!tf.math.reduce_mean/Mean:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall"^container/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!container/StatefulPartitionedCall!container/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
d
F__inference_dropout_layer_call_and_return_conditional_losses_511311988

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????+@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????+@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
d
+__inference_dropout_layer_call_fn_511315320

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_5113122372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_7_layer_call_and_return_conditional_losses_511316044

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?p
?
D__inference_model_layer_call_and_return_conditional_losses_511314551

inputs?
%conv2d_conv2d_readvariableop_resource:  4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: L
2depthwise_conv2d_depthwise_readvariableop_resource: >
0depthwise_conv2d_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_1_depthwise_readvariableop_resource: @
2depthwise_conv2d_1_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'depthwise_conv2d/BiasAdd/ReadVariableOp?)depthwise_conv2d/depthwise/ReadVariableOp?)depthwise_conv2d_1/BiasAdd/ReadVariableOp?+depthwise_conv2d_1/depthwise/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

re_lu/Relu?
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp2depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp?
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape?
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate?
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
depthwise_conv2d/depthwise?
'depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp0depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'depthwise_conv2d/BiasAdd/ReadVariableOp?
depthwise_conv2d/BiasAddBiasAdd#depthwise_conv2d/depthwise:output:0/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
depthwise_conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!depthwise_conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_1/Relu?
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp?
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape?
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate?
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_1/Relu:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
depthwise_conv2d_1/depthwise?
)depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_1/BiasAdd/ReadVariableOp?
depthwise_conv2d_1/BiasAddBiasAdd%depthwise_conv2d_1/depthwise:output:01depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
depthwise_conv2d_1/BiasAdd?
re_lu_2/ReluRelu#depthwise_conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_2/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2re_lu_2/Relu:activations:0inputs concatenate/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????@2
concatenate/concat?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconcatenate/concat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
conv2d_1/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
re_lu_3/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
re_lu_3/Relu?
IdentityIdentityre_lu_3/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^depthwise_conv2d/BiasAdd/ReadVariableOp*^depthwise_conv2d/depthwise/ReadVariableOp*^depthwise_conv2d_1/BiasAdd/ReadVariableOp,^depthwise_conv2d_1/depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'depthwise_conv2d/BiasAdd/ReadVariableOp'depthwise_conv2d/BiasAdd/ReadVariableOp2V
)depthwise_conv2d/depthwise/ReadVariableOp)depthwise_conv2d/depthwise/ReadVariableOp2V
)depthwise_conv2d_1/BiasAdd/ReadVariableOp)depthwise_conv2d_1/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_1/depthwise/ReadVariableOp+depthwise_conv2d_1/depthwise/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_layer_call_fn_511315535

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_5113097262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
G
+__inference_re_lu_1_layer_call_fn_511315648

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_5113101422
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
E
)__inference_re_lu_layer_call_fn_511315576

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_5113101212
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_re_lu_9_layer_call_and_return_conditional_losses_511312022

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????+@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_511311036

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_4_layer_call_fn_511314147

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_5113116852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
+__inference_model_2_layer_call_fn_511312157	
input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: $

unknown_37: 

unknown_38: $

unknown_39:`@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45:@@

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@B

unknown_52:B
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_5113120462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????^

_user_specified_nameinput
?
G
+__inference_dropout_layer_call_fn_511315315

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_5113119882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
G
+__inference_re_lu_8_layer_call_fn_511314286

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_5113117232
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????\ :W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_1_layer_call_fn_511315594

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5113098312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
v
J__inference_concatenate_layer_call_and_return_conditional_losses_511315676
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????@2
concat}
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+??????????????????????????? :+??????????????????????????? :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/1
?
b
F__inference_re_lu_5_layer_call_and_return_conditional_losses_511315930

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_511315695

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_7_layer_call_and_return_conditional_losses_511311130

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_511309948

inputs;
!depthwise_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
	depthwise?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?o
?
F__inference_model_1_layer_call_and_return_conditional_losses_511311925

inputsA
'conv2d_2_conv2d_readvariableop_resource:@ 6
(conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_2_depthwise_readvariableop_resource: @
2depthwise_conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_3_depthwise_readvariableop_resource: @
2depthwise_conv2d_3_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:`@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity??5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?)depthwise_conv2d_2/BiasAdd/ReadVariableOp?+depthwise_conv2d_2/depthwise/ReadVariableOp?)depthwise_conv2d_3/BiasAdd/ReadVariableOp?+depthwise_conv2d_3/depthwise/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
conv2d_2/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
re_lu_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_4/Relu?
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp?
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_2/depthwise/Shape?
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate?
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise?
)depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_2/BiasAdd/ReadVariableOp?
depthwise_conv2d_2/BiasAddBiasAdd%depthwise_conv2d_2/depthwise:output:01depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
depthwise_conv2d_2/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d_2/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
re_lu_5/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_5/Relu?
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp?
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_3/depthwise/Shape?
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate?
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_5/Relu:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise?
)depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_3/BiasAdd/ReadVariableOp?
depthwise_conv2d_3/BiasAddBiasAdd%depthwise_conv2d_3/depthwise:output:01depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
depthwise_conv2d_3/BiasAdd?
re_lu_6/ReluRelu#depthwise_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_6/Relux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2re_lu_6/Relu:activations:0inputs"concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????Z`2
concatenate_1/concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcatenate_1/concat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@2
conv2d_3/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????X@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
re_lu_7/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????X@2
re_lu_7/Relu?
IdentityIdentityre_lu_7/Relu:activations:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*^depthwise_conv2d_2/BiasAdd/ReadVariableOp,^depthwise_conv2d_2/depthwise/ReadVariableOp*^depthwise_conv2d_3/BiasAdd/ReadVariableOp,^depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*/
_output_shapes
:?????????X@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????Z@: : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2V
)depthwise_conv2d_2/BiasAdd/ReadVariableOp)depthwise_conv2d_2/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_2/depthwise/ReadVariableOp+depthwise_conv2d_2/depthwise/ReadVariableOp2V
)depthwise_conv2d_3/BiasAdd/ReadVariableOp)depthwise_conv2d_3/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_3/depthwise/ReadVariableOp+depthwise_conv2d_3/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?r
?
F__inference_model_1_layer_call_and_return_conditional_losses_511315051

inputsA
'conv2d_2_conv2d_readvariableop_resource:@ 6
(conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_2_depthwise_readvariableop_resource: @
2depthwise_conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_3_depthwise_readvariableop_resource: @
2depthwise_conv2d_3_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:`@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity??5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?)depthwise_conv2d_2/BiasAdd/ReadVariableOp?+depthwise_conv2d_2/depthwise/ReadVariableOp?)depthwise_conv2d_3/BiasAdd/ReadVariableOp?+depthwise_conv2d_3/depthwise/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
conv2d_2/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
re_lu_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_4/Relu?
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp?
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_2/depthwise/Shape?
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate?
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise?
)depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_2/BiasAdd/ReadVariableOp?
depthwise_conv2d_2/BiasAddBiasAdd%depthwise_conv2d_2/depthwise:output:01depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
depthwise_conv2d_2/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d_2/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
re_lu_5/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_5/Relu?
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp?
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_3/depthwise/Shape?
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate?
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_5/Relu:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise?
)depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_3/BiasAdd/ReadVariableOp?
depthwise_conv2d_3/BiasAddBiasAdd%depthwise_conv2d_3/depthwise:output:01depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
depthwise_conv2d_3/BiasAdd?
re_lu_6/ReluRelu#depthwise_conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_6/Relux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2re_lu_6/Relu:activations:0inputs"concatenate_1/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????`2
concatenate_1/concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcatenate_1/concat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
conv2d_3/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
re_lu_7/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
re_lu_7/Relu?
IdentityIdentityre_lu_7/Relu:activations:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*^depthwise_conv2d_2/BiasAdd/ReadVariableOp,^depthwise_conv2d_2/depthwise/ReadVariableOp*^depthwise_conv2d_3/BiasAdd/ReadVariableOp,^depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2V
)depthwise_conv2d_2/BiasAdd/ReadVariableOp)depthwise_conv2d_2/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_2/depthwise/ReadVariableOp+depthwise_conv2d_2/depthwise/ReadVariableOp2V
)depthwise_conv2d_3/BiasAdd/ReadVariableOp)depthwise_conv2d_3/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_3/depthwise/ReadVariableOp+depthwise_conv2d_3/depthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_511310766

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_511310883

inputs;
!depthwise_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
	depthwise?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_511309875

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_depthwise_conv2d_3_layer_call_fn_511310893

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_5113108832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?C
?

F__inference_model_1_layer_call_and_return_conditional_losses_511311474
input_2,
conv2d_2_511311421:@  
conv2d_2_511311423: -
batch_normalization_3_511311426: -
batch_normalization_3_511311428: -
batch_normalization_3_511311430: -
batch_normalization_3_511311432: 6
depthwise_conv2d_2_511311436: *
depthwise_conv2d_2_511311438: -
batch_normalization_4_511311441: -
batch_normalization_4_511311443: -
batch_normalization_4_511311445: -
batch_normalization_4_511311447: 6
depthwise_conv2d_3_511311451: *
depthwise_conv2d_3_511311453: ,
conv2d_3_511311458:`@ 
conv2d_3_511311460:@-
batch_normalization_5_511311463:@-
batch_normalization_5_511311465:@-
batch_normalization_5_511311467:@-
batch_normalization_5_511311469:@
identity??-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?*depthwise_conv2d_2/StatefulPartitionedCall?*depthwise_conv2d_3/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_511311421conv2d_2_511311423*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_5113110362"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_511311426batch_normalization_3_511311428batch_normalization_3_511311430batch_normalization_3_511311432*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5113106172/
-batch_normalization_3/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_5113110562
re_lu_4/PartitionedCall?
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0depthwise_conv2d_2_511311436depthwise_conv2d_2_511311438*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_5113107342,
*depthwise_conv2d_2/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_4_511311441batch_normalization_4_511311443batch_normalization_4_511311445batch_normalization_4_511311447*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5113107662/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_5113110772
re_lu_5/PartitionedCall?
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0depthwise_conv2d_3_511311451depthwise_conv2d_3_511311453*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_5113108832,
*depthwise_conv2d_3/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_5113110892
re_lu_6/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_concatenate_1_layer_call_and_return_conditional_losses_5113110982
concatenate_1/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_3_511311458conv2d_3_511311460*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_5113111102"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_5_511311463batch_normalization_5_511311465batch_normalization_5_511311467batch_normalization_5_511311469*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5113109152/
-batch_normalization_5/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_5113111302
re_lu_7/PartitionedCall?
IdentityIdentity re_lu_7/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????@
!
_user_specified_name	input_2
?B
?

D__inference_model_layer_call_and_return_conditional_losses_511310539
input_1*
conv2d_511310486:  
conv2d_511310488: +
batch_normalization_511310491: +
batch_normalization_511310493: +
batch_normalization_511310495: +
batch_normalization_511310497: 4
depthwise_conv2d_511310501: (
depthwise_conv2d_511310503: -
batch_normalization_1_511310506: -
batch_normalization_1_511310508: -
batch_normalization_1_511310510: -
batch_normalization_1_511310512: 6
depthwise_conv2d_1_511310516: *
depthwise_conv2d_1_511310518: ,
conv2d_1_511310523:@@ 
conv2d_1_511310525:@-
batch_normalization_2_511310528:@-
batch_normalization_2_511310530:@-
batch_normalization_2_511310532:@-
batch_normalization_2_511310534:@
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(depthwise_conv2d/StatefulPartitionedCall?*depthwise_conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_511310486conv2d_511310488*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_5113101012 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_511310491batch_normalization_511310493batch_normalization_511310495batch_normalization_511310497*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_5113096822-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_5113101212
re_lu/PartitionedCall?
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0depthwise_conv2d_511310501depthwise_conv2d_511310503*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_5113097992*
(depthwise_conv2d/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:0batch_normalization_1_511310506batch_normalization_1_511310508batch_normalization_1_511310510batch_normalization_1_511310512*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5113098312/
-batch_normalization_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_5113101422
re_lu_1/PartitionedCall?
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0depthwise_conv2d_1_511310516depthwise_conv2d_1_511310518*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_5113099482,
*depthwise_conv2d_1/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_5113101542
re_lu_2/PartitionedCall?
concatenate/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_5113101632
concatenate/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_1_511310523conv2d_1_511310525*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_5113101752"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_511310528batch_normalization_2_511310530batch_normalization_2_511310532batch_normalization_2_511310534*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5113099802/
-batch_normalization_2/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_5113101952
re_lu_3/PartitionedCall?
IdentityIdentity re_lu_3/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall:j f
A
_output_shapes/
-:+??????????????????????????? 
!
_user_specified_name	input_1
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511312007

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????+@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_4_layer_call_and_return_conditional_losses_511311056

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
+__inference_model_2_layer_call_fn_511313116	
input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: $

unknown_37: 

unknown_38: $

unknown_39:`@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45:@@

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@B

unknown_52:B
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B*H
_read_only_resource_inputs*
(&	
!"#$'()*+,/01256*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_5113128922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????^

_user_specified_nameinput
??
?
F__inference_model_1_layer_call_and_return_conditional_losses_511312377

inputsA
'conv2d_2_conv2d_readvariableop_resource:@ 6
(conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_2_depthwise_readvariableop_resource: @
2depthwise_conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_3_depthwise_readvariableop_resource: @
2depthwise_conv2d_3_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:`@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity??$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?)depthwise_conv2d_2/BiasAdd/ReadVariableOp?+depthwise_conv2d_2/depthwise/ReadVariableOp?)depthwise_conv2d_3/BiasAdd/ReadVariableOp?+depthwise_conv2d_3/depthwise/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
conv2d_2/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
re_lu_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_4/Relu?
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp?
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_2/depthwise/Shape?
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate?
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise?
)depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_2/BiasAdd/ReadVariableOp?
depthwise_conv2d_2/BiasAddBiasAdd%depthwise_conv2d_2/depthwise:output:01depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
depthwise_conv2d_2/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d_2/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
re_lu_5/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_5/Relu?
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp?
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_3/depthwise/Shape?
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate?
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_5/Relu:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise?
)depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_3/BiasAdd/ReadVariableOp?
depthwise_conv2d_3/BiasAddBiasAdd%depthwise_conv2d_3/depthwise:output:01depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
depthwise_conv2d_3/BiasAdd?
re_lu_6/ReluRelu#depthwise_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_6/Relux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2re_lu_6/Relu:activations:0inputs"concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????Z`2
concatenate_1/concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcatenate_1/concat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@2
conv2d_3/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????X@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
re_lu_7/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????X@2
re_lu_7/Relu?	
IdentityIdentityre_lu_7/Relu:activations:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*^depthwise_conv2d_2/BiasAdd/ReadVariableOp,^depthwise_conv2d_2/depthwise/ReadVariableOp*^depthwise_conv2d_3/BiasAdd/ReadVariableOp,^depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*/
_output_shapes
:?????????X@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????Z@: : : : : : : : : : : : : : : : : : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2V
)depthwise_conv2d_2/BiasAdd/ReadVariableOp)depthwise_conv2d_2/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_2/depthwise/ReadVariableOp+depthwise_conv2d_2/depthwise/ReadVariableOp2V
)depthwise_conv2d_3/BiasAdd/ReadVariableOp)depthwise_conv2d_3/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_3/depthwise/ReadVariableOp+depthwise_conv2d_3/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_6_layer_call_fn_511314209

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5113126292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????\ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_511315757

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315443

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????+@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_511309726

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?o
?
F__inference_model_1_layer_call_and_return_conditional_losses_511315211

inputsA
'conv2d_2_conv2d_readvariableop_resource:@ 6
(conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_2_depthwise_readvariableop_resource: @
2depthwise_conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_3_depthwise_readvariableop_resource: @
2depthwise_conv2d_3_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:`@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity??5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?)depthwise_conv2d_2/BiasAdd/ReadVariableOp?+depthwise_conv2d_2/depthwise/ReadVariableOp?)depthwise_conv2d_3/BiasAdd/ReadVariableOp?+depthwise_conv2d_3/depthwise/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
conv2d_2/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
re_lu_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_4/Relu?
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp?
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_2/depthwise/Shape?
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate?
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise?
)depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_2/BiasAdd/ReadVariableOp?
depthwise_conv2d_2/BiasAddBiasAdd%depthwise_conv2d_2/depthwise:output:01depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
depthwise_conv2d_2/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d_2/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
re_lu_5/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_5/Relu?
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp?
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_3/depthwise/Shape?
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate?
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_5/Relu:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise?
)depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_3/BiasAdd/ReadVariableOp?
depthwise_conv2d_3/BiasAddBiasAdd%depthwise_conv2d_3/depthwise:output:01depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
depthwise_conv2d_3/BiasAdd?
re_lu_6/ReluRelu#depthwise_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_6/Relux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2re_lu_6/Relu:activations:0inputs"concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????Z`2
concatenate_1/concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcatenate_1/concat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@2
conv2d_3/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????X@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
re_lu_7/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????X@2
re_lu_7/Relu?
IdentityIdentityre_lu_7/Relu:activations:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*^depthwise_conv2d_2/BiasAdd/ReadVariableOp,^depthwise_conv2d_2/depthwise/ReadVariableOp*^depthwise_conv2d_3/BiasAdd/ReadVariableOp,^depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*/
_output_shapes
:?????????X@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????Z@: : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2V
)depthwise_conv2d_2/BiasAdd/ReadVariableOp)depthwise_conv2d_2/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_2/depthwise/ReadVariableOp+depthwise_conv2d_2/depthwise/ReadVariableOp2V
)depthwise_conv2d_3/BiasAdd/ReadVariableOp)depthwise_conv2d_3/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_3/depthwise/ReadVariableOp+depthwise_conv2d_3/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
??
?
D__inference_model_layer_call_and_return_conditional_losses_511314631

inputs?
%conv2d_conv2d_readvariableop_resource:  4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: L
2depthwise_conv2d_depthwise_readvariableop_resource: >
0depthwise_conv2d_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_1_depthwise_readvariableop_resource: @
2depthwise_conv2d_1_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'depthwise_conv2d/BiasAdd/ReadVariableOp?)depthwise_conv2d/depthwise/ReadVariableOp?)depthwise_conv2d_1/BiasAdd/ReadVariableOp?+depthwise_conv2d_1/depthwise/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

re_lu/Relu?
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp2depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp?
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape?
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate?
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
depthwise_conv2d/depthwise?
'depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp0depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'depthwise_conv2d/BiasAdd/ReadVariableOp?
depthwise_conv2d/BiasAddBiasAdd#depthwise_conv2d/depthwise:output:0/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
depthwise_conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!depthwise_conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_1/Relu?
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp?
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape?
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate?
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_1/Relu:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
depthwise_conv2d_1/depthwise?
)depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_1/BiasAdd/ReadVariableOp?
depthwise_conv2d_1/BiasAddBiasAdd%depthwise_conv2d_1/depthwise:output:01depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
depthwise_conv2d_1/BiasAdd?
re_lu_2/ReluRelu#depthwise_conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
re_lu_2/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2re_lu_2/Relu:activations:0inputs concatenate/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????@2
concatenate/concat?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconcatenate/concat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
conv2d_1/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
re_lu_3/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
re_lu_3/Relu?	
IdentityIdentityre_lu_3/Relu:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^depthwise_conv2d/BiasAdd/ReadVariableOp*^depthwise_conv2d/depthwise/ReadVariableOp*^depthwise_conv2d_1/BiasAdd/ReadVariableOp,^depthwise_conv2d_1/depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'depthwise_conv2d/BiasAdd/ReadVariableOp'depthwise_conv2d/BiasAdd/ReadVariableOp2V
)depthwise_conv2d/depthwise/ReadVariableOp)depthwise_conv2d/depthwise/ReadVariableOp2V
)depthwise_conv2d_1/BiasAdd/ReadVariableOp)depthwise_conv2d_1/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_1/depthwise/ReadVariableOp+depthwise_conv2d_1/depthwise/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
+__inference_model_1_layer_call_fn_511314926

inputs!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:`@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????X@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113119252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????X@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????Z@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_511310241
input_1!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113101982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+??????????????????????????? 
!
_user_specified_name	input_1
?
?
+__inference_model_2_layer_call_fn_511313711

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: $

unknown_37: 

unknown_38: $

unknown_39:`@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45:@@

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@B

unknown_52:B
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B*H
_read_only_resource_inputs*
(&	
!"#$'()*+,/01256*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_5113128922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_511310661

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
F__inference_dropout_layer_call_and_return_conditional_losses_511315337

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????+@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????+@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????+@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????+@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????+@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_511310915

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_3_layer_call_fn_511315812

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5113106612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511312629

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????\ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_6_layer_call_fn_511314196

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5113117082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????\ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
e
F__inference_dropout_layer_call_and_return_conditional_losses_511312237

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????+@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????+@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????+@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????+@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????+@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
?
'__inference_signature_wrapper_511313485	
input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: $

unknown_37: 

unknown_38: $

unknown_39:`@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45:@@

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@B

unknown_52:B
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__wrapped_model_5113095342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????^

_user_specified_nameinput
?
?
9__inference_batch_normalization_1_layer_call_fn_511315607

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5113098752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_511315571

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_511315920

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_511315972

inputs8
conv2d_readvariableop_resource:`@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_4_layer_call_fn_511315884

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5113108102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_511310617

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
G
+__inference_re_lu_2_layer_call_fn_511315658

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_5113101542
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
,__inference_conv2d_1_layer_call_fn_511315685

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_5113101752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_2_layer_call_and_return_conditional_losses_511315663

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_511315848

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
D__inference_model_layer_call_and_return_conditional_losses_511312547

inputs?
%conv2d_conv2d_readvariableop_resource:  4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: L
2depthwise_conv2d_depthwise_readvariableop_resource: >
0depthwise_conv2d_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_1_depthwise_readvariableop_resource: @
2depthwise_conv2d_1_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'depthwise_conv2d/BiasAdd/ReadVariableOp?)depthwise_conv2d/depthwise/ReadVariableOp?)depthwise_conv2d_1/BiasAdd/ReadVariableOp?+depthwise_conv2d_1/depthwise/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2

re_lu/Relu?
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp2depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp?
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape?
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate?
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
depthwise_conv2d/depthwise?
'depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp0depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'depthwise_conv2d/BiasAdd/ReadVariableOp?
depthwise_conv2d/BiasAddBiasAdd#depthwise_conv2d/depthwise:output:0/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
depthwise_conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!depthwise_conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_1/Relu?
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp?
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape?
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate?
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_1/Relu:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
depthwise_conv2d_1/depthwise?
)depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_1/BiasAdd/ReadVariableOp?
depthwise_conv2d_1/BiasAddBiasAdd%depthwise_conv2d_1/depthwise:output:01depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
depthwise_conv2d_1/BiasAdd?
re_lu_2/ReluRelu#depthwise_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_2/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2re_lu_2/Relu:activations:0inputs concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????\@2
concatenate/concat?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconcatenate/concat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@2
conv2d_1/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
re_lu_3/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z@2
re_lu_3/Relu?	
IdentityIdentityre_lu_3/Relu:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^depthwise_conv2d/BiasAdd/ReadVariableOp*^depthwise_conv2d/depthwise/ReadVariableOp*^depthwise_conv2d_1/BiasAdd/ReadVariableOp,^depthwise_conv2d_1/depthwise/ReadVariableOp*
T0*/
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????\ : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'depthwise_conv2d/BiasAdd/ReadVariableOp'depthwise_conv2d/BiasAdd/ReadVariableOp2V
)depthwise_conv2d/depthwise/ReadVariableOp)depthwise_conv2d/depthwise/ReadVariableOp2V
)depthwise_conv2d_1/BiasAdd/ReadVariableOp)depthwise_conv2d_1/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_1/depthwise/ReadVariableOp+depthwise_conv2d_1/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511312206

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????+@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?B
?

D__inference_model_layer_call_and_return_conditional_losses_511310198

inputs*
conv2d_511310102:  
conv2d_511310104: +
batch_normalization_511310107: +
batch_normalization_511310109: +
batch_normalization_511310111: +
batch_normalization_511310113: 4
depthwise_conv2d_511310123: (
depthwise_conv2d_511310125: -
batch_normalization_1_511310128: -
batch_normalization_1_511310130: -
batch_normalization_1_511310132: -
batch_normalization_1_511310134: 6
depthwise_conv2d_1_511310144: *
depthwise_conv2d_1_511310146: ,
conv2d_1_511310176:@@ 
conv2d_1_511310178:@-
batch_normalization_2_511310181:@-
batch_normalization_2_511310183:@-
batch_normalization_2_511310185:@-
batch_normalization_2_511310187:@
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(depthwise_conv2d/StatefulPartitionedCall?*depthwise_conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_511310102conv2d_511310104*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_5113101012 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_511310107batch_normalization_511310109batch_normalization_511310111batch_normalization_511310113*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_5113096822-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_5113101212
re_lu/PartitionedCall?
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0depthwise_conv2d_511310123depthwise_conv2d_511310125*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_5113097992*
(depthwise_conv2d/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:0batch_normalization_1_511310128batch_normalization_1_511310130batch_normalization_1_511310132batch_normalization_1_511310134*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5113098312/
-batch_normalization_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_5113101422
re_lu_1/PartitionedCall?
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0depthwise_conv2d_1_511310144depthwise_conv2d_1_511310146*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_5113099482,
*depthwise_conv2d_1/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_5113101542
re_lu_2/PartitionedCall?
concatenate/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_5113101632
concatenate/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_1_511310176conv2d_1_511310178*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_5113101752"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_511310181batch_normalization_2_511310183batch_normalization_2_511310185batch_normalization_2_511310187*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5113099802/
-batch_normalization_2/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_5113101952
re_lu_3/PartitionedCall?
IdentityIdentity re_lu_3/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_depthwise_conv2d_2_layer_call_fn_511310744

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_5113107342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_5_layer_call_fn_511315998

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5113109592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
D__inference_model_layer_call_and_return_conditional_losses_511314791

inputs?
%conv2d_conv2d_readvariableop_resource:  4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: L
2depthwise_conv2d_depthwise_readvariableop_resource: >
0depthwise_conv2d_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_1_depthwise_readvariableop_resource: @
2depthwise_conv2d_1_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'depthwise_conv2d/BiasAdd/ReadVariableOp?)depthwise_conv2d/depthwise/ReadVariableOp?)depthwise_conv2d_1/BiasAdd/ReadVariableOp?+depthwise_conv2d_1/depthwise/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2

re_lu/Relu?
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp2depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp?
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape?
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate?
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
depthwise_conv2d/depthwise?
'depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp0depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'depthwise_conv2d/BiasAdd/ReadVariableOp?
depthwise_conv2d/BiasAddBiasAdd#depthwise_conv2d/depthwise:output:0/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
depthwise_conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!depthwise_conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_1/Relu?
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp?
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape?
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate?
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_1/Relu:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
depthwise_conv2d_1/depthwise?
)depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_1/BiasAdd/ReadVariableOp?
depthwise_conv2d_1/BiasAddBiasAdd%depthwise_conv2d_1/depthwise:output:01depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
depthwise_conv2d_1/BiasAdd?
re_lu_2/ReluRelu#depthwise_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_2/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2re_lu_2/Relu:activations:0inputs concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????\@2
concatenate/concat?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconcatenate/concat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@2
conv2d_1/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
re_lu_3/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z@2
re_lu_3/Relu?	
IdentityIdentityre_lu_3/Relu:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^depthwise_conv2d/BiasAdd/ReadVariableOp*^depthwise_conv2d/depthwise/ReadVariableOp*^depthwise_conv2d_1/BiasAdd/ReadVariableOp,^depthwise_conv2d_1/depthwise/ReadVariableOp*
T0*/
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????\ : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'depthwise_conv2d/BiasAdd/ReadVariableOp'depthwise_conv2d/BiasAdd/ReadVariableOp2V
)depthwise_conv2d/depthwise/ReadVariableOp)depthwise_conv2d/depthwise/ReadVariableOp2V
)depthwise_conv2d_1/BiasAdd/ReadVariableOp)depthwise_conv2d_1/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_1/depthwise/ReadVariableOp+depthwise_conv2d_1/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
G
+__inference_re_lu_4_layer_call_fn_511315853

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_5113110562
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_layer_call_and_return_conditional_losses_511310101

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_re_lu_6_layer_call_and_return_conditional_losses_511315940

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?O
?
F__inference_model_2_layer_call_and_return_conditional_losses_511312046

inputs,
conv2d_4_511311686:  
conv2d_4_511311688: -
batch_normalization_6_511311709: -
batch_normalization_6_511311711: -
batch_normalization_6_511311713: -
batch_normalization_6_511311715: )
model_511311805:  
model_511311807: 
model_511311809: 
model_511311811: 
model_511311813: 
model_511311815: )
model_511311817: 
model_511311819: 
model_511311821: 
model_511311823: 
model_511311825: 
model_511311827: )
model_511311829: 
model_511311831: )
model_511311833:@@
model_511311835:@
model_511311837:@
model_511311839:@
model_511311841:@
model_511311843:@+
model_1_511311926:@ 
model_1_511311928: 
model_1_511311930: 
model_1_511311932: 
model_1_511311934: 
model_1_511311936: +
model_1_511311938: 
model_1_511311940: 
model_1_511311942: 
model_1_511311944: 
model_1_511311946: 
model_1_511311948: +
model_1_511311950: 
model_1_511311952: +
model_1_511311954:`@
model_1_511311956:@
model_1_511311958:@
model_1_511311960:@
model_1_511311962:@
model_1_511311964:@,
conv2d_5_511311978:@@ 
conv2d_5_511311980:@-
batch_normalization_7_511312008:@-
batch_normalization_7_511312010:@-
batch_normalization_7_511312012:@-
batch_normalization_7_511312014:@-
container_511312036:@B!
container_511312038:B
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?!container/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_511311686conv2d_4_511311688*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_5113116852"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_6_511311709batch_normalization_6_511311711batch_normalization_6_511311713batch_normalization_6_511311715*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5113117082/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_5113117232
re_lu_8/PartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0model_511311805model_511311807model_511311809model_511311811model_511311813model_511311815model_511311817model_511311819model_511311821model_511311823model_511311825model_511311827model_511311829model_511311831model_511311833model_511311835model_511311837model_511311839model_511311841model_511311843* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????Z@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113118042
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0model_1_511311926model_1_511311928model_1_511311930model_1_511311932model_1_511311934model_1_511311936model_1_511311938model_1_511311940model_1_511311942model_1_511311944model_1_511311946model_1_511311948model_1_511311950model_1_511311952model_1_511311954model_1_511311956model_1_511311958model_1_511311960model_1_511311962model_1_511311964* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????X@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113119252!
model_1/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(model_1/StatefulPartitionedCall:output:0conv2d_5_511311978conv2d_5_511311980*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_5113119772"
 conv2d_5/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_5113119882
dropout/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0batch_normalization_7_511312008batch_normalization_7_511312010batch_normalization_7_511312012batch_normalization_7_511312014*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5113120072/
-batch_normalization_7/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_9_layer_call_and_return_conditional_losses_5113120222
re_lu_9/PartitionedCall?
!average_pooling2d/PartitionedCallPartitionedCall re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_5113116622#
!average_pooling2d/PartitionedCall?
!container/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0container_511312036container_511312038*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_container_layer_call_and_return_conditional_losses_5113120352#
!container/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose*container/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????B2"
 tf.compat.v1.transpose/transpose?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean$tf.compat.v1.transpose/transpose:y:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????B2
tf.math.reduce_mean/Mean?
IdentityIdentity!tf.math.reduce_mean/Mean:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall"^container/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????B2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!container/StatefulPartitionedCall!container/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511311552

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
G
+__inference_re_lu_7_layer_call_fn_511316039

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_5113111302
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_2_layer_call_fn_511315776

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_5113110362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
]
1__inference_concatenate_1_layer_call_fn_511315946
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_concatenate_1_layer_call_and_return_conditional_losses_5113110982
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+??????????????????????????? :+???????????????????????????@:k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/1
?
?
6__inference_depthwise_conv2d_1_layer_call_fn_511309958

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_5113099482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_511314471

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????Z@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113125472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????\ : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
+__inference_model_1_layer_call_fn_511314836

inputs!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:`@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_1_layer_call_and_return_conditional_losses_5113111332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+???????????????????????????@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_7_layer_call_fn_511315389

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5113122062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
??
?
F__inference_model_1_layer_call_and_return_conditional_losses_511315291

inputsA
'conv2d_2_conv2d_readvariableop_resource:@ 6
(conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_2_depthwise_readvariableop_resource: @
2depthwise_conv2d_2_biasadd_readvariableop_resource: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_3_depthwise_readvariableop_resource: @
2depthwise_conv2d_3_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:`@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity??$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?)depthwise_conv2d_2/BiasAdd/ReadVariableOp?+depthwise_conv2d_2/depthwise/ReadVariableOp?)depthwise_conv2d_3/BiasAdd/ReadVariableOp?+depthwise_conv2d_3/depthwise/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
conv2d_2/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
re_lu_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_4/Relu?
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_2_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp?
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_2/depthwise/Shape?
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate?
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise?
)depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_2/BiasAdd/ReadVariableOp?
depthwise_conv2d_2/BiasAddBiasAdd%depthwise_conv2d_2/depthwise:output:01depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
depthwise_conv2d_2/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d_2/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
re_lu_5/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_5/Relu?
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_3_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp?
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_3/depthwise/Shape?
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate?
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_5/Relu:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z *
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise?
)depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_3/BiasAdd/ReadVariableOp?
depthwise_conv2d_3/BiasAddBiasAdd%depthwise_conv2d_3/depthwise:output:01depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z 2
depthwise_conv2d_3/BiasAdd?
re_lu_6/ReluRelu#depthwise_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Z 2
re_lu_6/Relux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2re_lu_6/Relu:activations:0inputs"concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????Z`2
concatenate_1/concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcatenate_1/concat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X@2
conv2d_3/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????X@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
re_lu_7/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????X@2
re_lu_7/Relu?	
IdentityIdentityre_lu_7/Relu:activations:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*^depthwise_conv2d_2/BiasAdd/ReadVariableOp,^depthwise_conv2d_2/depthwise/ReadVariableOp*^depthwise_conv2d_3/BiasAdd/ReadVariableOp,^depthwise_conv2d_3/depthwise/ReadVariableOp*
T0*/
_output_shapes
:?????????X@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????Z@: : : : : : : : : : : : : : : : : : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2V
)depthwise_conv2d_2/BiasAdd/ReadVariableOp)depthwise_conv2d_2/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_2/depthwise/ReadVariableOp+depthwise_conv2d_2/depthwise/ReadVariableOp2V
)depthwise_conv2d_3/BiasAdd/ReadVariableOp)depthwise_conv2d_3/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_3/depthwise/ReadVariableOp+depthwise_conv2d_3/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315461

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????+@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_511315625

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?$
%__inference__traced_restore_511316401
file_prefix:
 assignvariableop_conv2d_4_kernel: .
 assignvariableop_1_conv2d_4_bias: <
.assignvariableop_2_batch_normalization_6_gamma: ;
-assignvariableop_3_batch_normalization_6_beta: B
4assignvariableop_4_batch_normalization_6_moving_mean: F
8assignvariableop_5_batch_normalization_6_moving_variance: <
"assignvariableop_6_conv2d_5_kernel:@@.
 assignvariableop_7_conv2d_5_bias:@<
.assignvariableop_8_batch_normalization_7_gamma:@;
-assignvariableop_9_batch_normalization_7_beta:@C
5assignvariableop_10_batch_normalization_7_moving_mean:@G
9assignvariableop_11_batch_normalization_7_moving_variance:@>
$assignvariableop_12_container_kernel:@B0
"assignvariableop_13_container_bias:B;
!assignvariableop_14_conv2d_kernel:  -
assignvariableop_15_conv2d_bias: ;
-assignvariableop_16_batch_normalization_gamma: :
,assignvariableop_17_batch_normalization_beta: A
3assignvariableop_18_batch_normalization_moving_mean: E
7assignvariableop_19_batch_normalization_moving_variance: O
5assignvariableop_20_depthwise_conv2d_depthwise_kernel: 7
)assignvariableop_21_depthwise_conv2d_bias: =
/assignvariableop_22_batch_normalization_1_gamma: <
.assignvariableop_23_batch_normalization_1_beta: C
5assignvariableop_24_batch_normalization_1_moving_mean: G
9assignvariableop_25_batch_normalization_1_moving_variance: Q
7assignvariableop_26_depthwise_conv2d_1_depthwise_kernel: 9
+assignvariableop_27_depthwise_conv2d_1_bias: =
#assignvariableop_28_conv2d_1_kernel:@@/
!assignvariableop_29_conv2d_1_bias:@=
/assignvariableop_30_batch_normalization_2_gamma:@<
.assignvariableop_31_batch_normalization_2_beta:@C
5assignvariableop_32_batch_normalization_2_moving_mean:@G
9assignvariableop_33_batch_normalization_2_moving_variance:@=
#assignvariableop_34_conv2d_2_kernel:@ /
!assignvariableop_35_conv2d_2_bias: =
/assignvariableop_36_batch_normalization_3_gamma: <
.assignvariableop_37_batch_normalization_3_beta: C
5assignvariableop_38_batch_normalization_3_moving_mean: G
9assignvariableop_39_batch_normalization_3_moving_variance: Q
7assignvariableop_40_depthwise_conv2d_2_depthwise_kernel: 9
+assignvariableop_41_depthwise_conv2d_2_bias: =
/assignvariableop_42_batch_normalization_4_gamma: <
.assignvariableop_43_batch_normalization_4_beta: C
5assignvariableop_44_batch_normalization_4_moving_mean: G
9assignvariableop_45_batch_normalization_4_moving_variance: Q
7assignvariableop_46_depthwise_conv2d_3_depthwise_kernel: 9
+assignvariableop_47_depthwise_conv2d_3_bias: =
#assignvariableop_48_conv2d_3_kernel:`@/
!assignvariableop_49_conv2d_3_bias:@=
/assignvariableop_50_batch_normalization_5_gamma:@<
.assignvariableop_51_batch_normalization_5_beta:@C
5assignvariableop_52_batch_normalization_5_moving_mean:@G
9assignvariableop_53_batch_normalization_5_moving_variance:@
identity_55??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
9272
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_6_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_6_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_6_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_6_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_7_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_7_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_7_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_7_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_container_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_container_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_conv2d_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_batch_normalization_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp,assignvariableop_17_batch_normalization_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp3assignvariableop_18_batch_normalization_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp5assignvariableop_20_depthwise_conv2d_depthwise_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_depthwise_conv2d_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_1_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_batch_normalization_1_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp5assignvariableop_24_batch_normalization_1_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp9assignvariableop_25_batch_normalization_1_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp7assignvariableop_26_depthwise_conv2d_1_depthwise_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_depthwise_conv2d_1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv2d_1_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv2d_1_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_batch_normalization_2_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_batch_normalization_2_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_2_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp9assignvariableop_33_batch_normalization_2_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp#assignvariableop_34_conv2d_2_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp!assignvariableop_35_conv2d_2_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_3_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_3_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_3_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_3_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp7assignvariableop_40_depthwise_conv2d_2_depthwise_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_depthwise_conv2d_2_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_4_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp.assignvariableop_43_batch_normalization_4_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp5assignvariableop_44_batch_normalization_4_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp9assignvariableop_45_batch_normalization_4_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp7assignvariableop_46_depthwise_conv2d_3_depthwise_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_depthwise_conv2d_3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp#assignvariableop_48_conv2d_3_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp!assignvariableop_49_conv2d_3_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp/assignvariableop_50_batch_normalization_5_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp.assignvariableop_51_batch_normalization_5_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp5assignvariableop_52_batch_normalization_5_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp9assignvariableop_53_batch_normalization_5_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54?	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*?
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
9__inference_batch_normalization_7_layer_call_fn_511315376

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????+@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5113120072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????+@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
?
E__inference_conv2d_layer_call_and_return_conditional_losses_511315509

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_re_lu_1_layer_call_and_return_conditional_losses_511310142

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
G
+__inference_re_lu_6_layer_call_fn_511315935

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_5113110892
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_re_lu_8_layer_call_and_return_conditional_losses_511311723

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????\ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????\ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????\ :W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
d
F__inference_dropout_layer_call_and_return_conditional_losses_511315325

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????+@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????+@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????+@:W S
/
_output_shapes
:?????????+@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314227

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_7_layer_call_fn_511315350

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5113115522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_re_lu_3_layer_call_and_return_conditional_losses_511315767

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_511309799

inputs;
!depthwise_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
	depthwise?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_5_layer_call_fn_511315985

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5113109152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?m
?
D__inference_model_layer_call_and_return_conditional_losses_511314711

inputs?
%conv2d_conv2d_readvariableop_resource:  4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: L
2depthwise_conv2d_depthwise_readvariableop_resource: >
0depthwise_conv2d_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: N
4depthwise_conv2d_1_depthwise_readvariableop_resource: @
2depthwise_conv2d_1_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'depthwise_conv2d/BiasAdd/ReadVariableOp?)depthwise_conv2d/depthwise/ReadVariableOp?)depthwise_conv2d_1/BiasAdd/ReadVariableOp?+depthwise_conv2d_1/depthwise/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2

re_lu/Relu?
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp2depthwise_conv2d_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp?
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape?
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate?
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
depthwise_conv2d/depthwise?
'depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp0depthwise_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'depthwise_conv2d/BiasAdd/ReadVariableOp?
depthwise_conv2d/BiasAddBiasAdd#depthwise_conv2d/depthwise:output:0/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
depthwise_conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!depthwise_conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????\ : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_1/Relu?
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_1_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp?
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape?
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate?
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_1/Relu:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ *
paddingSAME*
strides
2
depthwise_conv2d_1/depthwise?
)depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2depthwise_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)depthwise_conv2d_1/BiasAdd/ReadVariableOp?
depthwise_conv2d_1/BiasAddBiasAdd%depthwise_conv2d_1/depthwise:output:01depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\ 2
depthwise_conv2d_1/BiasAdd?
re_lu_2/ReluRelu#depthwise_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\ 2
re_lu_2/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2re_lu_2/Relu:activations:0inputs concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????\@2
concatenate/concat?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconcatenate/concat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Z@2
conv2d_1/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????Z@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
re_lu_3/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????Z@2
re_lu_3/Relu?
IdentityIdentityre_lu_3/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^depthwise_conv2d/BiasAdd/ReadVariableOp*^depthwise_conv2d/depthwise/ReadVariableOp*^depthwise_conv2d_1/BiasAdd/ReadVariableOp,^depthwise_conv2d_1/depthwise/ReadVariableOp*
T0*/
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????\ : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'depthwise_conv2d/BiasAdd/ReadVariableOp'depthwise_conv2d/BiasAdd/ReadVariableOp2V
)depthwise_conv2d/depthwise/ReadVariableOp)depthwise_conv2d/depthwise/ReadVariableOp2V
)depthwise_conv2d_1/BiasAdd/ReadVariableOp)depthwise_conv2d_1/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_1/depthwise/ReadVariableOp+depthwise_conv2d_1/depthwise/ReadVariableOp:W S
/
_output_shapes
:?????????\ 
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_511314381

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5113103952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:+??????????????????????????? : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input6
serving_default_input:0?????????^K
tf.math.reduce_mean4
StatefulPartitionedCall:0?????????Btensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"׬
_tf_keras_network??{"name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 94, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["re_lu_2", 0, 0, {}], ["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["re_lu_3", 0, 0]]}, "name": "model", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["re_lu_6", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["re_lu_7", 0, 0]]}, "name": "model_1", "inbound_nodes": [[["model", 1, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["model_1", 1, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_9", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "container", "trainable": true, "dtype": "float32", "filters": 66, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "container", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": true, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose", "inbound_nodes": [["container", 0, 0, {"perm": [0, 3, 1, 2], "name": "transpose", "conjugate": false}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean", "inbound_nodes": [["tf.compat.v1.transpose", 0, 0, {"axis": 2, "name": "output"}]]}], "input_layers": [["input", 0, 0]], "output_layers": [["tf.math.reduce_mean", 0, 0]]}, "shared_object_id": 98, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 24, 94, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 94, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 24, 94, 3]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 94, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["re_lu_2", 0, 0, {}], ["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["re_lu_3", 0, 0]]}, "name": "model", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]], "shared_object_id": 45}, {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["re_lu_6", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["re_lu_7", 0, 0]]}, "name": "model_1", "inbound_nodes": [[["model", 1, 0, {}]]], "shared_object_id": 81}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 82}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 83}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["model_1", 1, 0, {}]]], "shared_object_id": 84}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]], "shared_object_id": 85}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 86}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 87}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 88}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 89}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 90}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_9", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]], "shared_object_id": 91}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]], "shared_object_id": 92}, {"class_name": "Conv2D", "config": {"name": "container", "trainable": true, "dtype": "float32", "filters": 66, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 93}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 94}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "container", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]], "shared_object_id": 95}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": true, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose", "inbound_nodes": [["container", 0, 0, {"perm": [0, 3, 1, 2], "name": "transpose", "conjugate": false}]], "shared_object_id": 96}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean", "inbound_nodes": [["tf.compat.v1.transpose", 0, 0, {"axis": 2, "name": "output"}]], "shared_object_id": 97}], "input_layers": [["input", 0, 0]], "output_layers": [["tf.math.reduce_mean", 0, 0]]}}}
?
_init_input_shape"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 94, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 94, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 100}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 94, 3]}}
?

axis
	gamma
beta
moving_mean
moving_variance
 regularization_losses
!	variables
"trainable_variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv2d_4", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 101}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 92, 32]}}
?
$regularization_losses
%	variables
&trainable_variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]], "shared_object_id": 9}
?u
(layer-0
)layer_with_weights-0
)layer-1
*layer_with_weights-1
*layer-2
+layer-3
,layer_with_weights-2
,layer-4
-layer_with_weights-3
-layer-5
.layer-6
/layer_with_weights-4
/layer-7
0layer-8
1layer-9
2layer_with_weights-5
2layer-10
3layer_with_weights-6
3layer-11
4layer-12
5regularization_losses
6	variables
7trainable_variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?q
_tf_keras_network?q{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["re_lu_2", 0, 0, {}], ["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["re_lu_3", 0, 0]]}, "inbound_nodes": [[["re_lu_8", 0, 0, {}]]], "shared_object_id": 45, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 32]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 32]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]], "shared_object_id": 28}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]], "shared_object_id": 29}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]], "shared_object_id": 33}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]], "shared_object_id": 34}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["re_lu_2", 0, 0, {}], ["input_1", 0, 0, {}]]], "shared_object_id": 35}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 36}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 38}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 40}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 42}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]], "shared_object_id": 44}], "input_layers": [["input_1", 0, 0]], "output_layers": [["re_lu_3", 0, 0]]}}}
?u
9layer-0
:layer_with_weights-0
:layer-1
;layer_with_weights-1
;layer-2
<layer-3
=layer_with_weights-2
=layer-4
>layer_with_weights-3
>layer-5
?layer-6
@layer_with_weights-4
@layer-7
Alayer-8
Blayer-9
Clayer_with_weights-5
Clayer-10
Dlayer_with_weights-6
Dlayer-11
Elayer-12
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?q
_tf_keras_network?q{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["re_lu_6", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["re_lu_7", 0, 0]]}, "inbound_nodes": [[["model", 1, 0, {}]]], "shared_object_id": 81, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 64]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 64]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 46}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 49}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 51}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 53}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]], "shared_object_id": 55}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]], "shared_object_id": 59}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 60}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 61}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 63}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]], "shared_object_id": 64}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]], "shared_object_id": 65}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 67}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 68}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]], "shared_object_id": 69}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]], "shared_object_id": 70}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["re_lu_6", 0, 0, {}], ["input_2", 0, 0, {}]]], "shared_object_id": 71}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 72}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 73}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 74}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 75}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 76}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 77}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 78}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]], "shared_object_id": 79}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]], "shared_object_id": 80}], "input_layers": [["input_2", 0, 0]], "output_layers": [["re_lu_7", 0, 0]]}}}
?

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 82}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 83}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["model_1", 1, 0, {}]]], "shared_object_id": 84, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 104}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 88, 64]}}
?
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv2d_5", 0, 0, {}]]], "shared_object_id": 85}
?

Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 86}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 87}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 88}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 89}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 90, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 105}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 43, 64]}}
?
]regularization_losses
^	variables
_trainable_variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]], "shared_object_id": 91}
?
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["re_lu_9", 0, 0, {}]]], "shared_object_id": 92, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 106}}
?

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "container", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "container", "trainable": true, "dtype": "float32", "filters": 66, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 93}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 94}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]], "shared_object_id": 95, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 107}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 21, 64]}}
?
k	keras_api"?
_tf_keras_layer?{"name": "tf.compat.v1.transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": true, "dtype": "float32", "function": "compat.v1.transpose"}, "inbound_nodes": [["container", 0, 0, {"perm": [0, 3, 1, 2], "name": "transpose", "conjugate": false}]], "shared_object_id": 96}
?
l	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["tf.compat.v1.transpose", 0, 0, {"axis": 2, "name": "output"}]], "shared_object_id": 97}
 "
trackable_list_wrapper
?
0
1
2
3
4
5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
J46
K47
U48
V49
W50
X51
e52
f53"
trackable_list_wrapper
?
0
1
2
3
m4
n5
o6
p7
s8
t9
u10
v11
y12
z13
{14
|15
}16
~17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
J32
K33
U34
V35
e36
f37"
trackable_list_wrapper
?
?layer_metrics
regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
	variables
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
):' 2conv2d_4/kernel
: 2conv2d_4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?layer_metrics
regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
	variables
?layers
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?layer_metrics
 regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
!	variables
?layers
"trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
$regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
%	variables
?layers
&trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 32]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

mkernel
nbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 108}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?

	?axis
	ogamma
pbeta
qmoving_mean
rmoving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv2d", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 109}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["batch_normalization", 0, 0, {}]]], "shared_object_id": 19}
?
sdepthwise_kernel
tbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "depthwise_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "depthwise_regularizer": null, "depthwise_constraint": null}, "inbound_nodes": [[["re_lu", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 110}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?
	?axis
	ugamma
vbeta
wmoving_mean
xmoving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 111}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]], "shared_object_id": 29}
?
ydepthwise_kernel
zbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "depthwise_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "depthwise_regularizer": null, "depthwise_constraint": null}, "inbound_nodes": [[["re_lu_1", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 112}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]], "shared_object_id": 34}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["re_lu_2", 0, 0, {}], ["input_1", 0, 0, {}]]], "shared_object_id": 35, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 32]}, {"class_name": "TensorShape", "items": [null, null, null, 32]}]}
?

{kernel
|bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 36}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 113}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
	?axis
	}gamma
~beta
moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 40}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 42}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 114}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]], "shared_object_id": 44}
 "
trackable_list_wrapper
?
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13
{14
|15
}16
~17
18
?19"
trackable_list_wrapper
?
m0
n1
o2
p3
s4
t5
u6
v7
y8
z9
{10
|11
}12
~13"
trackable_list_wrapper
?
?layer_metrics
5regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
6	variables
7trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 64]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 115}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 51}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 53}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv2d_2", 0, 0, {}]]], "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 116}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]], "shared_object_id": 55}
?
?depthwise_kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "depthwise_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "depthwise_regularizer": null, "depthwise_constraint": null}, "inbound_nodes": [[["re_lu_4", 0, 0, {}]]], "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 117}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 60}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 61}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 63}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]], "shared_object_id": 64, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 118}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]], "shared_object_id": 65}
?
?depthwise_kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "depthwise_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 67}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 68}, "depthwise_regularizer": null, "depthwise_constraint": null}, "inbound_nodes": [[["re_lu_5", 0, 0, {}]]], "shared_object_id": 69, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 119}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]], "shared_object_id": 70}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["re_lu_6", 0, 0, {}], ["input_2", 0, 0, {}]]], "shared_object_id": 71, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 32]}, {"class_name": "TensorShape", "items": [null, null, null, 64]}]}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 72}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 73}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 74, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 96}}, "shared_object_id": 120}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 96]}}
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 75}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 76}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 77}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 78}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv2d_3", 0, 0, {}]]], "shared_object_id": 79, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 121}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]], "shared_object_id": 80}
 "
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
?
?layer_metrics
Fregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
G	variables
Htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
?layer_metrics
Lregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
M	variables
?layers
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Pregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
Q	variables
?layers
Rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
<
U0
V1
W2
X3"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
?layer_metrics
Yregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
Z	variables
?layers
[trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
]regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
^	variables
?layers
_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
aregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
b	variables
?layers
ctrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@B2container/kernel
:B2container/bias
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
?layer_metrics
gregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
h	variables
?layers
itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
':%  2conv2d/kernel
: 2conv2d/bias
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
;:9 2!depthwise_conv2d/depthwise_kernel
#:! 2depthwise_conv2d/bias
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
=:; 2#depthwise_conv2d_1/depthwise_kernel
%:# 2depthwise_conv2d_1/bias
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
):'@ 2conv2d_2/kernel
: 2conv2d_2/bias
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
=:; 2#depthwise_conv2d_2/depthwise_kernel
%:# 2depthwise_conv2d_2/bias
):' 2batch_normalization_4/gamma
(:& 2batch_normalization_4/beta
1:/  (2!batch_normalization_4/moving_mean
5:3  (2%batch_normalization_4/moving_variance
=:; 2#depthwise_conv2d_3/depthwise_kernel
%:# 2depthwise_conv2d_3/bias
):'`@2conv2d_3/kernel
:@2conv2d_3/bias
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
 "
trackable_dict_wrapper
?
0
1
q2
r3
w4
x5
6
?7
?8
?9
?10
?11
?12
?13
W14
X15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
o0
p1
q2
r3"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
u0
v1
w2
x3"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
=
}0
~1
2
?3"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
K
q0
r1
w2
x3
4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?	variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
+__inference_model_2_layer_call_fn_511312157
+__inference_model_2_layer_call_fn_511313598
+__inference_model_2_layer_call_fn_511313711
+__inference_model_2_layer_call_fn_511313116?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference__wrapped_model_511309534?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *,?)
'?$
input?????????^
?2?
F__inference_model_2_layer_call_and_return_conditional_losses_511313921
F__inference_model_2_layer_call_and_return_conditional_losses_511314138
F__inference_model_2_layer_call_and_return_conditional_losses_511313243
F__inference_model_2_layer_call_and_return_conditional_losses_511313370?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_conv2d_4_layer_call_fn_511314147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_511314157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_batch_normalization_6_layer_call_fn_511314170
9__inference_batch_normalization_6_layer_call_fn_511314183
9__inference_batch_normalization_6_layer_call_fn_511314196
9__inference_batch_normalization_6_layer_call_fn_511314209?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314227
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314245
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314263
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314281?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_re_lu_8_layer_call_fn_511314286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_8_layer_call_and_return_conditional_losses_511314291?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_model_layer_call_fn_511310241
)__inference_model_layer_call_fn_511314336
)__inference_model_layer_call_fn_511314381
)__inference_model_layer_call_fn_511310483
)__inference_model_layer_call_fn_511314426
)__inference_model_layer_call_fn_511314471?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_model_layer_call_and_return_conditional_losses_511314551
D__inference_model_layer_call_and_return_conditional_losses_511314631
D__inference_model_layer_call_and_return_conditional_losses_511310539
D__inference_model_layer_call_and_return_conditional_losses_511310595
D__inference_model_layer_call_and_return_conditional_losses_511314711
D__inference_model_layer_call_and_return_conditional_losses_511314791?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_model_1_layer_call_fn_511311176
+__inference_model_1_layer_call_fn_511314836
+__inference_model_1_layer_call_fn_511314881
+__inference_model_1_layer_call_fn_511311418
+__inference_model_1_layer_call_fn_511314926
+__inference_model_1_layer_call_fn_511314971?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_model_1_layer_call_and_return_conditional_losses_511315051
F__inference_model_1_layer_call_and_return_conditional_losses_511315131
F__inference_model_1_layer_call_and_return_conditional_losses_511311474
F__inference_model_1_layer_call_and_return_conditional_losses_511311530
F__inference_model_1_layer_call_and_return_conditional_losses_511315211
F__inference_model_1_layer_call_and_return_conditional_losses_511315291?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_conv2d_5_layer_call_fn_511315300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_511315310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_layer_call_fn_511315315
+__inference_dropout_layer_call_fn_511315320?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_layer_call_and_return_conditional_losses_511315325
F__inference_dropout_layer_call_and_return_conditional_losses_511315337?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_7_layer_call_fn_511315350
9__inference_batch_normalization_7_layer_call_fn_511315363
9__inference_batch_normalization_7_layer_call_fn_511315376
9__inference_batch_normalization_7_layer_call_fn_511315389?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315407
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315425
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315443
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315461?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_re_lu_9_layer_call_fn_511315466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_9_layer_call_and_return_conditional_losses_511315471?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_average_pooling2d_layer_call_fn_511311668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_511311662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_container_layer_call_fn_511315480?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_container_layer_call_and_return_conditional_losses_511315490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_511313485input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_layer_call_fn_511315499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_layer_call_and_return_conditional_losses_511315509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_layer_call_fn_511315522
7__inference_batch_normalization_layer_call_fn_511315535?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_511315553
R__inference_batch_normalization_layer_call_and_return_conditional_losses_511315571?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_re_lu_layer_call_fn_511315576?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_re_lu_layer_call_and_return_conditional_losses_511315581?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_depthwise_conv2d_layer_call_fn_511309809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
O__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_511309799?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
9__inference_batch_normalization_1_layer_call_fn_511315594
9__inference_batch_normalization_1_layer_call_fn_511315607?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_511315625
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_511315643?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_re_lu_1_layer_call_fn_511315648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_1_layer_call_and_return_conditional_losses_511315653?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_depthwise_conv2d_1_layer_call_fn_511309958?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
Q__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_511309948?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
+__inference_re_lu_2_layer_call_fn_511315658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_2_layer_call_and_return_conditional_losses_511315663?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_concatenate_layer_call_fn_511315669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_concatenate_layer_call_and_return_conditional_losses_511315676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_1_layer_call_fn_511315685?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_511315695?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_batch_normalization_2_layer_call_fn_511315708
9__inference_batch_normalization_2_layer_call_fn_511315721?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_511315739
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_511315757?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_re_lu_3_layer_call_fn_511315762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_3_layer_call_and_return_conditional_losses_511315767?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_2_layer_call_fn_511315776?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_511315786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_batch_normalization_3_layer_call_fn_511315799
9__inference_batch_normalization_3_layer_call_fn_511315812?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_511315830
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_511315848?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_re_lu_4_layer_call_fn_511315853?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_4_layer_call_and_return_conditional_losses_511315858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_depthwise_conv2d_2_layer_call_fn_511310744?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
Q__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_511310734?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
9__inference_batch_normalization_4_layer_call_fn_511315871
9__inference_batch_normalization_4_layer_call_fn_511315884?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_511315902
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_511315920?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_re_lu_5_layer_call_fn_511315925?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_5_layer_call_and_return_conditional_losses_511315930?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_depthwise_conv2d_3_layer_call_fn_511310893?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
Q__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_511310883?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
+__inference_re_lu_6_layer_call_fn_511315935?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_6_layer_call_and_return_conditional_losses_511315940?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_concatenate_1_layer_call_fn_511315946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_concatenate_1_layer_call_and_return_conditional_losses_511315953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_3_layer_call_fn_511315962?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_511315972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_batch_normalization_5_layer_call_fn_511315985
9__inference_batch_normalization_5_layer_call_fn_511315998?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_511316016
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_511316034?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_re_lu_7_layer_call_fn_511316039?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_7_layer_call_and_return_conditional_losses_511316044?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
$__inference__wrapped_model_511309534?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef6?3
,?)
'?$
input?????????^
? "M?J
H
tf.math.reduce_mean1?.
tf.math.reduce_mean?????????B?
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_511311662?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
5__inference_average_pooling2d_layer_call_fn_511311668?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_511315625?uvwxM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_511315643?uvwxM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
9__inference_batch_normalization_1_layer_call_fn_511315594?uvwxM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_1_layer_call_fn_511315607?uvwxM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_511315739?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_511315757?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
9__inference_batch_normalization_2_layer_call_fn_511315708?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_2_layer_call_fn_511315721?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_511315830?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_511315848?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
9__inference_batch_normalization_3_layer_call_fn_511315799?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_3_layer_call_fn_511315812?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_511315902?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_511315920?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
9__inference_batch_normalization_4_layer_call_fn_511315871?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_4_layer_call_fn_511315884?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_511316016?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_511316034?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
9__inference_batch_normalization_5_layer_call_fn_511315985?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_5_layer_call_fn_511315998?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314227?M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314245?M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314263r;?8
1?.
(?%
inputs?????????\ 
p 
? "-?*
#? 
0?????????\ 
? ?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_511314281r;?8
1?.
(?%
inputs?????????\ 
p
? "-?*
#? 
0?????????\ 
? ?
9__inference_batch_normalization_6_layer_call_fn_511314170?M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_6_layer_call_fn_511314183?M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_6_layer_call_fn_511314196e;?8
1?.
(?%
inputs?????????\ 
p 
? " ??????????\ ?
9__inference_batch_normalization_6_layer_call_fn_511314209e;?8
1?.
(?%
inputs?????????\ 
p
? " ??????????\ ?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315407?UVWXM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315425?UVWXM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315443rUVWX;?8
1?.
(?%
inputs?????????+@
p 
? "-?*
#? 
0?????????+@
? ?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511315461rUVWX;?8
1?.
(?%
inputs?????????+@
p
? "-?*
#? 
0?????????+@
? ?
9__inference_batch_normalization_7_layer_call_fn_511315350?UVWXM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_7_layer_call_fn_511315363?UVWXM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
9__inference_batch_normalization_7_layer_call_fn_511315376eUVWX;?8
1?.
(?%
inputs?????????+@
p 
? " ??????????+@?
9__inference_batch_normalization_7_layer_call_fn_511315389eUVWX;?8
1?.
(?%
inputs?????????+@
p
? " ??????????+@?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_511315553?opqrM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_511315571?opqrM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
7__inference_batch_normalization_layer_call_fn_511315522?opqrM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_layer_call_fn_511315535?opqrM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
L__inference_concatenate_1_layer_call_and_return_conditional_losses_511315953????
???
?|
<?9
inputs/0+??????????????????????????? 
<?9
inputs/1+???????????????????????????@
? "??<
5?2
0+???????????????????????????`
? ?
1__inference_concatenate_1_layer_call_fn_511315946????
???
?|
<?9
inputs/0+??????????????????????????? 
<?9
inputs/1+???????????????????????????@
? "2?/+???????????????????????????`?
J__inference_concatenate_layer_call_and_return_conditional_losses_511315676????
???
?|
<?9
inputs/0+??????????????????????????? 
<?9
inputs/1+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
/__inference_concatenate_layer_call_fn_511315669????
???
?|
<?9
inputs/0+??????????????????????????? 
<?9
inputs/1+??????????????????????????? 
? "2?/+???????????????????????????@?
H__inference_container_layer_call_and_return_conditional_losses_511315490lef7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????B
? ?
-__inference_container_layer_call_fn_511315480_ef7?4
-?*
(?%
inputs?????????@
? " ??????????B?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_511315695?{|I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_conv2d_1_layer_call_fn_511315685?{|I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_511315786???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
,__inference_conv2d_2_layer_call_fn_511315776???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_511315972???I?F
??<
:?7
inputs+???????????????????????????`
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_conv2d_3_layer_call_fn_511315962???I?F
??<
:?7
inputs+???????????????????????????`
? "2?/+???????????????????????????@?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_511314157l7?4
-?*
(?%
inputs?????????^
? "-?*
#? 
0?????????\ 
? ?
,__inference_conv2d_4_layer_call_fn_511314147_7?4
-?*
(?%
inputs?????????^
? " ??????????\ ?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_511315310lJK7?4
-?*
(?%
inputs?????????X@
? "-?*
#? 
0?????????+@
? ?
,__inference_conv2d_5_layer_call_fn_511315300_JK7?4
-?*
(?%
inputs?????????X@
? " ??????????+@?
E__inference_conv2d_layer_call_and_return_conditional_losses_511315509?mnI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
*__inference_conv2d_layer_call_fn_511315499?mnI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
Q__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_511309948?yzI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_depthwise_conv2d_1_layer_call_fn_511309958?yzI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
Q__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_511310734???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_depthwise_conv2d_2_layer_call_fn_511310744???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
Q__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_511310883???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_depthwise_conv2d_3_layer_call_fn_511310893???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
O__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_511309799?stI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_depthwise_conv2d_layer_call_fn_511309809?stI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
F__inference_dropout_layer_call_and_return_conditional_losses_511315325l;?8
1?.
(?%
inputs?????????+@
p 
? "-?*
#? 
0?????????+@
? ?
F__inference_dropout_layer_call_and_return_conditional_losses_511315337l;?8
1?.
(?%
inputs?????????+@
p
? "-?*
#? 
0?????????+@
? ?
+__inference_dropout_layer_call_fn_511315315_;?8
1?.
(?%
inputs?????????+@
p 
? " ??????????+@?
+__inference_dropout_layer_call_fn_511315320_;?8
1?.
(?%
inputs?????????+@
p
? " ??????????+@?
F__inference_model_1_layer_call_and_return_conditional_losses_511311474?(????????????????????R?O
H?E
;?8
input_2+???????????????????????????@
p 

 
? "??<
5?2
0+???????????????????????????@
? ?
F__inference_model_1_layer_call_and_return_conditional_losses_511311530?(????????????????????R?O
H?E
;?8
input_2+???????????????????????????@
p

 
? "??<
5?2
0+???????????????????????????@
? ?
F__inference_model_1_layer_call_and_return_conditional_losses_511315051?(????????????????????Q?N
G?D
:?7
inputs+???????????????????????????@
p 

 
? "??<
5?2
0+???????????????????????????@
? ?
F__inference_model_1_layer_call_and_return_conditional_losses_511315131?(????????????????????Q?N
G?D
:?7
inputs+???????????????????????????@
p

 
? "??<
5?2
0+???????????????????????????@
? ?
F__inference_model_1_layer_call_and_return_conditional_losses_511315211?(??????????????????????<
5?2
(?%
inputs?????????Z@
p 

 
? "-?*
#? 
0?????????X@
? ?
F__inference_model_1_layer_call_and_return_conditional_losses_511315291?(??????????????????????<
5?2
(?%
inputs?????????Z@
p

 
? "-?*
#? 
0?????????X@
? ?
+__inference_model_1_layer_call_fn_511311176?(????????????????????R?O
H?E
;?8
input_2+???????????????????????????@
p 

 
? "2?/+???????????????????????????@?
+__inference_model_1_layer_call_fn_511311418?(????????????????????R?O
H?E
;?8
input_2+???????????????????????????@
p

 
? "2?/+???????????????????????????@?
+__inference_model_1_layer_call_fn_511314836?(????????????????????Q?N
G?D
:?7
inputs+???????????????????????????@
p 

 
? "2?/+???????????????????????????@?
+__inference_model_1_layer_call_fn_511314881?(????????????????????Q?N
G?D
:?7
inputs+???????????????????????????@
p

 
? "2?/+???????????????????????????@?
+__inference_model_1_layer_call_fn_511314926?(??????????????????????<
5?2
(?%
inputs?????????Z@
p 

 
? " ??????????X@?
+__inference_model_1_layer_call_fn_511314971?(??????????????????????<
5?2
(?%
inputs?????????Z@
p

 
? " ??????????X@?
F__inference_model_2_layer_call_and_return_conditional_losses_511313243?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef>?;
4?1
'?$
input?????????^
p 

 
? ")?&
?
0?????????B
? ?
F__inference_model_2_layer_call_and_return_conditional_losses_511313370?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef>?;
4?1
'?$
input?????????^
p

 
? ")?&
?
0?????????B
? ?
F__inference_model_2_layer_call_and_return_conditional_losses_511313921?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef??<
5?2
(?%
inputs?????????^
p 

 
? ")?&
?
0?????????B
? ?
F__inference_model_2_layer_call_and_return_conditional_losses_511314138?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef??<
5?2
(?%
inputs?????????^
p

 
? ")?&
?
0?????????B
? ?
+__inference_model_2_layer_call_fn_511312157?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef>?;
4?1
'?$
input?????????^
p 

 
? "??????????B?
+__inference_model_2_layer_call_fn_511313116?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef>?;
4?1
'?$
input?????????^
p

 
? "??????????B?
+__inference_model_2_layer_call_fn_511313598?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef??<
5?2
(?%
inputs?????????^
p 

 
? "??????????B?
+__inference_model_2_layer_call_fn_511313711?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef??<
5?2
(?%
inputs?????????^
p

 
? "??????????B?
D__inference_model_layer_call_and_return_conditional_losses_511310539?mnopqrstuvwxyz{|}~?R?O
H?E
;?8
input_1+??????????????????????????? 
p 

 
? "??<
5?2
0+???????????????????????????@
? ?
D__inference_model_layer_call_and_return_conditional_losses_511310595?mnopqrstuvwxyz{|}~?R?O
H?E
;?8
input_1+??????????????????????????? 
p

 
? "??<
5?2
0+???????????????????????????@
? ?
D__inference_model_layer_call_and_return_conditional_losses_511314551?mnopqrstuvwxyz{|}~?Q?N
G?D
:?7
inputs+??????????????????????????? 
p 

 
? "??<
5?2
0+???????????????????????????@
? ?
D__inference_model_layer_call_and_return_conditional_losses_511314631?mnopqrstuvwxyz{|}~?Q?N
G?D
:?7
inputs+??????????????????????????? 
p

 
? "??<
5?2
0+???????????????????????????@
? ?
D__inference_model_layer_call_and_return_conditional_losses_511314711?mnopqrstuvwxyz{|}~???<
5?2
(?%
inputs?????????\ 
p 

 
? "-?*
#? 
0?????????Z@
? ?
D__inference_model_layer_call_and_return_conditional_losses_511314791?mnopqrstuvwxyz{|}~???<
5?2
(?%
inputs?????????\ 
p

 
? "-?*
#? 
0?????????Z@
? ?
)__inference_model_layer_call_fn_511310241?mnopqrstuvwxyz{|}~?R?O
H?E
;?8
input_1+??????????????????????????? 
p 

 
? "2?/+???????????????????????????@?
)__inference_model_layer_call_fn_511310483?mnopqrstuvwxyz{|}~?R?O
H?E
;?8
input_1+??????????????????????????? 
p

 
? "2?/+???????????????????????????@?
)__inference_model_layer_call_fn_511314336?mnopqrstuvwxyz{|}~?Q?N
G?D
:?7
inputs+??????????????????????????? 
p 

 
? "2?/+???????????????????????????@?
)__inference_model_layer_call_fn_511314381?mnopqrstuvwxyz{|}~?Q?N
G?D
:?7
inputs+??????????????????????????? 
p

 
? "2?/+???????????????????????????@?
)__inference_model_layer_call_fn_511314426zmnopqrstuvwxyz{|}~???<
5?2
(?%
inputs?????????\ 
p 

 
? " ??????????Z@?
)__inference_model_layer_call_fn_511314471zmnopqrstuvwxyz{|}~???<
5?2
(?%
inputs?????????\ 
p

 
? " ??????????Z@?
F__inference_re_lu_1_layer_call_and_return_conditional_losses_511315653?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_re_lu_1_layer_call_fn_511315648I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
F__inference_re_lu_2_layer_call_and_return_conditional_losses_511315663?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_re_lu_2_layer_call_fn_511315658I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
F__inference_re_lu_3_layer_call_and_return_conditional_losses_511315767?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_re_lu_3_layer_call_fn_511315762I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
F__inference_re_lu_4_layer_call_and_return_conditional_losses_511315858?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_re_lu_4_layer_call_fn_511315853I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
F__inference_re_lu_5_layer_call_and_return_conditional_losses_511315930?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_re_lu_5_layer_call_fn_511315925I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
F__inference_re_lu_6_layer_call_and_return_conditional_losses_511315940?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_re_lu_6_layer_call_fn_511315935I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
F__inference_re_lu_7_layer_call_and_return_conditional_losses_511316044?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_re_lu_7_layer_call_fn_511316039I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
F__inference_re_lu_8_layer_call_and_return_conditional_losses_511314291h7?4
-?*
(?%
inputs?????????\ 
? "-?*
#? 
0?????????\ 
? ?
+__inference_re_lu_8_layer_call_fn_511314286[7?4
-?*
(?%
inputs?????????\ 
? " ??????????\ ?
F__inference_re_lu_9_layer_call_and_return_conditional_losses_511315471h7?4
-?*
(?%
inputs?????????+@
? "-?*
#? 
0?????????+@
? ?
+__inference_re_lu_9_layer_call_fn_511315466[7?4
-?*
(?%
inputs?????????+@
? " ??????????+@?
D__inference_re_lu_layer_call_and_return_conditional_losses_511315581?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_re_lu_layer_call_fn_511315576I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
'__inference_signature_wrapper_511313485?Kmnopqrstuvwxyz{|}~?????????????????????JKUVWXef??<
? 
5?2
0
input'?$
input?????????^"M?J
H
tf.math.reduce_mean1?.
tf.math.reduce_mean?????????B