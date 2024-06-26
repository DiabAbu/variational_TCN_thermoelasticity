# In[16]:
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torch
print('PyTorch version=', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from tcn import TemporalConvNet as TCN 
import rff
from torch import linalg as LA
import json
from scipy.interpolate import griddata
import time
from scipy.spatial import Delaunay
import scipy.spatial

Length = 1 
Radius = 0.25

E        = 70e3
nu       = 0.3
lmbda_np = E*nu/((1+nu)*(1-2*nu))
mu_np    = E/2/(1+nu)
rho_np   = 2700.
alpha    = 2.31e-5
kappa_np = alpha*(2*mu_np + 3*lmbda_np)
cV_np    = 910e-6 * rho_np
k_np     = 237e-6
T0_np    = 293.
DThole_np = 20
Nincr     = 100
t         = np.logspace(1, 4, Nincr+1)
dt_array = np.diff(t)
rel_tol_network = 1e-16

device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    device = torch.device('cuda')
    device_string = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device_string = 'cpu'
    print("CUDA not available, running on CPU")
    
degree = 1
mesh   = Mesh("plate_hole_length_1_radius_quarter_mesh_density_25.xml")
dimens = mesh.geometry().dim() 

def find_element_connectivity():
    Vue = VectorElement('CG', mesh.ufl_cell(), degree)
    Vte = FiniteElement('CG', mesh.ufl_cell(), degree)
    V   = FunctionSpace(mesh, MixedElement([Vue, Vte]))
    V_sub_1_collapsed = V.sub(1).collapse()
    dofmap_T = V_sub_1_collapsed.dofmap()
    connectivity_list = []
    for cell in cells(mesh):
        local_to_global_vertex_map = dofmap_T.cell_dofs(cell.index())
        connectivity_list.append(local_to_global_vertex_map)
    connectivity_array = np.array(connectivity_list)
    connectivity_array = connectivity_array.astype(np.int64)
    return connectivity_array
connectivity_array = find_element_connectivity()
connectivity_tensor = torch.tensor(connectivity_array).long().to(device)

Training_times_vec = []        

def ConvergenceCheck( arry , rel_tol ):
    num_check = 10
    if len( arry ) < 2 * num_check :
        return False

    mean1 = np.mean( arry[ -2*num_check : -num_check ] )
    mean2 = np.mean( arry[ -num_check : ] )

    if np.abs( mean2 ) < 1e-6:
        print('Loss value converged to abs tol of 1e-6' )
        return True     

    if ( np.abs( mean1 - mean2 ) / np.abs( mean2 ) ) < rel_tol:
        print('Loss value converged to rel tol of ' + str(rel_tol) )
        return True
    else:
        return False
    
D_in          = 6   # dtime, x_coord, y_coord, E11, E22, E12
D_out         = 1   # temperature

T_n       = np.load('T_n.npy')
flux_n    = np.load('flux_n.npy')
inputs_n  = np.load('inputs_n.npy')
T_g       = np.load('T_g.npy')
flux_g    = np.load('flux_g.npy')
inputs_g  = np.load('inputs_g.npy')
coord_g_T = np.load('coord_g_T.npy')
coord_n   = np.load('coord_n.npy')
print('inputs_g shape:', inputs_g.shape)
print('inputs_n shape:', inputs_n.shape)
print('coord_n', coord_n.shape)

torch.manual_seed(2020)
torch.set_printoptions(precision=5)

T_n      = torch.tensor(T_n).double()
flux_n   = torch.tensor(flux_n).double()
inputs_n = torch.tensor(inputs_n).double()
inputs_n = inputs_n.to(device)
inputs_n.requires_grad_(True);   inputs_n.retain_grad()

T_g      = torch.tensor(T_g).double()
flux_g   = torch.tensor(flux_g).double()
inputs_g = torch.tensor(inputs_g).double()
inputs_g = inputs_g.to(device)
inputs_g.requires_grad_(True);   inputs_g.retain_grad()

with open('T_n_GT.json', 'r') as handle:
    T_n_GT = json.load(handle)
for key, value in T_n_GT.items():
    T_n_GT[key] = np.reshape(value["data"], value["shape"])

with open('ux_n_GT.json', 'r') as handle:
    ux_n_GT = json.load(handle)
for key, value in ux_n_GT.items():
    ux_n_GT[key] = np.reshape(value["data"], value["shape"])

with open('uy_n_GT.json', 'r') as handle:
    uy_n_GT = json.load(handle)
for key, value in uy_n_GT.items():
    uy_n_GT[key] = np.reshape(value["data"], value["shape"])

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size):
        super(Seq2Seq, self).__init__()
        num_channels1  = [16] * 4
        num_channels2  = [16] * 4
        enc_out_size   = 4
        act_func       = 'tanh' # tanh relu silu

        self.tcn1    = TCN(input_size,        num_channels1, act_func, kernel_size=12, dropout=0.00).double() 
        self.tcn2    = TCN(num_channels1[-1], num_channels2, act_func, kernel_size=12, dropout=0.00).double()
        self.encd    = rff.layers.GaussianEncoding(sigma=0.07, input_size=num_channels2[-1], encoded_size=enc_out_size).double()
        self.linear1 = nn.Linear(2*enc_out_size, output_size).double()
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        y  = self.tcn1(x.transpose(1,2))
        y  = self.tcn2(y)
        y  = self.encd(y.transpose(1,2))
        y  = self.linear1(y)
        return y

model = Seq2Seq(input_size=D_in, output_size=D_out)
model = model.to(device)
load_flag = True
if load_flag == True:
    PATH  = './SFvPItcn_plate_hole_model.ckpt'
    model = torch.load(PATH)
    model.eval()
    print('Model is loaded')
#Defining adam optimizer
adam_opt       = 'OFF'
adam_epochs    = 0
if adam_opt=='ON':
    adam_epochs    = 1000
    adam_lr_rate   = 0.01
    adamoptimizer  = torch.optim.Adam(model.parameters(), lr=adam_lr_rate)
#defining lbfgs optimizer
lbfgs_epochs    = adam_epochs + 50
LBFGS_max_iter  = 300
lbfgs_lr_rate   = 1.25
history_size    = 100
lbfgsoptimizer  = torch.optim.LBFGS(model.parameters(), lr=lbfgs_lr_rate, history_size = history_size, max_iter=LBFGS_max_iter, line_search_fn='strong_wolfe', tolerance_change=1e-15, tolerance_grad=1e-15)
##########################################################

def get_T(inputs):
    T_net   = model(inputs)
    radius2 = torch.pow(inputs[:,:,1], 2) + torch.pow(inputs[:,:,2], 2)
    radius  = torch.pow(radius2, 0.5).unsqueeze(2)
    T       = radius/Radius * DThole_np + torch.einsum('ijk,ijk->ijk', T_net, (radius - Radius)/Radius)
    return T

def compute_rate_of_change(variable, dt):
    N, T = variable.shape
    zero_start = torch.zeros((N, 1), dtype=variable.dtype, device=variable.device)
    variable = torch.cat([zero_start, variable], dim=1)
    return (variable[:, 1:] - variable[:, :-1]) / dt

def compute_rate_of_change_central(variable, dt):
    N, T = variable.shape
    rate_of_change = torch.zeros((N, T), dtype=variable.dtype, device=variable.device)
    
    rate_of_change[:, 1:-1] = (variable[:, 2:] - variable[:, :-2]) / (2 * dt[:, 1:-1])
    
    rate_of_change[:, 0] = (variable[:, 1] - variable[:, 0]) / dt[:, 0]
    
    rate_of_change[:, -1] = (variable[:, -1] - variable[:, -2]) / dt[:, -1]
    
    return rate_of_change

def shape_functions():
    vertices   = torch.tensor(coord_n[connectivity_array]).double().to(device)
    x_gauss  = torch.tensor(coord_g_T[:, 0]).double().to(device)
    y_gauss  = torch.tensor(coord_g_T[:, 1]).double().to(device)

    x_vertices = vertices[:,:,0]
    y_vertices = vertices[:,:,1]

    A = 0.5 * (x_vertices[:,0]*(y_vertices[:,1]-y_vertices[:,2]) + 
               x_vertices[:,1]*(y_vertices[:,2]-y_vertices[:,0]) + 
               x_vertices[:,2]*(y_vertices[:,0]-y_vertices[:,1]))

    print(x_vertices.shape, y_vertices.shape, A.shape)

    Ni = (1 / (2 * A)) * (x_vertices[:,1]*y_vertices[:,2] - x_vertices[:,2]*y_vertices[:,1] + x_gauss*(y_vertices[:,1]-y_vertices[:,2]) + y_gauss*(x_vertices[:,2]-x_vertices[:,1]))
    Nj = (1 / (2 * A)) * (x_vertices[:,2]*y_vertices[:,0] - x_vertices[:,0]*y_vertices[:,2] + x_gauss*(y_vertices[:,2]-y_vertices[:,0]) + y_gauss*(x_vertices[:,0]-x_vertices[:,2]))
    Nk = (1 / (2 * A)) * (x_vertices[:,0]*y_vertices[:,1] - x_vertices[:,1]*y_vertices[:,0] + x_gauss*(y_vertices[:,0]-y_vertices[:,1]) + y_gauss*(x_vertices[:,1]-x_vertices[:,0]))

    grad_Ni_x = (y_vertices[:, 1] - y_vertices[:, 2]) / (2 * A)
    grad_Ni_y = (x_vertices[:, 2] - x_vertices[:, 1]) / (2 * A)
    grad_Nj_x = (y_vertices[:, 2] - y_vertices[:, 0]) / (2 * A)
    grad_Nj_y = (x_vertices[:, 0] - x_vertices[:, 2]) / (2 * A)
    grad_Nk_x = (y_vertices[:, 0] - y_vertices[:, 1]) / (2 * A)
    grad_Nk_y = (x_vertices[:, 1] - x_vertices[:, 0]) / (2 * A)

    Ni_expanded = Ni[:, None].expand(-1, Nincr)
    Nj_expanded = Nj[:, None].expand(-1, Nincr)
    Nk_expanded = Nk[:, None].expand(-1, Nincr)

    return Ni_expanded, Nj_expanded, Nk_expanded, grad_Ni_x, grad_Ni_y, grad_Nj_x, grad_Nj_y, grad_Nk_x, grad_Nk_y

Ni_expanded, Nj_expanded, Nk_expanded, grad_Ni_x, grad_Ni_y, grad_Nj_x, grad_Nj_y, grad_Nk_x, grad_Nk_y = shape_functions()

def Gauss_Evaluations(u):

    u_vertices = u[connectivity_tensor, :] 

    u_gauss = Ni_expanded * u_vertices[:, 0, :] + Nj_expanded * u_vertices[:, 1, :] + Nk_expanded * u_vertices[:, 2, :]

    grad_u_x = grad_Ni_x[:, None] * u_vertices[:, 0, :] + grad_Nj_x[:, None] * u_vertices[:, 1, :] + grad_Nk_x[:, None] * u_vertices[:, 2, :]
    grad_u_y = grad_Ni_y[:, None] * u_vertices[:, 0, :] + grad_Nj_y[:, None] * u_vertices[:, 1, :] + grad_Nk_y[:, None] * u_vertices[:, 2, :]
    grad_u   = torch.cat([grad_u_x[:,:,None], grad_u_y[:,:,None]], dim=2)
    return u_gauss.unsqueeze(2), grad_u

def compute_areas(coord_g, simplices):
    areas = np.zeros(simplices.shape[0])
    for i, simplex in enumerate(simplices):
        v = coord_g[simplex]

        areas[i] = 0.5 * np.abs(np.linalg.det([v[1] - v[0], v[2] - v[0]]))
    return areas

def integrate(f_values, simplices, areas):
    f_avg = torch.mean(f_values[simplices.long()], dim=1)

    integral = torch.sum(areas.view(-1, 1) * f_avg, dim=0)

    return integral

coord_g_ar = inputs_g[:, 0, 1:3].cpu().detach().numpy()
tri = Delaunay(coord_g_ar)  
areas = compute_areas(coord_g_ar, tri.simplices)
areas = torch.tensor(areas).double()
simplices = torch.from_numpy(tri.simplices)

def loss_function(epoch, inputs_g, T_g, flux_g, areas, simplices):
    

    dt         = inputs_g[:,:,0].unsqueeze(2)
    E11        = inputs_g[:,:,3].unsqueeze(2)
    E22        = inputs_g[:,:,4].unsqueeze(2)
    E12        = inputs_g[:,:,5].unsqueeze(2)
    Etr        = E11 + E22
    time_inc   = inputs_g[0, :, 0].unsqueeze(0)

    T_ntwrk_n  = get_T(inputs_n) 
    T_net_g, g = Gauss_Evaluations(T_ntwrk_n.squeeze(2))
    Gt_integ1  = torch.cumsum(g[:,:,0].unsqueeze(2) * dt, dim=1)
    Gt_integ2  = torch.cumsum(g[:,:,1].unsqueeze(2) * dt, dim=1)

    v          = mu_np * (E11 * E11 + E22 * E22 + 2 * E12 * E12) + lmbda_np/2 * Etr * Etr + 0.5 * cV_np/T0_np * T_net_g * T_net_g
    V          = integrate(v.squeeze(2), simplices, areas)

    m          = 0.5 * k_np / T0_np * (Gt_integ1 * Gt_integ1 + Gt_integ2 * Gt_integ2)
    M          = integrate(m.squeeze(2), simplices, areas)
    D          = compute_rate_of_change(M.unsqueeze(0), time_inc).squeeze()

    seq        = V + D
    L1         = torch.sum(seq)

    r_Tdata_g  = T_net_g - T_g
    L2         =  LA.norm(r_Tdata_g)

    q_g        = -k_np * g
    r_qdata_g1 = q_g[:,:,0] - flux_g[:,:,0]
    r_qdata_g2 = q_g[:,:,1] - flux_g[:,:,1]
    R_qdata_g1 = LA.norm(r_qdata_g1, axis = 0)
    R_qdata_g2 = LA.norm(r_qdata_g2, axis = 0)
    L3         = torch.sum(R_qdata_g1) + torch.sum(R_qdata_g2)

    loss          = L1 + L2 + L3
    print(' Epoch=', epoch, ' LOSS=', loss.item(), ' V form=', L1.item(), ' T Gauss Loss=', L2.item(), ' q Gauss Loss=', L3.item())
    return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of trainable parameters=', count_parameters(model))


loss_history = {}
Temp_history = np.zeros((Nincr, len(coord_n), 1))
tempL        = []
start_time = time.time()
if adam_opt == 'ON':
    print("--------------------")
    print("Adam training begins")
    for epoch in range(adam_epochs):
        def closure():
            loss    = loss_function(epoch, inputs_g, T_g, flux_g, areas, simplices)
            adamoptimizer.zero_grad()
            loss.backward(retain_graph=True)
            tempL.append(loss.item())
            return loss
        adamoptimizer.step(closure)
        if ConvergenceCheck(tempL , rel_tol_network):
            break

print("--------------------")
print("LBFGS training begins")
for epoch in range(adam_epochs, lbfgs_epochs):
    def closure():
        loss    = loss_function(epoch, inputs_g, T_g, flux_g, areas, simplices)
        lbfgsoptimizer.zero_grad()
        loss.backward(retain_graph=True)
        tempL.append(loss.item())
        return loss
    lbfgsoptimizer.step(closure)
    if ConvergenceCheck(tempL , rel_tol_network):
        break
end_time = time.time()
print('Training time = ', end_time-start_time)
PATH = './SFvPItcn_plate_hole_model.ckpt'
torch.save(model, PATH)
print('Model is saved')

T_ntwrk_n1  = get_T(inputs_n[:coord_n.shape[0]//4, :, :]).cpu().detach().numpy()
T_ntwrk_n2  = get_T(inputs_n[coord_n.shape[0]//4:2*coord_n.shape[0]//4, :, :]).cpu().detach().numpy()
T_ntwrk_n3  = get_T(inputs_n[2*coord_n.shape[0]//4:3*coord_n.shape[0]//4, :, :]).cpu().detach().numpy()
T_ntwrk_n4  = get_T(inputs_n[3*coord_n.shape[0]//4:, :, :]).cpu().detach().numpy()
T_ntwrk_n   = np.concatenate((T_ntwrk_n1, T_ntwrk_n2, T_ntwrk_n3, T_ntwrk_n4), axis = 0)

VT          = FunctionSpace(mesh, "CG", degree)
T_n_ifenn  = {}
abs_error_T  = {}
rel_error_T  = {}
for key in T_n_GT.keys():
    key2               = int(key)
    T_n_ifenn[key2]    = torch.tensor(T_ntwrk_n[:,key2,:].squeeze()).double() 
    abs_error_T[key2]  = torch.absolute(T_n_ifenn[key2]  - torch.tensor(T_n_GT[key]))
    rel_error_T[key2]  = torch.div(abs_error_T[key2], torch.tensor(T_n_GT[key]))   

inc_plot    = 0
Delta_T_net = Function(VT)
Delta_T_net.vector().set_local(T_ntwrk_n[:,inc_plot,:])
plt.figure()
p = plot(Delta_T_net, title="Temperature variation inc 0")
plt.xlim((0, Length))
plt.ylim((0, Length))
plt.colorbar(p)
plt.show()

inc_plot    = 49
Delta_T_net = Function(VT)
Delta_T_net.vector().set_local(T_ntwrk_n[:,inc_plot,:])
plt.figure()
p = plot(Delta_T_net, title="Temperature variation inc 49")
plt.xlim((0, Length))
plt.ylim((0, Length))
plt.colorbar(p)
plt.show()

inc_plot    = 99
Delta_T_net = Function(VT)
Delta_T_net.vector().set_local(T_ntwrk_n[:,inc_plot,:])
plt.figure()
p = plot(Delta_T_net, title="Temperature variation inc 99")
plt.xlim((0, Length))
plt.ylim((0, Length))
plt.colorbar(p)
plt.show()

T_n_ifenn  = {}
abs_error_T  = {}
rel_error_T  = {}
for key in T_n_GT.keys():
    key2               = int(key)
    T_n_ifenn[key2]    = torch.tensor(T_ntwrk_n[:,key2,:].squeeze()).double() 
    abs_error_T[key2]  = torch.absolute(T_n_ifenn[key2]  - torch.tensor(T_n_GT[key]))
    rel_error_T[key2]  = torch.div(abs_error_T[key2], torch.tensor(T_n_GT[key]))   

inc_plot = 99
x  = coord_n[:, 0]
y  = coord_n[:, 1]
T  = T_ntwrk_n[:,inc_plot,:].squeeze()
Ea = abs_error_T[inc_plot].cpu().numpy()
Er = rel_error_T[inc_plot].cpu().numpy()

abs_error_T_fe  = Function(VT)
abs_error_T_np  = Ea
abs_error_T_fe.vector().set_local(abs_error_T_np)
plt.figure()
p = plot(abs_error_T_fe)
plt.xlim((0, Length))
plt.ylim((0, Length))
plt.colorbar(p)
plt.show()

rel_error_T_fe  = Function(VT)
rel_error_T_np  = Er
rel_error_T_fe.vector().set_local(rel_error_T_np)
plt.figure()
p = plot(rel_error_T_fe)
plt.xlim((0, Length))
plt.ylim((0, Length))
plt.colorbar(p)
plt.show()