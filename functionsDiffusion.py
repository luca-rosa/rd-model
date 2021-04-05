import numpy as np
import scipy
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm, animation, rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import time
import os
from matplotlib.backends.backend_pdf import PdfPages
import itertools


def ficks(s, w):
    return(laplace(s) / np.power(w, 2))


def hill(s, K, lam):
    h = s**lam / (K**lam + s**lam)
    return(h)


def multi_plots(sim, title=""):
    f, ax = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(10, 10))

    f.suptitle(title, fontsize=40)
    im1 = ax[0, 0].imshow(sim[3], interpolation="none", cmap=cm.viridis, vmin=0, vmax=1)
    ax[0, 0].set_title("Sender")
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[0, 0].axis("off")
    ax[0, 0].xaxis.label.set_size(7)
    ax[0, 0].yaxis.label.set_size(7)
    cb = f.colorbar(im1, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im2 = ax[0, 1].imshow(sim[5], interpolation="none", cmap=cm.viridis, vmin=0, vmax=1)
    ax[0, 1].set_title("Receiver")
    ax[0, 1].axis("off")
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im2, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im3 = ax[0, 2].imshow(sim[1], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[0, 2].set_title("Arabinose")
    ax[0, 2].axis("off")
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im3, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im4 = ax[1, 0].imshow(sim[3], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1, 0].set_title("LuxI")
    ax[1, 0].axis("off")
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im4, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im5 = ax[1, 1].imshow(sim[4], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1, 1].set_title("C6")
    ax[1, 1].axis("off")
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im5, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im6 = ax[1, 2].imshow(sim[6], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1, 2].set_title("GFP")
    ax[1, 2].axis("off")
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im6, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im7 = ax[2, 0].imshow(sim[2], interpolation="none", cmap=cm.viridis, vmin=0, vmax=100)
    ax[2, 0].set_title("Nutrients")
    ax[2, 0].axis("off")
    divider = make_axes_locatable(ax[2, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im7, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    ax[2, 1].axis('off')
    ax[2, 2].axis('off')

    return(f)


def multi_plots_vertical(sim, title=""):
    f, ax = plt.subplots(4, 2, sharex=False, sharey=False, figsize=(3, 6))

    f.suptitle(title, fontsize=40)
    im1 = ax[0, 0].imshow(sim[3], interpolation="none", cmap=cm.viridis, vmin=0, vmax=1)
    ax[0, 0].set_title("Sender")
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[0, 0].axis("off")
    ax[0, 0].xaxis.label.set_size(7)
    ax[0, 0].yaxis.label.set_size(7)
    cb = f.colorbar(im1, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im2 = ax[0, 1].imshow(sim[5], interpolation="none", cmap=cm.viridis, vmin=0, vmax=1)
    ax[0, 1].set_title("Receiver")
    ax[0, 1].axis("off")
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im2, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im3 = ax[1, 0].imshow(sim[1], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1, 0].set_title("Arabinose")
    ax[1, 0].axis("off")
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im3, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im4 = ax[1, 1].imshow(sim[3], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1, 1].set_title("LuxI")
    ax[1, 1].axis("off")
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im4, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im5 = ax[2, 0].imshow(sim[4], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[2, 0].set_title("C6")
    ax[2, 0].axis("off")
    divider = make_axes_locatable(ax[2, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im5, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im6 = ax[2, 1].imshow(sim[6], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[2, 1].set_title("GFP")
    ax[2, 1].axis("off")
    divider = make_axes_locatable(ax[2, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im6, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    im7 = ax[3, 0].imshow(sim[2], interpolation="none", cmap=cm.viridis, vmin=0, vmax=100)
    ax[3, 0].set_title("Nutrients")
    ax[3, 0].axis("off")
    divider = make_axes_locatable(ax[3, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = f.colorbar(im7, cax=cax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    ax[3, 1].axis('off')

    return(f)

def plots(sim, names):

    n_plots = sim.shape[0]
    x = int(np.ceil(n_plots / 3))

    f, ax = plt.subplots(x, 3, sharex=True, sharey=False, figsize=(15, 15))

    for i, val in enumerate(ax.flatten()):

        if i < n_plots:
            im = ax.flatten()[i].imshow(sim[i], cmap=cm.viridis, vmin=0)
            ax.flatten()[i].set_title(names[i])
            divider = make_axes_locatable(ax.flatten()[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            f.colorbar(im, cax=cax, shrink=0.8, label='')
        else:
            ax.flatten()[i].axis("off")


def get_vertex_coordinates(vertex_numbers, n_rows, n_cols):
    '''
    use to get grid coordinates of vertices

    args:
        vertex_numbers: the numbers of the vertices you want coordinates for 0 <= vertex_number < n_rows * n_cols
        n_rows, n_cols: number of rows and columns in the finite difference simulation, a total of n-rows*n_cols vertices

    returns:
        vertex_coordinates: the coordinates on the finite difference grid of the supplied vertex number: [[r0, c0]; [r1,c1]; ... [rn,cn]]
            these use matrix indexing, in the format (row, col) starting from the top left of the grid
    '''

    vertex_coordinates = np.hstack((vertex_numbers // n_rows, vertex_numbers % n_cols))

    return vertex_coordinates


def get_vertex_positions(vertex_numbers, n_rows, n_cols, w):
    '''
    use to get the positions (in mm) of vertices on the real grid

    args:
        vertex_numbers: the numbers of the vertices you want coordinates for 0 <= vertex_number < n_rows * n_cols
        n_rows, n_cols: number of rows and columns in the finite difference simulation, a total of n-rows*n_cols vertices
        w: the distance between finite difference vertices
    returns:
        vertex_positions: the positions on the finite difference grid of the supplied vertex number (in mm from the top left of the grid):
            [[r0, c0]; [r1,c1]; ... [rn,cn]]
    '''

    vertex_coordinates = get_vertex_coordinates(vertex_numbers, n_rows, n_cols)

    vertex_positions = vertex_coordinates * w

    return vertex_positions


def assign_vertices(vertex_positions, node_positions, node_radius):
    '''
    assigns vertices to be part of nodes in node_positions with radius: node radius.


    args:
        vertex_positions: the positions of the vertices to be tested
        node_positions, node_radius: positions and radius of the nodes we want vertices for
    returns:
        vertex_numbers: the numbers of the vertices that are within on of the nodes
        indicators: vector with an index for each vertex indicating whether it is inside a node (value = 1) or outside all nodes (value = 0)

     NOTE: this assigns position based on real life position, not the grid coordinates i.e the distance in mm
    '''

    indicators = np.zeros(len(vertex_positions))

    if node_positions == []:
        return [], indicators

    if node_positions[0] is not None:
        node_positions = np.array(node_positions)
        differences = vertex_positions - node_positions[:, None]

        vertex_numbers = np.where(np.linalg.norm(differences, axis=2) < node_radius)[1].reshape(-1, 1)

        indicators[vertex_numbers] = 1

    indicators = np.array(indicators, dtype=np.int32)

    return vertex_numbers, indicators

# this is the only one you really need to use


def get_node_coordinates(node_positions, node_radius, n_rows, n_cols, w):
    '''
       gets the coordinates of the vertices inside the nodes with position node_positions with radius: node radius.

       args:
           vertex_positions: the positions of the vertices to be tested
           node_positions, node_radius: positions and radius of the nodes we want vertices for
           n_rows, n_cols: the number of rows and cols on the finite difference grid
       returns:
           coordinates: the coordinates of the vertices that are within on of the nodes

        NOTE: this assigns position based on real life position, not the grid coordinates i.e the distance in mm
       '''

    # use the individual functions if repeating these two lines for each node type is too slow
    all_vertex_numbers = np.arange(n_rows * n_cols).reshape(-1, 1)  # reshpae to colum vector
    all_vertex_positions = get_vertex_positions(all_vertex_numbers, n_rows, n_cols, w)

    vertex_numbers, vertex_indicators = assign_vertices(all_vertex_positions, node_positions, node_radius)
    coordinates = get_vertex_coordinates(vertex_numbers, n_rows, n_cols)

    return coordinates


def runModel(t_final, dt, theta, U):
    shape = U.shape
    n_rows, n_cols = shape[1:]
    t_points = int(t_final / dt)

    t = np.arange(0, t_final, dt)
    U_init = U.flatten()  # solve_ivp wants initial condition as 1d array
    start_time = time.time()

    sim_ivp = solve_ivp(model_small, [0, t_final], U_init,
                        t_eval=t, args=(shape, theta))

    sim_ivp = sim_ivp.y.reshape(7, n_rows, n_cols, t_points)

    return(sim_ivp)


def full_setup(sender_seed, receiver_seed, ara=0, ahl=0):
    n_rows = n_cols = 46
    w = 0.75
    U = np.zeros([7, n_rows, n_cols])
    shape = U.shape
    size = U.size

    U[2] = 100  # set nutrients

    ## COORDINATES ##
#     dist = 0.75
    centre = (n_rows * w) / 2
    receiver_radius = 2

    # this is the cooridnates of the centre of each colony for one axis
    spacing = [centre - receiver_radius * 2 - 0.5, centre - receiver_radius * 4 - 1, centre - receiver_radius * 6 - 1.5,
               centre + receiver_radius * 2 + 0.5, centre + receiver_radius * 4 + 1, centre + receiver_radius * 6 + 1.5, centre]

    # since the grid it's square this gets
    coo_pos = list(itertools.product(spacing, repeat=2))

    # this sorts all cordinates left to right
    # e.g.  1 2 3
    #       4 5 6
    coo_pos.sort()
    coo_pos = np.array(coo_pos)
#     print(coo_pos)

#     print(type(coo_pos[[receiver_seed]]))
#     print(coo_pos[[receiver_seed]])

    ### RECEIVERS ###
    if receiver_seed:
        receiver_coordinates = get_node_coordinates(coo_pos[[receiver_seed]], receiver_radius, n_rows, n_cols, w)

        rows = receiver_coordinates[:, 0]
        cols = receiver_coordinates[:, 1]
        U[5][rows, cols] = 0.5

    U[4] = ahl  # set ahl concentration cast in the agar

    ###----SENDER---####

    if sender_seed:
        sender_radius = receiver_radius
        sender_coordinates = get_node_coordinates(coo_pos[[sender_seed]], sender_radius, n_rows, n_cols, w)

        # set initial sender conc
        rows = sender_coordinates[:, 0]
        cols = sender_coordinates[:, 1]
        U[3][rows, cols] = 0.5  # senders seeding
        U[1][rows, cols] = ara  # arabinose initial concentration
    return(U)


def get_coo_from_pos(positions):
    ## COORDINATES ##
    n_rows = n_cols = 46
    w = 0.75
    centre = (n_rows * w) / 2
    receiver_radius = 2

    # this is the cooridnates of the centre of each colony for one axis
    spacing = [centre - receiver_radius * 2 - 0.5, centre - receiver_radius * 4 - 1, centre - receiver_radius * 6 - 1.5,
               centre + receiver_radius * 2 + 0.5, centre + receiver_radius * 4 + 1, centre + receiver_radius * 6 + 1.5, centre]

    # since the grid it's square this gets
    coo_pos = list(itertools.product(spacing, repeat=2))

    # this sorts all cordinates left to right
    # e.g.  1 2 3
    #       4 5 6
    coo_pos.sort()
    coo_pos = np.array(coo_pos)

    coordinates = get_node_coordinates(coo_pos[[positions]], receiver_radius, 46, 46, w)

    rows = coordinates[:, 0]
    cols = coordinates[:, 1]

    r = coordinates[:, 0]
    c = coordinates[:, 1]

    return(r, c)


def model_small(t, U_flat, shape, theta):
    U_grid = U_flat.reshape(shape)

    x_s = theta['x_s']
    x_a = theta['x_a']
    x_g = theta['x_g']
    lambda_a = theta["lambda_a"]
    K_a = theta['K_a']
    D = theta['D']
    D_a = theta['D_a']
    w = theta['w']
    rho_n = theta['rho_n']
    rc = theta['rc']
    Dc = theta['Dc']
    rho = theta["rho"]
    lambda_n = theta['lambda_n']
    lambda_g = theta['lambda_g']
    K_g = theta['K_g']
    K_n = theta['K_n']

    # 0 LuxI
    # 1 Arabinose
    # 2 Nutrients
    # 3 Sender
    # 4 C6
    # 5 Receiver
    # 6 GFP

    N = hill(U_grid[2], K_n, lambda_n)

    LuxI_ficks = ficks(U_grid[0], w)
    arabinose_ficks = ficks(U_grid[1], w)
    n_ficks = ficks(U_grid[2], w)
    S_ficks = ficks(U_grid[3], w)
    c6_ficks = ficks(U_grid[4], w)
    R_ficks = ficks(U_grid[5], w)

    S = Dc * S_ficks + rc * N * U_grid[3]
    R = Dc * R_ficks + rc * N * U_grid[5]
    LuxI = x_s * hill(U_grid[1], K_a, lambda_a) * U_grid[3]
    c6 = D_a * c6_ficks + (x_a * U_grid[0]) - rho * U_grid[4]
    arabinose = D * arabinose_ficks
    n = D * n_ficks - rho_n * N * (U_grid[3] + U_grid[5])
    gfp = x_g * U_grid[5] * hill(U_grid[4], K_g, lambda_g) * U_grid[5]

    return(np.concatenate((LuxI.flatten(),
                           arabinose.flatten(),
                           n.flatten(),
                           S.flatten(),
                           c6.flatten(),
                           R.flatten(),
                           gfp.flatten())))
