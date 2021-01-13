import nibabel as nib
import argparse
from scipy import ndimage 
import numpy as np
from scipy.ndimage import label
import csv
import sys
from numpy import conjugate, sqrt
from numpy import linalg as LA
import math

def label_objects(image, sc):
    l, n = label(image, sc) 
    return l, n

def structurecomponent(s):
    if s == 1:
        sc = np.array([[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,1,1],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]]]) 
    elif s ==2:
        sc = np.array([[[0,1,0],[1,1,1],[0,1,0]], [[1,1,1],[1,1,1],[1,1,1]], [[0,1,0],[1,1,1],[0,1,0]]])
    elif s ==3:
        sc = np.array([[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])

    return sc

def get_all_indices(l,n):
    "Function to store each labelled component's indices"
    indices = [] #initialize list to store indices
    com = []
    for lab in range(1,n+1):        #for each object in the range of objects found
        #where a labelled object is the same as the current item in the loop, make the segment all equal to 1, and where it is not the same make the background voxels equal to zero 
        seg_lab = np.where(l==lab, np.ones_like(l), np.zeros_like(l))   
        
        #where the segment array is the same as 1, retrieve the indices
        indices_lab = np.where(seg_lab==1)    
        
        #add the indices to the list of indices          
        indices.append(indices_lab)  

        com.append(ndimage.measurements.center_of_mass(seg_lab))         
        
    return indices, com

def calc_euc_distance(x1,y1,z1,x2,y2, z2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return dist

def generate_bounding_image(image):
    cox,coy,coz = ndimage.measurements.center_of_mass(image)
    indices = np.where(image ==1)
    X,Y,Z = indices
    
    #find radius
    radius = 0
    for x,y,z in zip(X,Y,Z):
        if image[x,y,z] ==1:
            ed=calc_euc_distance(cox, coy, coz,x, y, z)
            if ed>radius:
                radius=ed
    
    index_x1 = int(round(cox - radius))
    index_x2 = int(round(cox + radius))
    index_y1 = int(round(coy - radius))
    index_y2 = int(round(coy + radius))
    index_z1 = int(round(coz - radius))
    index_z2 = int(round(coz + radius))

    bounding_image = image[index_x1:index_x2+1, index_y1:index_y2+1, index_z1:index_z2+1]   
    
    N_x = bounding_image.shape[0]
    N_y = bounding_image.shape[1]
    N_z = bounding_image.shape[2]
    
    return bounding_image,N_x,N_y,N_z
   
    
def factorial(n):
    if n == 0: 
        return 1
    return n * factorial(n - 1)


def binomial(n, k):
    c = factorial(n) / ( factorial(k) * factorial((n - k)))
    return c


def calculate_q(nu, k, l):

    term1 = ((-1) ** (k + nu))/(2. ** (2. * k))
    term2 = sqrt(((2 * l) + (4 * k) + 3) / 3)
    term3=binomial(2 * k, k)
    term4=binomial(k, nu)
    term5=binomial((2 * (k + l + nu) + 1), 2 * k)
    term6=binomial(k + l + nu, k)
    result = term1*term2*term3*term4*term5/term6

    return result 


    
def calculate_chi(n, l, m):
    "For each n,l,m tuple calculate the chi for the various r,s,t combinations"
    Table = np.zeros((n+1,n+1,n+1), dtype = complex)     
    list_r = []
    list_s = []
    list_t = []
    chi=0
    
    term1 = sqrt((2. * l + 1.) * factorial(int(l + m)) *
                  factorial(int((l - m)))) / factorial(int(l))
    term2 = 2 ** (-m)

    k = int((n - l) / 2)
    
    w = term1 * term2
    
    for nu in range(k + 1):
        q = calculate_q(nu, k, l)
        term3 = q*w
        for alpha in range(nu + 1):
            b1 = binomial(nu, alpha)
            term4 = term3*b1
            for beta in range(nu - alpha + 1):
                b2 = binomial(nu - alpha, beta)
                term5 = term4 * b2
                for u in range(m + 1):
                    b5 = binomial(m, u)
                    term6 = term5 * b5 * (1j**u) * ((-1.)**(m - u))
                    for mu in range(int((l - m) / 2) + 1):
                        b6 = binomial(l, mu)
                        b7 = binomial(l - mu, m + mu)
                        term7 = term6* (2**(-2 *mu)) * b6 * b7
                        for eta in range(mu + 1):
                            r = 2. * (eta + alpha) + u
                            s = 2. * (mu - eta + beta) + m - u
                            t = 2. * (nu - alpha - beta - mu) + l - m
                            chi = term7 * binomial(mu, eta)
                            
                            if r + s + t <= n and r >= 0 and s>= 0 and t>=0:
                                
                                "Create a lookup table (array index) so the chi calculated for an r,s,t combo can be retrieved"
                                Table[int(r)][int(s)][int(t)] = chi
                                list_r.append(int(r))
                                list_s.append(int(s))
                                list_t.append(int(t))

    return Table,list_r,list_s,list_t



def get_coefficients(N):
    "Make a list of the chi in the same order as the main program n,l,m loop so it can be indexed correctly"    
    all_coeff = []
    full_sequence_r = []
    full_sequence_s = []
    full_sequence_t = []
    for n in range(N+1):
        for l in range(0, n+1):
            if (n-l)% 2 == 0:
                for m in range(0,l+1):
                    tab_chi, list_r,list_s,list_t = calculate_chi(n,l,m)
                    all_coeff.append(tab_chi)
                    full_sequence_r.append(list_r)
                    full_sequence_s.append(list_s)
                    full_sequence_t.append(list_t)
                    
    return all_coeff,full_sequence_r,full_sequence_s,full_sequence_t
    

def symmetry(i,j,k,Nx,Ny,Nz, r,s,t,image):
    fA = 0
    
    f1 = [i,j,k]
    f2 = [Nx- i, j, k]
    f3 = [Nx - i, Ny - j, k]
    f4 = [i, Ny - j, k]
    f5 = [i, j, Nz - k]
    f6 = [Nx - i, j, Nz - k]
    f7 = [Nx - i,Ny - j,Nz - k]
    f8 = [i, Ny - j, Nz - k]
    
#case 1    
    if (r % 2) == 0 and (s % 2) == 0 and (t % 2) == 0:
        fA = (image[f1[0]][f1[1]][f1[2]] + image[f2[0]][f2[1]][f2[2]] + image[f3[0]][f3[1]][f3[2]] + image[f4[0]][f4[1]][f4[2]] 
            + image[f5[0]][f5[1]][f5[2]] + image[f6[0]][f6[1]][f6[2]] + image[f7[0]][f7[1]][f7[2]] + image[f8[0]][f8[1]][f8[2]] )
#case 2           
    elif (r % 2) == 0 and (s % 2) == 0 and (t % 2) != 0:
        fA = (- image[f1[0]][f1[1]][f1[2]] - image[f2[0]][f2[1]][f2[2]] - image[f3[0]][f3[1]][f3[2]] - image[f4[0]][f4[1]][f4[2]] 
              + image[f5[0]][f5[1]][f5[2]] + image[f6[0]][f6[1]][f6[2]] + image[f7[0]][f7[1]][f7[2]] + image[f8[0]][f8[1]][f8[2]] )
#case 3         
    elif (r % 2) == 0 and (s % 2) != 0 and (t % 2) == 0:
        fA = (- image[f1[0]][f1[1]][f1[2]] - image[f2[0]][f2[1]][f2[2]] + image[f3[0]][f3[1]][f3[2]] + image[f4[0]][f4[1]][f4[2]] 
              - image[f5[0]][f5[1]][f5[2]] - image[f6[0]][f6[1]][f6[2]] + image[f7[0]][f7[1]][f7[2]] + image[f8[0]][f8[1]][f8[2]] )  
#case 4        
    elif (r % 2) == 0 and (s % 2) != 0 and (t % 2) != 0:
        fA = (image[f1[0]][f1[1]][f1[2]] + image[f2[0]][f2[1]][f2[2]] - image[f3[0]][f3[1]][f3[2]] - image[f4[0]][f4[1]][f4[2]] 
            - image[f5[0]][f5[1]][f5[2]] - image[f6[0]][f6[1]][f6[2]] + image[f7[0]][f7[1]][f7[2]] + image[f8[0]][f8[1]][f8[2]] )    
#case 5
    elif (r % 2) != 0 and (s % 2) == 0 and (t % 2) == 0:
        fA = (- image[f1[0]][f1[1]][f1[2]] + image[f2[0]][f2[1]][f2[2]] + image[f3[0]][f3[1]][f3[2]] - image[f4[0]][f4[1]][f4[2]] 
              - image[f5[0]][f5[1]][f5[2]] + image[f6[0]][f6[1]][f6[2]] + image[f7[0]][f7[1]][f7[2]] - image[f8[0]][f8[1]][f8[2]] )    
#case 6 
    elif (r % 2) != 0 and (s % 2) == 0 and (t % 2) != 0:
        fA = (image[f1[0]][f1[1]][f1[2]] - image[f2[0]][f2[1]][f2[2]] - image[f3[0]][f3[1]][f3[2]] + image[f4[0]][f4[1]][f4[2]] 
            - image[f5[0]][f5[1]][f5[2]] + image[f6[0]][f6[1]][f6[2]] + image[f7[0]][f7[1]][f7[2]] - image[f8[0]][f8[1]][f8[2]] )    
#case 7    
    elif (r % 2) != 0 and (s % 2) != 0 and (t % 2) == 0:
        fA = (image[f1[0]][f1[1]][f1[2]] - image[f2[0]][f2[1]][f2[2]] + image[f3[0]][f3[1]][f3[2]] - image[f4[0]][f4[1]][f4[2]] 
            + image[f5[0]][f5[1]][f5[2]] - image[f6[0]][f6[1]][f6[2]] + image[f7[0]][f7[1]][f7[2]] - image[f8[0]][f8[1]][f8[2]] )
#case 8         
    elif (r % 2) != 0 and (s % 2) != 0 and (t % 2) != 0:
        fA = (- image[f1[0]][f1[1]][f1[2]] + image[f2[0]][f2[1]][f2[2]] - image[f3[0]][f3[1]][f3[2]] + image[f4[0]][f4[1]][f4[2]] 
              + image[f5[0]][f5[1]][f5[2]] - image[f6[0]][f6[1]][f6[2]] + image[f7[0]][f7[1]][f7[2]] - image[f8[0]][f8[1]][f8[2]] )
    
    return fA


def geometric_integrals(N_order,oper_x,oper_y,oper_z,Ncx,Ncy,Ncz):
                
    "Triple Integral"  
    
    Ir = []
    Is = []
    It = []
    
    def I(ind,Ncube):
        return ((2*ind)-1-Ncube)/(Ncube*sqrt(3))
    
    delta_x = I(2,Ncx) - I(1,Ncx)
    delta_y = I(2,Ncy) - I(1,Ncy)
    delta_z = I(2,Ncz) - I(1,Ncz)
    
    for r in range(0,N_order+1):
        I_temp=[]
        for i in range(1,oper_x+1):
            I_temp.append((((I(i,Ncx) + (delta_x/2)) ** (r+1)) - (((I(i,Ncx) - (delta_x/2)) ** (r+1)) ))/ (r+1))
        Ir.append(I_temp)
        
        I_temp=[]
        for i in range(1,oper_y+1):
            I_temp.append((((I(i,Ncy) + (delta_y/2)) ** (r+1)) - (((I(i,Ncy) - (delta_y/2)) ** (r+1)) ))/ (r+1))
        Is.append(I_temp)
        
        I_temp=[]
        for i in range(1,oper_z+1):
            I_temp.append((((I(i,Ncz) + (delta_z/2)) ** (r+1)) - (((I(i,Ncz) - (delta_z/2)) ** (r+1)) ))/ (r+1))
        It.append(I_temp)
        
    return Ir,Is,It


def calc_operator(Nxyz):
    if Nxyz % 2 == 0:
        operator = int(Nxyz/2)
    else:
        operator = int((Nxyz - 1)/2)
    return operator


def calculate_geometric(N_order,Ncx,Ncy,Ncz,image,r_seq,s_seq,t_seq):
        
    oper_x=calc_operator(Ncx)
    oper_y=calc_operator(Ncy)
    oper_z=calc_operator(Ncz)
    
    Ir,Is,It = geometric_integrals(N_order,oper_x,oper_y,oper_z,Ncx,Ncy,Ncz)
    table_Geo = np.zeros((N_order+1,N_order+1,N_order+1))
    
    for index in range(len(r_seq)):
        curr_lr = r_seq[index]
        curr_ls = s_seq[index]
        curr_lt = t_seq[index]

        for (r,s,t) in zip(curr_lr, curr_ls, curr_lt):
            if table_Geo[r][s][t] == 0: 
                G = 0
                for i in range(oper_x):
                    for j in range(oper_y):
                        for k in range(oper_z):
                            G += ( Ir[r][i] * Is[s][j] * It[t][k] * symmetry(i, j, k, Ncx-1,Ncy-1,Ncz-1, r, s, t, image))
                            
                table_Geo[r][s][t] = G
                
    return table_Geo


def zernike_descriptors(image, N_order, chi_coeff,r_seq,s_seq,t_seq):
    
    bound_img,Ncx,Ncy,Ncz= generate_bounding_image(image)    
    
    geometric_moments = calculate_geometric(N_order, Ncx,Ncy,Ncz,bound_img,r_seq,s_seq,t_seq)

    ZD3D=[]
    ZD3D_vectors=[]
    
    index = 0                
    for n in range(N_order+1):
        for l in range(0, n+1):
            if (n-l)% 2 == 0:
                
                "collecting moments into (2l +1)-dimensional vectors"
                omega_nl = np.zeros(((2*l)+1), dtype = complex)    # 
                i1 = 0
                for m in range(0,l+1):
                    omega = []
                    l_r = r_seq[index]
                    l_s = s_seq[index]
                    l_t = t_seq[index]
                    "For the r,s,t combination in the list Chi then store chi*geometric moment in omega"
                    for (r,s,t) in zip(l_r, l_s, l_t):

                            omega.append((conjugate(chi_coeff[index][r][s][t])) * geometric_moments[r][s][t]) 

                    "For the current omega nlm, sum all the omega r,s,t combos"
                    sum_omega = sum(omega)
                    
                    "Perform equation 13 to find the zernike moment omega nl"
                    omega_nlm = ( (3/(4 * np.pi)) * sum_omega) 
                    omega_nl[l+i1] = omega_nlm    
                    if m > 0:
                        "Symmetry rule for zernike moments of negative m"
                        omega_nl[l-i1] =  ((-1)**m) * conjugate(omega_nlm) 
                    
                    i1 = i1 + 1
                    index = index + 1
                ZD3D_vectors.append(omega_nl)
                
                "Store the zernike descriptor as norms of vectors omega nl"   
                acc= 0    
                for vector in omega_nl:
                    acc+= LA.norm(vector) 
                Fnl = sqrt(acc)
                ZD3D.append(round(Fnl,3))

    return ZD3D,ZD3D_vectors


def get_all_descriptors(labelled_image,num_lesions,all_lesion_indices,volumes,N_order,chi_coeff,r_sequence,s_sequence,t_sequence):
    
    All_descriptor_vectors = [0]* num_lesions
    All_zernike_descriptors = [0]* num_lesions
    
    for i in range(num_lesions):
        lab=i+1
        lesion_image = np.where(labelled_image==lab, np.ones_like(labelled_image), np.zeros_like(labelled_image))
        lesion_ZD3D,lesion_ZD3D_vectors = zernike_descriptors(lesion_image, N_order, chi_coeff,r_sequence,s_sequence,t_sequence)

        All_descriptor_vectors[i] = lesion_ZD3D_vectors
        All_zernike_descriptors[i] = lesion_ZD3D
        print(lab)
            
    return All_descriptor_vectors, All_zernike_descriptors     


def calc_volume(_list):
    vol=[]
    for i in range(len(_list)):
        vol.append(len(_list[i][0]))
    return vol
    
def main(argv):
    parser = argparse.ArgumentParser()
       
    parser.add_argument('-i', dest='les_seg', required=True)
    parser.add_argument('-neigh', dest='neighbourhood', default=1, choices=[1, 2, 3], 
                        action='store', type=int,
                        help='type of neighborhood applied when creating the '
                             'connected component structure')
    parser.add_argument('-n', type=int, dest='Order_of_moments', default=10)
    
    args = parser.parse_args()
    
    print(args.les_seg, args.neighbourhood,args.Order_of_moments)
    
    les_file = args.les_seg
    les_img = nib.load(les_file).get_fdata()
    
    filename = les_file.split('/')[-1].split('.nii.gz')[0]
    image_name = les_file.split('/')[-1]
    
    order=args.Order_of_moments 
    
    sc = structurecomponent(args.neighbourhood)  
    
    label_image, n_objects = label(les_img, sc)  
    
    lesion_indices, coms = get_all_indices(label_image, n_objects)
    volumes = calc_volume(lesion_indices)
    
    coefficients,r_sequence,s_sequence,t_sequence = get_coefficients(order)
    
    ZD3D_vectors,ZD3D = get_all_descriptors(label_image,n_objects,lesion_indices,volumes,order,coefficients,r_sequence,s_sequence,t_sequence)
    
    rows=[]
    for i in range(n_objects):
        rows.append([image_name, i+1, round(coms[i][0]), round(coms[i][1]), round(coms[i][2]), volumes[i], ZD3D[i][0], ZD3D[i][1], ZD3D[i][2],ZD3D[i][3], ZD3D[i][4],ZD3D[i][5], ZD3D[i][6],ZD3D[i][7],ZD3D[i][8],ZD3D[i][9],ZD3D[i][10], ZD3D[i][11], ZD3D[i][12],ZD3D[i][13],ZD3D[i][14],ZD3D[i][15],ZD3D[i][16],ZD3D[i][17],ZD3D[i][18],ZD3D[i][19],ZD3D[i][20],ZD3D[i][21],ZD3D[i][22],ZD3D[i][23], ZD3D[i][24],ZD3D[i][25], ZD3D[i][26],ZD3D[i][27],ZD3D[i][28], ZD3D[i][29],ZD3D[i][30], ZD3D[i][31], ZD3D[i][32],ZD3D[i][33],ZD3D[i][34],ZD3D[i][35]])
        
    headers = [["Image","Label", "CoM_x","CoM_y","CoM_z", "Volume","ZD 0,0", "ZD 1,1", "ZD 2,0","ZD 2,2","ZD 3,1","ZD 3,3","ZD 4,0","ZD 4,2","ZD 4,4","ZD 5,1","ZD 5,3","ZD 5,5","ZD 6,0", "ZD 6,2", "ZD 6,4", "ZD 6,6", "ZD 7,1", "ZD 7,3", "ZD 7,5", "ZD 7,7", "ZD 8,0","ZD 8,2","ZD 8,4","ZD 8,6", "ZD 8,8", "ZD 9,1", "ZD 9,3", "ZD 9,5", "ZD 9,7", "ZD 9,9", "ZD 10,0","ZD 10,2","ZD 10,4","ZD 10,6", "ZD 10,8", "ZD 10,10"]]
    with open('ZD3D_{}.csv'.format(filename), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(headers)
        writer.writerows(rows)
        
if __name__ == "__main__":
    main(sys.argv[1:])    
    
    

                        
