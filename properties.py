from scipy import ndimage 
import numpy as np
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure


class Ventricles(object):
    def __init__(self, ventr_seg):
        self.ventr_seg = ventr_seg
        
    def border_indices(self):
        bin_struct = generate_binary_structure(3,1)
        dil = ndimage.binary_dilation(self.ventr_seg, structure=bin_struct).astype(self.ventr_seg.dtype)
        binary_seg = np.asarray(self.ventr_seg, dtype=np.int8)
            
        #subtract the full segment from the original segment to obtain the external border
        external_border = dil - binary_seg 
        #return the external border indices
        return np.where(external_border ==1)

    

    
class Image(object):
    def __init__ (self,image, neigh, binary_number, ventr_border):
        self.les_image = image
        self.neigh= neigh
        self.bin_num = binary_number
        self.bin_struct = self.binary_structure()
        self.sc = self.structure_component()
        self.ventr_border = ventr_border
        self.label_image, self.n_objects = self.original_label()
        self.label_dil, self.n_dil = self.dilated_label()
        
        self.orig_indices, self.int_bord_indices, self.ext_bord_indices = self.get_all_indices()
        self.dil_indices = self.get_indices(self.label_dil, self.n_dil)

        self.superobject = self.track_dil()

        
    
    def structure_component(self):
        if self.neigh == 1:
            sc = np.array([[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,1,1],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]]]) 
        elif self.neigh ==2:
            sc = np.array([[[0,1,0],[1,1,1],[0,1,0]], [[1,1,1],[1,1,1],[1,1,1]], [[0,1,0],[1,1,1],[0,1,0]]])
        elif self.neigh ==3:
            sc = np.array([[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])
        return sc

    def binary_structure(self):
        if self.bin_num == 1:
            structure = generate_binary_structure(3,1)
        elif self.bin_num ==2:
            structure = generate_binary_structure(3,2)
        elif self.bin_num ==3:
            structure = generate_binary_structure(3,2)
        return structure
    
    
    def original_label(self):
        l,n = label(self.les_image, self.sc)
        return l,n
    
    def dilated_label(self):
        dilated = ndimage.binary_dilation(self.les_image, structure=self.bin_struct).astype(self.les_image.dtype) 
        l,n = label(dilated, self.sc)
        return l,n

    def eroded_label(self):
        eroded = ndimage.binary_erosion(self.les_image, structure=self.bin_struct).astype(self.les_image.dtype) 
        l,n = label(eroded, self.sc)
        return l,n 
        
    def get_indices(self,l,n):
        "Function to store each labelled component's indices"
        indices = [] #initialize list to store indices
        for lab in range(1,n+1):        #for each object in the range of objects found
            #where a labelled object is the same as the current item in the loop, make the segment all equal to 1, and where it is not the same make the background voxels equal to zero 
            seg_lab = np.where(l==lab, np.ones_like(1), np.zeros_like(1))   
            
            #where the segment array is the same as 1, retrieve the indices
            indices_lab = np.where(seg_lab==1)    
            
            #add the indices to the list of indices          
            indices.append(indices_lab)            
        
        return indices
    
    def get_all_indices(self):
        l = self.label_image
        int_indices = []
        ext_indices = []
        normal_indices = []
        for lab in range(1,self.n_objects+1):        #for each object in the range of objects found
            #where a labelled object is the same as the current item in the loop, make the segment all equal to 1, and where it is not the same make the background voxels equal to zero 
            lesion_seg = np.where(l==lab, np.ones_like(1), np.zeros_like(1))   
            normal_indices.append(np.where(lesion_seg==1))

            ero = ndimage.binary_erosion(lesion_seg, structure=self.bin_struct).astype(lesion_seg.dtype)
            dil = ndimage.binary_dilation(lesion_seg, structure=self.bin_struct).astype(lesion_seg.dtype)
            binary_seg = np.asarray(lesion_seg, dtype=np.int8)
            
            internal_border = binary_seg - ero
            external_border = dil - binary_seg 
            
            int_indices.append(np.where(internal_border ==1))
            ext_indices.append(np.where(external_border ==1))
   
        return normal_indices, int_indices, ext_indices
    
    def track_dil(self):
        "function that maps which new components, the original components form when dilated"
        track_dilation = []
        superobject = []
        
        for index in range(len(self.orig_indices)):
            r = self.orig_indices[index]
            matchfound = False
            for j in range(len(self.dil_indices)):
                if matchfound == True:
                    break
                else:
                    r2 = self.dil_indices[j]
                    for i in range(len(r2[0])):
                        #if the indices of the original object are present in the current dilated component, then that object was its origin 
                        if r[0][0] == r2[0][i] and r[1][0] == r2[1][i] and r[2][0] == r2[2][i]: 
                            track_dilation.append(j+1) 
                            matchfound = True
                            break
                            
        #loops through the track_dilation list and maps out which component merge and those that don't merge with any
        for i in range(len(track_dilation)):
            current_merge = []
            for j in range(len(track_dilation)):
                if track_dilation[i] == track_dilation[j]: 
                    current_merge.append(j+1)
            superobject.append(current_merge)
            
        
        return superobject
           
    
    def object_indices(self,n):
        return self.orig_indices[n-1]
    
    def object_int_border_indices(self,n):
        return self.int_bord_indices[n-1]
    
    def object_ext_border_indices(self,n):
        return self.ext_bord_indices[n-1]
    
    def object_merges(self,n):
        return self.superobject[n-1]
    
    def object_merge_volumes(self,n):
        merge_list = self.superobject[n-1]
        merge_vol_list = []
        "self.voxel_dim = voxel_dim" #factor which i multiply each voxel by
        for i in merge_list:
            merge_vol_list.append(len(self.orig_indices[i-1][0]))
        
        return merge_vol_list
            
    def object_eroded_indices(self,n):
        
        l = self.label_image
        les_seg = np.where(l==n, np.ones_like(l), np.zeros_like(l))
        eroded = ndimage.binary_erosion(les_seg, structure=self.bin_struct).astype(les_seg.dtype)
        l,n = label(eroded,self.sc)
        eroded_indices = [] 
        
        for lab in range(1,n+1):
            seg_lab = np.where(l==lab, np.ones_like(l), np.zeros_like(l))   
            
            #where the segment array is the same as 1, retrieve the indices
            indices_lab = np.where(seg_lab==1)    
            
            #add the indices to the list of indices          
            eroded_indices.append(indices_lab)            
        
        return eroded_indices, n

    
    def periv_bord_indices(self,n):
        "Function to find the peri-ventricular lesions and their indices that touch the border"   
        les_border = self.int_bord_indices[n-1]
        
        external_indices = self.ventr_border
        
        x = []                      #create empty lists so the three index 
        y = []                      #values of each boundary voxel can be appended
        z = []
    
        for i in range(len(les_border[0])):  
            if self.check(les_border[0][i], les_border[1][i], les_border[2][i], external_indices) == True:   
                x.append(les_border[0][i])           
                y.append(les_border[1][i])
                z.append(les_border[2][i])
                                                       
        #preparing the boundary indices in a three list of tuples
        new_tuple = (x),(y),(z)     
        
        #convert the list of boundary indices into a tuple of three arrays
        periv_indices = tuple(map(np.array, new_tuple)) 
            
        return periv_indices
           
    def check(self,x, y, z, indices):         
            "Function to check if border indices of the lesions overlap with the ventricle border"
            r2 = indices
            
            for index in range(len(r2[0])):
                matchfound = False 
                
                #loop through the segment indices to find any matches with the current labelled lesion indices
                if r2[0][index] == x and r2[1][index] == y and r2[2][index] == z:
                    matchfound = True                  
                if matchfound == True:      
                    return matchfound   
    
        
    
    
    
class Lesion(object):
    def __init__(self, label, imageobj):
        self.label = label
        self.imageobj = imageobj
        
        self.lesion_indices = self.get_indices()
        self.eroded_indices, self.num_eroded = self.imageobj.object_eroded_indices(self.label)
        self.merged_volumes = self.get_merged_volumes()
        self.merge_volume = self.total_merge_volume()
        self.list_merges = self.get_list_merges()
        
        self.periv_bord_indices = self.get_periv_bord_indices()
        self.int_bord_indices = self.get_internal_border_indices()
        self.ext_bord_indices = self.get_external_border_indices()
        
        self.volume = self.calc_volume()
        
        self.num_merges = self.calc_num_merges()
        self.num_split = self.calc_num_split()
        self.periv_bord_volume = self.calc_periv_bord_volume()
        self.periv_bord = self.calc_periv_bord()
        self.largest_eroded_component = self.calc_largest_split()
        self.cc_volume = self.calc_core_volume()
        self.cc_fraction = self.calc_core_fraction()
        
        self.coalescence_score = self.calc_coalescence_score()
        
    def get_indices(self):
        return self.imageobj.object_indices(self.label)
    
    def get_internal_border_indices(self):
        return self.imageobj.object_int_border_indices(self.label)
    
    def get_external_border_indices(self):
        return self.imageobj.object_ext_border_indices(self.label)
    
    
    def calc_volume(self):
        return len(self.lesion_indices[0])
    
    def dilation_result(self):
        if self.num_merges ==1:
            return "No" 
        else:
            return "Yes"
        
        
    def get_list_merges(self):
        return self.imageobj.object_merges(self.label)
    
    def calc_num_merges(self):
        return len(self.list_merges)
    
    def get_merged_volumes(self):
        return self.imageobj.object_merge_volumes(self.label)
    
    def total_merge_volume(self):
        return sum(self.get_merged_volumes())
    
    
    def calc_core_volume(self):
       core_vol = 0
       if len(self.merged_volumes) == 1:
           return 0
       else:
           for i in self.merged_volumes:
               if i > core_vol:
                   core_vol = i
           return core_vol
    
    def calc_core_fraction(self):
        if self.cc_volume == 0:
            return 0
        else:
            return round(self.volume/self.cc_volume,2)
    
    def erosion_result(self):
        if self.num_eroded==0:
            return "Disappears"
        elif self.num_eroded == 1:
            return "Reduces"
        else:
            return "Yes"
    
    def calc_num_split(self):
        if self.num_eroded==0 or self.num_eroded==1:
            return 0
        else:
            return self.num_eroded
        
    def calc_largest_split(self):
        if self.num_eroded==0 or self.num_eroded==1:
            return 0
        else:
            poss_large = []
            for i in range(0,self.num_eroded):
                poss_large.append(len(self.eroded_indices[i][0]))        
            return max(poss_large)

    def get_periv_bord_indices(self):
        return self.imageobj.periv_bord_indices(self.label)           

    def calc_periv_bord_volume(self):
        return len(self.periv_bord_indices[0])
    
    def calc_periv_bord(self):
        """Function to compute the overall border volume of an identified periventricular lesion
            and the percentage of the border which is touching the ventricles"""
                
        return round(((len(self.periv_bord_indices[0]) / len(self.int_bord_indices[0])) * 100),2)
    
    
    def calc_coalescence_score(self):
        "Function to define the coalescence score of lesion object"
        merge = self.dilation_result()
        vol = self.volume

        core_vol = self.cc_volume
        largest_split = self.largest_eroded_component
        
        split = self.erosion_result()
               
        
        #1
        if split == "Disappears" and merge == "No": 
            return 1

        #2
        elif split == "Disappears" and merge == "Yes": 
            return 2
                    
        #4,5,6,7
        elif vol <= 80:
            if (split=="Reduces") and (merge=="No"):
                return 3
            elif (split == "Reduces") and (merge == "Yes"):  
                return 4        
            elif (split == "Yes") and (merge == "No"):  
                return 5   
            elif (split == "Yes") and (merge == "Yes"):
                return 6
                
        #8-13
        elif (vol > 80 and vol < 500):
            if type(largest_split) == int:
                if (largest_split < 0.2 * vol) and (merge == "No" or vol <= 0.2 * core_vol):
                    return 7
                                    
                elif (largest_split < 0.2 * vol) and (vol > 0.2 * core_vol):
                    return 8 
                      
                elif (split == "Reduces" or largest_split >= 0.2 * vol) and (merge == "No" or vol <= 0.2 * core_vol):
                    return 9
                                   
                elif (split == "Reduces" or largest_split >= 0.2 * vol) and (vol > 0.2 * core_vol):
                    return 10
        
        elif (vol >= 500  and vol < 1500) and largest_split < 0.2 * vol:  
            return 11 
       
        elif (vol >= 500 and vol < 1500)  and (split == "Reduces" or largest_split >= 0.2 * vol):  
            return 12
            
        #13
        elif vol >= 1500  and largest_split < 0.2 * vol:  
            return 13     
            
        #14
        elif vol >= 1500  and (split == "Reduces" or largest_split >= 0.2 * vol):  
            return 14 

    
    
    
        