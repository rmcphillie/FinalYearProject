import sys
import numpy as np
import nibabel as nib
import argparse
from properties import Image, Lesion, Ventricles
import csv


def main(argv):
    parser = argparse.ArgumentParser()
       
    parser.add_argument('-i', dest='les_seg', required=True)
    parser.add_argument('-v', dest='ventr_seg', default=None )
    parser.add_argument('-min', type=int, dest='minimum_border', default=25, 
                       help= 'minimum percentage of border which is in contact with the ventricles to be excluded from classify.csv')
    parser.add_argument('-neigh', dest='neighbourhood', default=1, choices=[1, 2, 3], 
                        action='store', type=int,
                        help='type of neighborhood applied when creating the '
                             'connected component structure')
    parser.add_argument('-bin', dest='binary_structure', default=1, choices=[1,2,3],
                        action='store', type=int,
                        help='the type of binary structure used to dilate and erode structures')
    
    args = parser.parse_args()
    
    
    print(args.les_seg, args.ventr_seg, args.minimum_border, args.neighbourhood, args.binary_structure)
    
    les_file = args.les_seg
    ventr_file = args.ventr_seg
    ventr_img = nib.load(ventr_file).get_fdata()  
    les_img = nib.load(les_file).get_fdata()
    
    filename = les_file.split('/')[-1].split('.nii.gz')[0]
    image_name = les_file.split('/')[-1]
    
    Ventr_bord_indices = Ventricles(ventr_img).border_indices()
    
    Image_object = Image(les_img, args.neighbourhood, args.binary_structure, Ventr_bord_indices)
    num_lesions = Image_object.n_objects
    
    list_lesions = []
    for i in range(1,num_lesions+1):
        list_lesions.append(Lesion(i,Image_object))

    
    periv_headers = [["Label", "No. of border voxels touching ventricles", "Percentage of lesion border touching ventricles (%)", "Volume (mm3)"]]
    
    periv_rows = []
    for i in range(len(list_lesions)):
        if list_lesions[i].periv_bord >0:
            periv_rows.append([list_lesions[i].label, list_lesions[i].periv_bord_volume, list_lesions[i].periv_bord, list_lesions[i].volume])
        
    with open('periventricular_lesions_{}.csv'.format(filename), 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerows(periv_headers)
        writer.writerows(periv_rows)
    
    classify_headers = [["Image","Label" , "Volume", "Ero - Does it split?", "Ero - No. components after split", "Ero - Vol of Largest component after split", "Dil - Does it merge?", "Dil - No. of components which merge","Dil - Vol of merged components", "Core component vol", "Component Vol / Core Vol", "Coalescence score"]]
    classify_rows = []
    for i in range(len(list_lesions)):
        if list_lesions[i].periv_bord < args.minimum_volume:
            classify_rows.append([image_name, list_lesions[i].label, list_lesions[i].volume, list_lesions[i].erosion_result(), list_lesions[i].num_split, list_lesions[i].largest_eroded_component, list_lesions[i].dilation_result(), list_lesions[i].num_merges, list_lesions[i].merge_volume, list_lesions[i].cc_volume, list_lesions[i].cc_fraction, list_lesions[i].coalescence_score])
    
    with open('Classify_{}.csv'.format(filename), 'w', newline='') as file2:
        writer = csv.writer(file2)
        writer.writerows(classify_headers)
        writer.writerows(classify_rows)
    
    
    


if __name__ == "__main__":
    main(sys.argv[1:])
