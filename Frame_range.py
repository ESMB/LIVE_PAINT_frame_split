#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 18:30:37 2021

@author: Mathew
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import frc
from sklearn.cluster import DBSCAN
from PIL import Image
# Camera settings
Pixel_size=103.0
    

# Settings
image_width=512
image_height=512
scale=8
photon_adu = 0.0265/0.96

# Number of frames to split the images into
frame_range_to_plot=1000

# Thresholds
prec_thresh=25

filename_contains="FitResults.txt"

# Folders to analyse:
root_path=r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/"

# Cluterinng
            
to_cluster=1
eps_threshold=1
minimum_locs_threshold=100


pathList=[]



pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_100ng_Tom20_101B_4")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_100ng_Tom20_101B_3")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_100ng_Tom20_101B_2")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_500ng_Tom20_101B_5")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_500ng_Tom20_101B_2")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_500ng_Tom20_101B_3")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_500ng_Tom20_101B_4")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_100ng_Tom20_101B_1")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_500ng_Tom20_101B_8")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_500ng_Tom20_101B_1")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_500ng_Tom20_101B_6")
pathList.append(r"/Users/Mathew/Documents/Current analysis/20221117_wash_post_PDL_transfection_concs_and_df_results/Nd_2_dox_1_5_ngmL_500ng_Tom20_101B_7")

resolution=[]
clus_resolution=[]
mean_precision=[]
mean_signal=[]
mean_SBR=[]

#  Generate SR image (points)
def generate_SR(coords):
    SR_plot_def=np.zeros((image_width*scale,image_height*scale),dtype=float)
    j=0
    for i in coords:
        
        xcoord=coords[j,0]
        ycoord=coords[j,1]
        scale_xcoord=round(xcoord*scale)
        scale_ycoord=round(ycoord*scale)
        # if(scale_xcoord<image_width and scale_ycoord<image_height):
        SR_plot_def[scale_ycoord,scale_xcoord]+=1
        
        j+=1
    return SR_plot_def

def generate_SR_range(coords,num):
    SR_plot_def=np.zeros((image_width*scale,image_height*scale),dtype=float)
    SR_plot_num=np.zeros((image_width*scale,image_height*scale),dtype=float)
    j=0
    for i in coords:
        
        xcoord=coords[j,0]
        ycoord=coords[j,1]
        scale_xcoord=round(xcoord*scale)
        scale_ycoord=round(ycoord*scale)
        # if(scale_xcoord<image_width and scale_ycoord<image_height):
        SR_plot_def[scale_ycoord,scale_xcoord]+=1
        SR_plot_num[scale_ycoord,scale_xcoord]+=num
        j+=1
    
    SR_norm=np.divide(SR_plot_num,SR_plot_def,where=SR_plot_def!=0)
        
        
    return SR_norm,SR_plot_def

def SRGaussian(size, fwhm, center):

    sizex=size[0]
    sizey=size[1]
    x = np.arange(0, sizex, 1, float)
    y = x[0:sizey,np.newaxis]
    # y = x[:,np.newaxis]


    x0 = center[0]
    y0 = center[1]
    
    wx=fwhm[0]
    wy=fwhm[1]
    
    return np.exp(-0.5 * (np.square(x-x0)/np.square(wx) + np.square(y-y0)/np.square(wy)) )

def gkern(l,sigx,sigy):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx)/np.square(sigx) + np.square(yy)/np.square(sigy)) )
    # print(np.sum(kernel))
    # test=kernel/np.max(kernel)
    # print(test.max())
    return kernel/np.sum(kernel)


def generate_SR_prec(coords,precsx,precsy):
    box_size=20
    SR_prec_plot_def=np.zeros((image_width*scale,image_height*scale),dtype=float)
    dims=np.shape(SR_prec_plot_def)
    print(dims)
    j=0
    for i in coords:

      
        precisionx=precsx[j]/Pixel_size*scale
        precisiony=precsy[j]/Pixel_size*scale
        xcoord=coords[j,0]
        ycoord=coords[j,1]
        scale_xcoord=round(xcoord*scale)
        scale_ycoord=round(ycoord*scale)
        
        
        
        sigmax=precisionx
        sigmay=precisiony
        
        
        # tempgauss=SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
        
        tempgauss=gkern(2*box_size,sigmax,sigmay)
        
        # SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
        
        
        
        ybox_min=scale_ycoord-box_size
        ybox_max=scale_ycoord+box_size
        xbox_min=scale_xcoord-box_size
        xbox_max=scale_xcoord+box_size 
        
        
        if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
            SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss
        
        
           
        j+=1
    
    return SR_prec_plot_def

def cluster(coords):
     db = DBSCAN(eps=eps_threshold, min_samples=minimum_locs_threshold).fit(coords)
     labels = db.labels_
     n_clusters_ = len(set(labels)) - (1 if-1 in labels else 0)  # This is to calculate the number of clusters.
     print('Estimated number of clusters: %d' % n_clusters_)
     return labels
 
def generate_SR_cluster(coords,clusters):
    SR_plot_def=np.zeros((image_width*scale,image_height*scale),dtype=float)
    j=0
    for i in coords:
        if clusters[j]>-1:
            xcoord=coords[j,0]
            ycoord=coords[j,1]
            scale_xcoord=round(xcoord*scale)
            scale_ycoord=round(ycoord*scale)
            # if(scale_xcoord<image_width and scale_ycoord<image_height):
            SR_plot_def[scale_ycoord,scale_xcoord]+=1
        
        j+=1
    return SR_plot_def

def resolution_per_frame(frames,total):
    res=[]
    frame_range=[]
    length=int(total/frames)
    for i in range(1,length):
        framerem=i*frames
        print(framerem)
        loc_data = pd.read_table(fits_path)

        index_names = loc_data[loc_data['Frame']>framerem].index
        loc_data.drop(index_names, inplace = True)
       
           

        # Extract useful data:
        coords = np.array(list(zip(loc_data['X'],loc_data['Y'])))
    
            
        # Generate points SR (ESMB method):
        img=generate_SR(coords)
        
       
        img = frc.util.square_image(img, add_padding=False)
        img = frc.util.apply_tukey(img)
        # Apply 1FRC technique
        frc_curve = frc.one_frc(img)
        
        img_size = img.shape[0]
        xs_pix = np.arange(len(frc_curve)) / img_size
        
        xs_nm_freq = xs_pix * scale_frc
        frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
        
        res.append(frc_res)
        frame_range.append(framerem)
    plt.plot(frame_range,res)
    plt.xlabel('Frames',size=20)
    plt.ylabel('Resolution (nm)',size=20)
    plt.show()
    return res,frame_range
    

for path in pathList:
    print(path)
    path=path+"/"

    # Perform the fitting

    # Load the fits:
    for root, dirs, files in os.walk(path):
                for name in files:
                        if filename_contains in name:
                            if ".txt" in name:
                                if ".tif" not in name:
                                    resultsname = name
                                    print(resultsname)
    
                                    fits_path=path+resultsname
                                    # fits_path=path+filename_contains
                                    
                                    
                                    loc_data = pd.read_table(fits_path)
                                    
                                    index_names = loc_data[loc_data['Precision (nm)']>prec_thresh].index
                                    loc_data.drop(index_names, inplace = True)
                                   
                                       
  
                          
                                  
             
                                    # Extract useful data:
                                    coords = np.array(list(zip(loc_data['X'],loc_data['Y'])))
                                    precsx= np.array(loc_data['Precision (nm)'])
                                    precsy= np.array(loc_data['Precision (nm)'])
                                    xcoords=np.array(loc_data['X'])
                                    ycoords=np.array(loc_data['Y'])
                                    signal=np.array(loc_data['Signal'])
                                    background=np.array(loc_data['Background'])
                                    
                                    
                                    precs_nm=precsx
                                    
                                    signal_above_background=(signal-background)*photon_adu
                                    ave_signal= signal_above_background.mean()
                                    signal_bg_ratio=signal/background
                                    ave_sbr=signal_bg_ratio.mean()
                                    
                                    average_precision=precs_nm.mean()
                                    
                                    plt.hist(precs_nm, bins = 50,range=[0,100], rwidth=0.9,color='#ff0000')
                                    plt.xlabel('Precision (nm)',size=20)
                                    plt.ylabel('Number of Features',size=20)
                                    plt.title('Localisation precision',size=20)
                                    plt.savefig(path+"Precision.pdf")
                                    plt.show()
                                    
                                    # Generate points SR (ESMB method):
                                    img=generate_SR(coords)
                                    
                                    
                              
                                    
                                    scale_frc = scale/Pixel_size

                                    img = frc.util.square_image(img, add_padding=False)
                                    img = frc.util.apply_tukey(img)
                                    # Apply 1FRC technique
                                    frc_curve = frc.one_frc(img)
                                    
                                    img_size = img.shape[0]
                                    xs_pix = np.arange(len(frc_curve)) / img_size
                                    
                                    xs_nm_freq = xs_pix * scale_frc
                                    frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
                                    
                                    text='Resolution = '+str(round(frc_res,2))+' nm'
                                    plt.plot(xs_nm_freq, thres(xs_nm_freq))
                                    plt.plot(xs_nm_freq, frc_curve)
                                    plt.xlabel('Spatial resolution (nm$^{-1}$)',size=20)
                                    plt.ylabel('FRC',size=20)
                                    
                                    plt.title(text,size=12)
                                    plt.savefig(path+"Resolution.pdf")
                                    plt.show()
                                    
                                    print(frc_res)
                                    
                                    if to_cluster==1:
                                        clusters=cluster(coords)
                                    
                                        # Check how many localisations per cluster
                                     
                                        cluster_list=clusters.tolist()    # Need to convert the dataframe into a list- so that we can use the count() function. 
                                        maximum=max(cluster_list)+1  
                                        
                                        
                                        cluster_contents=[]         # Make a list to store the number of clusters in
                                        
                                        for i in range(0,maximum):
                                            n=cluster_list.count(i)     # Count the number of times that the cluster number i is observed
                                           
                                            cluster_contents.append(n)  # Add to the list. 
                                        
                                        if len(cluster_contents)>0:
                                            average_locs=sum(cluster_contents)/len(cluster_contents)
                                     
                                            
                                            cluster_arr=np.array(cluster_contents)
                                        
                                            median_locs=np.median(cluster_arr)
                                            mean_locs=cluster_arr.mean()
                                            std_locs=cluster_arr.std()
                                            
                                        
                                            # Generate the SR image.
                                            SR_img=generate_SR_cluster(coords,clusters)
                                            
                                          
                                            img = frc.util.square_image(SR_img, add_padding=False)
                                            img = frc.util.apply_tukey(SR_img)
                                            # Apply 1FRC technique
                                            frc_curve = frc.one_frc(img)
                                            
                                            img_size = img.shape[0]
                                            xs_pix = np.arange(len(frc_curve)) / img_size
                                            
                                            xs_nm_freq = xs_pix * scale_frc
                                            clu_frc_res, clu_res_y, clu_thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
                                            
                                            textclus='Resolution = '+str(round(clu_frc_res,2))+' nm'
                                            plt.plot(xs_nm_freq, thres(xs_nm_freq))
                                            plt.plot(xs_nm_freq, frc_curve)
                                            plt.xlabel('Spatial resolution (nm$^{-1}$)',size=20)
                                            plt.ylabel('FRC',size=20)
                                            
                                            plt.title(textclus,size=12)
                                            plt.savefig(path+"Cluster_Resolution.pdf")
                                            plt.show()
                                            
                                            
                                            print(clu_frc_res)
                                    
                                    # Generate images and stats every x number of frames:
                                    
                                    locs_per_range=[]
                                    res_per_range=[]
                                    range_x=[]
                                        
                                    frames=loc_data['Frame']
                                    number_of_frames=frames.max()
                                    
                                    SR_plot_all=np.zeros((image_width*scale,image_height*scale),dtype=float)
                                    SR_plot_number=np.zeros((image_width*scale,image_height*scale),dtype=float)
                                    
                                    
                                    for i in range(0,number_of_frames,frame_range_to_plot):
                                        frame_low=i
                                        frame_high=i+frame_range_to_plot
                                        # print(frame_low,frame_high)
                                    
                                        
                                        loc_data_range=loc_data.copy()
                                        index_names = loc_data_range[loc_data_range['Precision (nm)']>prec_thresh].index
                                        loc_data_range.drop(index_names, inplace = True)
                                   
                                        index_names = loc_data_range[loc_data_range['Frame']<frame_low].index
                                        loc_data_range.drop(index_names, inplace = True)
  
                                        index_names = loc_data_range[loc_data_range['Frame']>frame_high].index
                                        loc_data_range.drop(index_names, inplace = True)
  
                                        frame_range=loc_data_range['Frame']
                 
                                        # Extract useful data from frame range
                                        coords_range = np.array(list(zip(loc_data_range['X'],loc_data_range['Y'])))
                                        precsx_range= np.array(loc_data_range['Precision (nm)'])
                                        precsy_range= np.array(loc_data_range['Precision (nm)'])
                                        xcoords_range=np.array(loc_data_range['X'])
                                        ycoords_range=np.array(loc_data_range['Y'])
                                        signal_range=np.array(loc_data_range['Signal'])
                                        background_range=np.array(loc_data_range['Background'])
                                        
                                        
                                        
                                        img_number,img_range=generate_SR_range(coords_range,i)
                                        
                                        SR_plot_all=SR_plot_all+img_number
                                        sr_to_add=SR_plot_number==0
                                        
                                        SR_plot_number=img_number+SR_plot_number
                                            
                                        ims = Image.fromarray(img_range)
                                        ims.save(path+str(i)+'_SR_points.tif')
                                        
                                        prec_range=generate_SR_prec(coords_range,precsx_range,precsy_range)
                                        ims = Image.fromarray(prec_range)
                                        ims.save(path+'Prec_'+str(i)+'_SR.tif')
                                        
                                        # ims = Image.fromarray(img_number)
                                        # ims.save(path+str(i)+'_SR_points_number.tif')
                                        
                                        scale_frc = scale/Pixel_size
    
                                        img = frc.util.square_image(img_range, add_padding=False)
                                        img = frc.util.apply_tukey(img_range)
                                        # Apply 1FRC technique
                                        frc_curve_range = frc.one_frc(img_range)
                                        
                                        img_size = img_range.shape[0]
                                        xs_pix = np.arange(len(frc_curve)) / img_size
                                        
                                        xs_nm_freq = xs_pix * scale_frc
                                        frc_res_range, res_y_range, thres_range = frc.frc_res(xs_nm_freq, frc_curve_range, img_size)
                                        
                                                                              
                                        res_per_range.append(frc_res_range)
                                        locs_per_range.append(len(frame_range))
                                        range_x.append(i)
                                    
                                    plt.plot(range_x, locs_per_range)
                                    plt.xlabel('Frame start',size=20)
                                    plt.ylabel('Number of localisations',size=20)
                                    plt.savefig(path+"Number_of_locs_per_frame.pdf")
                                    plt.show()
                                    
                                    plt.plot(range_x, res_per_range)
                                    plt.xlabel('Frame start',size=20)
                                    plt.ylabel('Resolution (nm)',size=20)
                                    plt.savefig(path+"Resolution_per_frame.pdf")
                                    plt.show()
                                    
                                    ims = Image.fromarray(SR_plot_number)
                                    ims.save(path+'SR_number.tif')
                                    
                                    resolution.append(frc_res)   
                                    # clus_resolution.append(clu_frc_res)
                                    mean_precision.append(average_precision)
                                    mean_signal.append(ave_signal)
                                    mean_SBR.append(ave_sbr)
                                    
                                    df = pd.DataFrame(list(zip(resolution,clus_resolution,mean_precision,mean_signal,mean_SBR)),columns =['Resolution', 'Custered Resolution','Precision','Signal','SBR'])
                                    df.to_csv(root_path+ 'Resolution.csv', sep = '\t')
                                        
                                    
                                    
                                    
