import cv2
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def convert_py_loc(loc):                                                           #Function converting user input to a readable path.
    loc = loc.replace('\\','\\\\')
    return loc

def kmeans_application(img):                                                       #Function to apply K-means clustering algorithm to the image.  
    ui_img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                              #Converting RGB to BGR.
    ui_img_km = img.reshape((ui_img_bgr.shape[0] * ui_img_bgr.shape[1], 3))        #Changing the shape to fit the array in the K-means cluster algorithm.

    n_cluster = range(1,24)                                                              #Number of clusters ranging from 1 to 24 is selected.
    kmeans = [KMeans(n_clusters = i, n_init='auto').fit(ui_img_km) for i in n_cluster]   #Kmeans algorithm is executed recursively with upto 24 clusters.
    return kmeans, ui_img_km

def best_cluster(K_Means,km_img):                                                                # Function returning best number of clustes.
    km_scores = [K_Means[i].score(km_img) for i in range(len(K_Means))]                  #Score is assigned to each cluster.
    km_scores_np = np.array(km_scores)                                                    #Scores are stored in a numpy array.
    km_scores_differ = km_scores_np/km_scores_np[0]
    km_dif_scores = np.diff(km_scores_differ)
    best_clusters = np.argwhere(km_dif_scores < np.quantile(km_dif_scores,0.84))[-1][0]   #Based on the Elbow Method, the ideal number of clusters is selected (Best number of colours in the Pallete).
    return best_clusters


def pallete(img_cen_rgb):                                                                          #Function for pallete generation.
    pat = np.zeros((50, 50, 3), np.uint8)
    patx = np.zeros((50, 50, 3), np.uint8)                                                         #Dummy numpy array generated of size 50 X 50 to build the palette.
    for i in range(0,len(img_cen_rgb)):
        if i == 0:
            patx[:] = img_cen_rgb[i]
        else:
            pat[:] = img_cen_rgb[i]
            patx=np.concatenate((patx[:],pat[:]),axis =1)                                              #Used to horizontally concatenate the pallete colours.
    return patx   
    
    
if __name__=='__main__':
    ui_img_loc = input('Please provide the location of UI form image:\n')   #Format of path (should be without quotes, example:  C:\Users\Downloads\sample.png
    ui_img_loc_p = convert_py_loc(ui_img_loc)                               #Converting user input to Python readable path
    ui_img = cv2.imread(ui_img_loc_p,1)                                     #reading the UI Image, where the second parameter 1 stands for colour image reading.
    ui_img1 = cv2.imread(ui_img_loc_p,1)
    ui_img2 = cv2.imread(ui_img_loc_p,1)
    ui_img_grey = cv2.imread(ui_img_loc_p,0)                                #reading the UI Image, where the second parameter 0 stands for grey scale image reading.


    widget_sample_1 = input('Please provide the cropped image of widget 1 from UI form image:\n')
    widget_sample_1_p = convert_py_loc(widget_sample_1)
    widget_1_img_color = cv2.imread(widget_sample_1_p,1)    #reading the widget Image, where the second parameter 1 stands for colour image reading.
    widget_1_img = cv2.imread(widget_sample_1_p,0)          #reading the widget Image, where the second parameter 0 stands for grey scale image reading.


    widget_sample_2 = input('Please provide the cropped image of widget 2 from UI form image:\n')
    widget_sample_2_p = convert_py_loc(widget_sample_2)
    widget_2_img_color = cv2.imread(widget_sample_2_p,1)    #reading the widget Image, where the second parameter 1 stands for colour image reading.
    widget_2_img = cv2.imread(widget_sample_2_p,0)          #reading the widget Image, where the second parameter 0 stands for grey scale image reading.



    kmeans_ui_img, km_ui_img = kmeans_application(ui_img)


    best_clus_ui_img = best_cluster(kmeans_ui_img, km_ui_img)

    ui_img_cen = kmeans_ui_img[best_clus_ui_img].cluster_centers_   #taking the centers of the best number of clusters.
    ui_img_cen_rgb = ui_img_cen.astype(int)                         #list of RGB colours in palette.
  
    print('Colours in pallete of UI form image are (in RGB format[R,G,B]):\n')
    for j in ui_img_cen_rgb:
        print(j.tolist())


    pallete_ui_img = pallete(ui_img_cen_rgb)


    #Widget 1 part

    w, h = widget_1_img.shape[::-1]                                           #storing the width and height of the cropped widget 1. 
    match1 = cv2.matchTemplate(ui_img_grey, widget_1_img, cv2.TM_CCOEFF_NORMED)   #Matching the widget 1 with the main UI form image using Normalised Correlation Coefficient method.

    min_value1, max_value1, min_location1, max_location1 = cv2.minMaxLoc(match1)  #Storing the location of match     

    position1 = [max_location1[0],max_location1[1],max_location1[0]+w, max_location1[1]+h] #Storing the location in desired format [Top, Left, Bottom, Right]

    print('Position of the widget 1 with respect to UI form image taking top left corner coordinate as [0,0] that is [top, left], format [Top, Left, Bottom, Right]:\n',position1)

    top_left1 = max_location1
    bottom_right1 = (top_left1[0]+w, top_left1[1]+h)
    pos_img1 = cv2.rectangle(ui_img1,top_left1,bottom_right1,(127,255,0),10 )      #Marking the widget location in main UI Form, with green box.




    kmeans_widget1_img, km_widget1_img = kmeans_application(widget_1_img_color)


    best_clus_widget1_img = best_cluster(kmeans_widget1_img, km_widget1_img)

    widget1_img_cen = kmeans_widget1_img[best_clus_widget1_img].cluster_centers_   #taking the centers of the best number of clusters.
    widget1_img_cen_rgb = widget1_img_cen.astype(int)                              #list of RGB colours in pallete.
  
    print('Colours in pallete of Widget 1 image are (in RGB format[R,G,B]):\n')
    for k in widget1_img_cen_rgb:
        print(k.tolist())


    pallete_widget1_img = pallete(widget1_img_cen)

    #Widget 2 part

    wi, hi = widget_2_img.shape[::-1]                                           #storing the width and height of the cropped widget 1. 
    match2 = cv2.matchTemplate(ui_img_grey, widget_2_img, cv2.TM_CCOEFF_NORMED)   #Matching the widget 1 with the main UI form image using Normalised Correlation Coefficient method.

    min_value2, max_value2, min_location2, max_location2 = cv2.minMaxLoc(match2)  #Storing the location of match     

    position2 = [max_location2[0],max_location2[1],max_location2[0]+wi, max_location2[1]+hi] #Storing the location in desired format [Top, Left, Bottom, Right]

    print('Position of the widget 2 with respect to UI form image taking top left corner coordinate as [0,0] that is [top, left], format [Top, Left, Bottom, Right]:\n',position2)

    top_left2 = max_location2
    bottom_right2 = (top_left2[0]+wi, top_left2[1]+hi)
    pos_img2 = cv2.rectangle(ui_img2,top_left2,bottom_right2,(127,255,0),10 )      #Marking the widget location in the main UI Form, with a green box.




    kmeans_widget2_img, km_widget2_img = kmeans_application(widget_2_img_color)


    best_clus_widget2_img = best_cluster(kmeans_widget2_img, km_widget2_img)

    widget2_img_cen = kmeans_widget2_img[best_clus_widget2_img].cluster_centers_   #taking the centers of the best number of clusters.
    widget2_img_cen_rgb = widget2_img_cen.astype(int)                              #list of RGB colours in pallete.
  
    print('Colours in pallete of Widget 2 image are (in RGB format[R,G,B]):\n')
    for l in widget2_img_cen_rgb:
        print(l.tolist())


    pallete_widget2_img = pallete(widget2_img_cen)
    


    print('Main UI form image generated in window: Main_UI_Image \n')
    print(' ')
    print('Pallete of UI form image generated in window: Main_UI_Pallete \n')
    print(' ')
    print('Cropped imgage of widget 1 generated in window: Cropped_Widget1 \n')
    print(' ')
    print('Position of widget 1 with respect to UI form image generated in window: Pos_Widget1_UI \n')
    print(' ')
    print('Pallete of Widget 1 image generated in window: Widget1_Pallete \n')
    print(' ')
    print('Cropped imgage of widget 2 generated in window: Cropped_Widget2 \n')
    print(' ')
    print('Position of widget 2 with respect to UI form image generated in window: Pos_Widget2_UI \n')
    print(' ')
    print('Pallete of Widget 2 image generated in window: Widget2_Pallete \n')
    print(' ')

    cv2.imshow('Main_UI_Image',ui_img)
    cv2.imshow('Main_UI_Pallete',pallete_ui_img)
    cv2.imshow('Cropped_Widget1',widget_1_img_color)
    cv2.imshow('Pos_Widget1_UI',pos_img1)
    cv2.imshow('Widget1_Pallete',pallete_widget1_img)
    cv2.imshow('Cropped_Widget2',widget_2_img_color)
    cv2.imshow('Pos_Widget2_UI',pos_img2)
    cv2.imshow('Widget2_Pallete',pallete_widget2_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows                                   #Displays the pallete for unlimited time.


