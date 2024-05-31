import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as hcluster
import math
import time
start_time=time.time()
file_path='C:/Users/15418/Downloads/filtered_apples.npy'
cloud=np.load(file_path)
#print(cloud)
def three_D_plot(cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x=cloud[:,0]
    y=cloud[:,1]
    z=cloud[:,2]
    ax.scatter(x, y, -z, zdir='z', c= 'red')
    #plt.savefig("demo.png")
    plt.show()

def testplot(cloud):
    x=cloud[:,0]
    y=cloud[:,1]
    plt.scatter(x,y)
    plt.show()
def xy_clustering(cloud, graph=False):
    x=cloud[:,0][0::100]
    y=cloud[:,1][0::100]
    data=np.dstack((x,y))[0]
    thresh = .02
    clusters = hcluster.fclusterdata(data, thresh, criterion="distance")
    if graph:
        plt.scatter(*np.transpose(data), c=clusters)
        plt.axis("equal")
        plt.show()
    return data, clusters

def get_cluster_center(data, clusters):
    stacked_array=[]
    for i in range(len(data)):
        stacked_array.append([data[i][0],data[i][1], clusters[i]])
    cluster_numbers=np.unique(clusters)
    cluster_centers=[]
    stacked_array=np.array(stacked_array)
    for number in cluster_numbers:
        mask=(stacked_array[:,2]==number)
        x=stacked_array[mask,0]
        y = stacked_array[mask, 1]
        x_average=np.mean(x)
        y_average=np.mean(y)
        #range to use for the rest of the pixels (not sampled originally
        y_min=np.min(y)
        y_max=np.max(y)
        y_range=((y_max-y_min)*1.5)/2
        x_min=np.min(x)
        x_max=np.max(x)
        x_range = ((x_max - x_min) * 1.5) / 2
        cluster_centers.append([x_average,y_average, x_range, y_range])
    return cluster_centers

def upsample(cloud,cluster_centers, graph=False):
    cloud_list=[[],[],[],[],[]]
    x=cloud[:,0]
    y=cloud[:,1]
    z = cloud[:, 2]
    data = np.dstack((x, y,z))[0]
    #empty array to put designated cluster in
    clusters=[]
    for point in data:
        for i in range(len(cluster_centers)):
            cluster=cluster_centers[i]
            if point[0]>cluster[0]-cluster[2] and point[0]<cluster[0]+cluster[2] and point[1]>cluster[1]-cluster[3] and point[1]<cluster[1]+cluster[3]:
                clusters.append(i)
                cloud_list[i].append(point)
                break
    #print(cloud_list)
    if graph:
        plt.scatter(*np.transpose(data), c=clusters)
        plt.axis("equal")
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #cloud=np.array(cloud_list[1])
        #x = cloud[:, 0]
        #y = cloud[:, 1]
        #z = cloud[:, 2]
        ax.scatter(x, y, z, zdir='z', c=clusters)
        plt.show()
    return cloud_list

#from https://jekel.me/2015/Least-Squares-Sphere-Fit/
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]

def get_spheres(cloud_list):
    centers=[]
    radii=[]
    for cloud in cloud_list:
        cloud=np.array(cloud)
        x_vals=cloud[:,0]
        y_vals=cloud[:,1]
        z_vals=cloud[:,2]
        sphere=sphereFit(x_vals,y_vals,z_vals)
        centers.append([sphere[1][0], sphere[2][0], sphere[3][0]])
        radius=sphere[0]
        radii.append(radius)
        sphere_x_max=sphere[1][0]+radius
        sphere_x_min=sphere[1][0]-radius
        xs=[sphere_x_max,sphere_x_min]
        sphere_y_max=sphere[2][0]+radius
        sphere_y_min=sphere[2][0]-radius
        ys=[sphere_y_max,sphere_y_min]
        sphere_z_max=sphere[3][0]+radius
        sphere_z_min=sphere[3][0]-radius
        zs=[sphere_z_max, sphere_z_min]
        quadrants=[]
        for x in xs:
            for y in ys:
                for z in zs:
                    quadrants.append([x,y,z])
        #downsample cloud
        cloud=cloud[:, :][0::25]
        #collect the point totals for each cloud
        point_totals=[]
        for quadrant in quadrants:
            num_points=0
            x_quad, y_quad, z_quad = quadrant
            if x_quad>sphere[1][0] and y_quad>sphere[2][0] and z_quad>sphere[3][0]:
                for point in cloud:
                    if point[0]>sphere[1][0] and point[1]>sphere[2][0] and point[2]>sphere[3][0]:
                        num_points+=1
                point_totals.append(num_points)
            elif x_quad<sphere[1][0] and y_quad>sphere[2][0] and z_quad>sphere[3][0]:
                for point in cloud:
                    if point[0]<sphere[1][0] and point[1]>sphere[2][0] and point[2]>sphere[3][0]:
                        num_points+=1
                point_totals.append(num_points)
            elif x_quad<sphere[1][0] and y_quad<sphere[2][0] and z_quad>sphere[3][0]:
                for point in cloud:
                    if point[0]<sphere[1][0] and point[1]<sphere[2][0] and point[2]>sphere[3][0]:
                        num_points+=1
                point_totals.append(num_points)
            elif x_quad>sphere[1][0] and y_quad<sphere[2][0] and z_quad>sphere[3][0]:
                for point in cloud:
                    if point[0]>sphere[1][0] and point[1]<sphere[2][0] and point[2]>sphere[3][0]:
                        num_points+=1
                point_totals.append(num_points)
            elif x_quad<sphere[1][0] and y_quad<sphere[2][0] and z_quad<sphere[3][0]:
                for point in cloud:
                    if point[0]<sphere[1][0] and point[1]<sphere[2][0] and point[2]<sphere[3][0]:
                        num_points+=1
                point_totals.append(num_points)
            elif x_quad>sphere[1][0] and y_quad<sphere[2][0] and z_quad<sphere[3][0]:
                for point in cloud:
                    if point[0]>sphere[1][0] and point[1]<sphere[2][0] and point[2]<sphere[3][0]:
                        num_points+=1
                point_totals.append(num_points)
            elif x_quad>sphere[1][0] and y_quad>sphere[2][0] and z_quad<sphere[3][0]:
                for point in cloud:
                    if point[0]>sphere[1][0] and point[1]>sphere[2][0] and point[2]<sphere[3][0]:
                        num_points+=1
                point_totals.append(num_points)
            elif x_quad<sphere[1][0] and y_quad>sphere[2][0] and z_quad<sphere[3][0]:
                for point in cloud:
                    if point[0]<sphere[1][0] and point[1]>sphere[2][0] and point[2]<sphere[3][0]:
                        num_points+=1
                point_totals.append(num_points)
    return centers, radii, point_totals

def get_spheres_polar(cloud_list,degree_step):
    centers=[]
    radii=[]
    all_spheres_point_totals=[]
    #running sum of all points to check that they all get assigned
    rs=0
    degrees = range(-180, 180, degree_step)
    quadrants = []
    #what are the quadrants?? same poitn could be described with two different angles


    #print(quadrants)
    for cloud in cloud_list:
        cloud=np.array(cloud)
        x_vals=cloud[:,0]
        y_vals=cloud[:,1]
        z_vals=cloud[:,2]
        sphere=sphereFit(x_vals,y_vals,z_vals)
        centers.append([sphere[1][0], sphere[2][0], sphere[3][0]])
        radius=sphere[0]
        radii.append(radius)
        #convert to polar
        polar_points=[]
        for point in cloud:
            #get x,y,z relative to cloud center
            x,y,z=point-[sphere[1][0], sphere[2][0], sphere[3][0]]
            #find the two angles
            t1=np.arctan2(y,x)*180/np.pi
            t2=np.arctan2(z,x)*180/np.pi
            r=np.sqrt(x**2+y**2+z**2)
            polar_points.append([t1,t2,r])
        #apple radius is usually .048ish
        #degree range: -180 to 180
        radius_max=radius*1.1
        radius_min=radius*.9


    #print(rs)
    return centers, radii, all_spheres_point_totals
#from https://stackoverflow.com/questions/64656951/plotting-spheres-of-radius-r
def plt_sphere(list_center, list_radius):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors=['r','g','b','y','m']
    i=0
    for c, r in zip(list_center, list_radius):
        # draw sphere
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_surface(x + c[0], y + c[1], z + c[2], color=colors[i],
                        alpha=0.5 * np.random.random() + 0.5)
        i+=1
    fig.show()
data, clusters=xy_clustering(cloud, graph=False)
cluster_centers=get_cluster_center(data,clusters)
cloud_list=upsample(cloud,cluster_centers, graph=False)
spheres=get_spheres(cloud_list)
#plt_sphere(spheres[0], spheres[1])
end_time=time.time()
print(end_time-start_time)
#plt.show()


