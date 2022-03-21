import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import norm
from numpy import arctan, pi, signbit
# from scipy.optimize import fsolve
# from scipy.integrate import trapezoid
# from scipy.optimize import newton
# from scipy.spatial.distance import directed_hausdorff


def vec_angle(v1, v2):
    u1 = v1 / norm(v1)
    u2 = v2 / norm(v2)

    y = u1 - u2
    x = u1 + u2

    a0 = 2 * arctan(norm(y) / norm(x))

    if (not signbit(a0)) or signbit(pi - a0):
        return a0
    elif signbit(a0):
        return 0.0
    else:
        return pi

def printt(var_str, var=None):
  printt.counter += 1
  if var is None:
      print(f"#{printt.counter}: {var_str}")
  else:
      print(f"#{printt.counter}: {var_str} = {var}")

class Curve(ABC):
    def f(self, t):
        return self.x(t), self.y(t), self.z(t)
    
    @abstractmethod
    def x(self,t):
        pass
    @abstractmethod
    def y(self,t):
        pass
    @abstractmethod
    def z(self,t):
        pass
    
    def catenary(self,t,a=1/2):
        return t, \
            a * np.cosh(t/a), \
                0 * t
                    
    def knot_square(self,t):
        return 3*np.sin(t) + 2*np.sin(3*t), \
                np.cos(t) - 2 * np.cos(3 * t),\
                np.cos(5*t)
    
    def knot_granny(self,t):
        return 3*np.sin(t) + 2*np.sin(3*t), \
                np.cos(t) - 2 * np.cos(3 * t),\
                np.sin(10*t)
    
    def conical_helix(self,t,alpha=pi/6,beta=pi/4):
        k = np.sin(alpha) / np.tan(beta)
        return alpha * np.exp(k * t) * np.cos(t), \
            alpha * np.exp(k * t) * np.sin(t), \
                alpha * np.exp(k * t) / np.tan(alpha)
    
    def hippopede(self,t,R=10,a=5):
        ''' a must be less than R, R is a radius of a ball'''
        if a >= R:
            a = R / 2;
            
        return a + (R - a) * np.cos(t), \
            (R-a) * np.sin(t), \
                2 * np.sqrt(a * (R-a)) * np.sin(t / 2)        
    def toric_solenoid(self,t,R=10,r=2,n=1):
        if n < 1:
            n = 1
        if r > R:
            r = R
        return (R + r * np.cos(n*t)) * np.cos(t), \
            (R + r * np.cos(n*t)) * np.sin(t), \
                r * np.sin(n*t)
    def knot_trefoil(self,t,epsilon=1):
        '''usually epsilon is either 1 or -1'''
        return np.cos(t) + 2 * np.cos(2*t), \
            np.sin(t) - 2 * np.sin(2*t), \
                2 * epsilon * np.sin(3*t)
    
    def psudogeodedisc(self,t,a=0.5,theta=pi/3):
        ''' usually theta is between 0 and half of PI'''
        return a * np.cos(t), \
            a * np.sin(t), \
                a * np.tan(theta) * np.cosh(t / np.tan(theta))
    
    def pancake(self,t):
        return 8*np.cos(t)/np.sqrt(66+2*np.cos(4*t)), \
            8*np.sin(t)/np.sqrt(66+2*np.cos(4*t)), \
                2*np.cos(2*t)/np.sqrt(66+2*np.cos(4*t))
                
    def open_knot(self,t,a=3,b=4):
        return a * np.cos(1*t), \
            a * np.sin(1*t), \
                b * np.cos(2*t)
    
    def cubical_parabola(self,t,a=1):
        return a * t, \
            a * t**2, \
                a * t**3
    
    def conical_rose(self,t,n=1,a=1,b=1):
        return a * np.cos(n*t) * np.cos(t), \
            a * np.cos(n*t) * np.sin(t), \
                b * np.cos(n*t)
    
    def rhumb_line(self,t,R=10,k=1):
        return R * np.cos(t) / np.cosh(k*t), \
            R * np.sin(t) / np.cosh(k*t), \
                R * np.tanh(k*t)
    
    def hyperbolic_conical_spiral(self, t, alpha=1.5):
        return alpha * np.cos(t) / t, \
            alpha * np.sin(t) / t, \
                alpha / np.tan(alpha) / t
    
    def viviani(self,t,R=10):
        return R * (np.cos(t))**2, \
            R * np.cos(t) * np.sin(t), \
                R * np.sin(t)
    
    def sin_j(self,t):
        return t, \
            np.sin(t), \
                0 * t
                
class DiscreteCurve(Curve):
    def __init__(self):
        self.figure = plt.figure()
        # syntax for 3-D projection
        self.ax = plt.axes(projection='3d')
        self.start_pt = []
        self.end_pt = []
        self.points = []
        self.meshs = []

    def rotate_matrix(self, axis_vector, theta):
        '''this function just for 3-D situation.
          which means the axis_vector should be lik [x y z]' column vector type
          axis_vector should be type of a numpy.ndarray with shape of (3,) '''

        W = np.matrix([[0, -1 * axis_vector[2], axis_vector[1]],
                       [axis_vector[2], 0, -1 * axis_vector[0]],
                       [-1 * axis_vector[1], axis_vector[0], 0]])

        I = np.matrix(np.identity(3))

        R = I + np.sin(theta) * W + ((1 - np.cos(theta)) * W ** 2)

        return R

    def show(self,elev=60,azim=45,enable_discrete_line=True):
        # self.ax.set_title('3D curve plot')
        self.ax.set_xlim([-1,21])
        self.ax.set_ylim([-.5,.5])
        if enable_discrete_line:
            self.ax.plot3D(self.points[0], self.points[1], self.points[2], "green")
        
        self.ax.plot3D(self.meshs[0],self.meshs[1], self.meshs[2], "purple")
        self.ax.view_init(elev,azim)
        plt.show()

    def discrete_curve_demo(self, vertex_number=16):
        gamma = np.array([[0, 0, 0]])
        outplot = gamma

        len = 10  # each section has a fixed length
        Tangent = np.array([1, 0, 0])
        Binormal = np.array([0, 0, 1])

        gamma = gamma + 10 * Tangent

        outplot = np.concatenate((outplot, gamma))

        for i in range(1, vertex_number):
            kappa = i / vertex_number
            torsion = (vertex_number - i) / vertex_number  # if torsion is zero the curve will be stuck on a plane

            RT = self.rotate_matrix(Binormal, kappa)

            Tangent = np.ravel(np.matmul(RT, Tangent))

            RB = self.rotate_matrix(Tangent, torsion)
            Binormal = np.ravel(np.matmul(RB, Binormal))

            gamma = gamma + len * Tangent

            outplot = np.concatenate([outplot, gamma])

        return outplot
    
    def x(self,t):
        return 3*np.sin(t) + 2*np.sin(3*t)
    
    def y(self,t):
        return np.cos(t) - 2 * np.cos(3 * t)
    
    def z(self,t):
        return np.cos(5*t)
    
    def f(self,t):
        return self.catenary(t,100)

    def discrete(self, pts_x, pte_x, segs):
        '''discrete curve to segments
        pts: start point with form like [x, y]
        pte: end point with the same formation fo pts
        segs: number of segments to be cut
        return: a list of all points sorted by x coordinates'''

        t = np.linspace(pts_x, pte_x, segs, endpoint = True)

        X,Y,Z = self.f(t)

        self.points = np.vstack([X, Y, Z])
        self.start_pt = self.points[0:3,0]
        self.end_pt = self.points[0:3,-1]
        
        return self.points
    
    def translation_matrix(self, delta_pt):
        '''delta_pt should be [dx,dy,dz] like'''
        return np.array([[1,0,0,delta_pt[0]],
                        [0,1,0,delta_pt[1]],
                        [0,0,1,delta_pt[2]],
                        [0,0,0,1]])
    
    def dual_catenary(self, pts, pte, segs):
        '''anytime the starting point is the most left 
        shape of a normal catenary curve'''
        
        # start from the first half part of the interval between pts to pte
        t1 = np.linspace((pts[0]-pte[0])/4,  \
                        (pte[0]-pts[0])/4, \
                            int(segs/2), False)
        X1,Y1,Z1 = self.f(t1)
        b = np.ones(int(segs/2))
        self.points = np.vstack([X1, Y1, Z1, b])
        # T = self.translation_matrix(pts+(pte[0]-pts[0])/4)
        # self.points = np.dot(T,self.points)
        
        x,y,z = self.f((pte[0]-pts[0])/4)
        X2 = np.append(X1, x)
        Y2 = np.append(Y1, y)
        Z2 = np.append(Z1, z)
        b2 = np.append(b,1)
        points1 = np.vstack([X2,Y2,Z2,b2])
        T1 = self.translation_matrix((pte-pts)/2)
        
        self.points = np.append(self.points, np.dot(T1, points1), axis=1)
 
        T2 = self.translation_matrix(pts - self.points[0:3,0])
        self.points = np.dot(T2, self.points)
        
        self.start_pt = self.points[:,0]
        self.end_pt = self.points[:,-1]
        
        # printt("dual points:", self.points)
        return self.points
    
    def dist_p2l(self, pt, lpts, lpte):
        ''' give the distance between a point to a line by a pairs of points
        pt: a point beside the line
        lpts: a start point of the line
        lpte: end point of the line '''
        if (lpts==lpte).all():
            return norm(pt-lpts)
        else:
            return norm(np.cross(lpte-lpts, lpts-pt))/norm(lpte-lpts)

    def curvature(self, pts=None):
        ''' calculate the curvature alone each point in a poit set
        pts: [[x0,x1,x2,...]
            [y0,y1,y2,...]
            [z0,z1,z2,...]]
        output: curvature list in terms of each point (x, y, z)'''

        if pts is None:
            pts = self.points

        K = np.gradient(pts[1, :], pts[0, :], edge_order=2) / (1 + (np.gradient(pts[1, :], pts[0, :]))**2)**1.5

        return K
   
    def is_curvature_good(self, ps, idx_s, idx_e, theta_acc):

        if (idx_e - idx_s) < 2:
            return True
        ang = 0
        i = idx_s
        while i < (idx_e-2):
            v1 = ps[:,i+1] - ps[:,i]
            v2 = ps[:,i+2] - ps[:,i+1]
            ang += vec_angle(v1, v2)
            i += 1
        
        if ang > theta_acc:
            return False
        else:
            return True
              
    def is_dist_good(self, ps, idx_s, idx_e, delta_acc):
        ''' check points in ps from index_s to index_e meet the
        requirement by delta control'''
        de_t = norm(ps[:,idx_s] - ps[:,idx_e]) * delta_acc
        for i in range(idx_s, idx_e):
            de = dc.dist_p2l(ps[:,i], ps[:,idx_s], ps[:,idx_e])
            if de > de_t:
                return False
        return True

    def mesh(self, delta_acc, theta_acc):
        '''
            delta_acc = 2 / 100    # control the deflection
            theta_acc = 1 / 100    # control the angle
        '''
        # get the size of pts to catch the total number of points
        if len(self.points) == 0:
            return
        
        ps = np.delete(self.points,(3),axis=0)
        l = ps.shape[1]
        if l < 3:
            return
        
        start_idx = 0
        i = start_idx + 2
        prev_end_idx = 1
        g_flag = False
        self.meshs = np.vstack(ps[:,0])
        while i<l:
            ''' to check if the line segment is good in angle acc,
            when the answer is YES this point will be ignored! '''
            if self.is_dist_good(ps,start_idx,i,delta_acc) & \
                self.is_curvature_good(ps,start_idx,i,theta_acc):
                prev_end_idx = i
                i += 1
                g_flag = True
                continue
            else:
                g_flag = False
                self.meshs = np.append(self.meshs, np.vstack(ps[:,prev_end_idx]), axis=1)
                start_idx = prev_end_idx
                i = start_idx + 1

        if g_flag:
            g_flag = False
            self.meshs = np.append(self.meshs, np.vstack(ps[:,prev_end_idx]), axis=1)
        
        # printt(meshs)
        printt(f"{self.meshs.shape[1]} points totally")
             

if __name__ == "__main__":
    printt.counter = 0

    dc = DiscreteCurve()
    # generate discrete points set alone the curve
    dc.dual_catenary(pts=np.array([0,0,0]), \
                     pte=np.array([20,0,0]), \
                         segs=101)
  
    dc.mesh(delta_acc=0.5/100, theta_acc=0.1)
    
    # dc.shift2([1,0,0])
    
    dc.show(90,-90)

