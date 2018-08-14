import sklearn as sk
import numpy as np
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt



class KFStruct():
    def __init__(self, x_p, phi,R ,Q, H):
        """
        :param x_p:
            prior estimate variable x^, shape: 1 x m
        :param phi:
            state transition matrix, shape: m x m
        :param R:
            noise mode, R = E[vk*vk^T]
        :param Q:
            noise mode, Q = E[wk*wk^T]
        :param H:
            connection matrix between x and z, shape: m x m
        X_p: prior estimate vectors
        phi: state transition matrix
        R: noise model
        Q: noise model
        H: connection matrix H between observation z and x_p
        X_p: prior estimate xk'
        Xk: estimate xk
        zk: observation state vector from senors
        K : m x m kalman gain matrix
        h: non-linear connection function
        f: non-linear transition function
        """
        self.X_p = x_p
        self.phi = phi
        self.H = H
        self.R = R
        self.Q = Q
        self.Xk = np.mat(np.zeros([1, np.shape(x_p)[1]]))
        self.Pk = np.mat(np.identity(np.shape(x_p)[1]))
        self.Pk_p = np.mat(np.identity(np.shape(x_p)[1]))
        self.zk = np.mat(np.zeros([1, np.shape(x_p)[1]]))
        self.K = self.cal_Kk()
        pass


    def cal_Kk(self):
        """
        calculate the Kalman gain:
            Kk = Pk' *H^T*(H*Pk' *H^T +R)^-1
        :return:
        """
        # measurement prediction covariance Sk
        sk = np.mat(self.H* self.Pk_p * np.transpose(self.H)) + self.R
        self.K = self.Pk_p * np.transpose(self.H) * np.linalg.inv(sk)
        return self.K

    def get_ObsData(self, data):
        """
        obtain observation data zk from sensor
        :return:
        """
        self.zk = data
        return self.zk


    def update_Xk(self):
        """
        update estimate:
        xk = xk' + Kk*(zk - (H* xk^T)^T)
        where:
            xk: 1 x m matrix
            zk: 1 x m matrix
            Kk: Kalman gain, scalar value
            H: m x m connection matrix
        :return:
        """
        # print("H",np.shape(self.H))
        # print("X_p:", np.shape(self.X_p))
        # linear Kalman filter
        ik = (self.zk.T - self.H * np.transpose(self.X_p))
        self.Xk = self.X_p + (self.K * ik).T
        return self.Xk

    def update_cov(self):
        """
        update covariance matrix Pk
        Pk = (I - K*H)* Pk'
        where:
            I: mxm identity matrix
            H: mxm connection matrix
            Pk' : mxm prior estimate covariance matrix
        :return:
        """
        idMat = np.mat(np.identity(np.shape(self.X_p)[1]))
        self.Pk = (idMat - np.multiply(self.K,self.H)) * self.Pk_p
        return self.Pk

    def update(self):
        # # update estimate
        self.update_Xk()
        # print("x%d:"%(t), data)
        # # update covariance
        self.update_cov()

    def get_PrjPrediction(self):
        """
        Project the xk to x_k+1 and update covariance matrix P_k+1'
        :return:
        """
        # project state from k to k+1
        self.X_p = (self.phi * np.transpose(self.X_p)).T
        # update Pk_p to state at K+1
        self.Pk_p = self.phi *self.Pk * np.transpose(self.phi) + self.Q
        return self.X_p

    def KFilter(self, data, expected_s,t=None):
        """
        Kalman filter operations
        t: time where k=t
        data: state vector xk
        :return:
            estimated state vector
        """

        # # obtain observation data
        self.get_ObsData(data)
        self.X_p = expected_s

        # update Kalman gain
        self.cal_Kk()

        # update step
        self.update()
        # print("cov Mat: ",covMat)
        # # project to new state and return filtered data
        prediction =self.get_PrjPrediction()
        return prediction


def loadData(model = "linear", f= None, df= None):
    """
    x : time array
    y : matrix of states
    state vector: [x1, x1', x2,x2']
    :return:
    """
    features_num =3
    dt = 0.5
    t = np.arange(1, 50.0,dt,dtype= float)
    y = np.mat(np.zeros([len(t), features_num]))
    H = np.mat(np.identity(features_num))
    # linear input examples
    if model.lower() == 'linear':
        a0= 1.2
        y[:,0] = np.mat( a0 *t**2/2).T
        y[:,1] = np.mat(a0 * t).T
        y[:,2] =np.mat(np.ones([1,len(t)]) * a0).T

        phi =np.mat([[1,dt,0],
                     [0,1,0],
                     [0,1,0]])
    elif model.lower() == 'accmove':
        a0 = 1.2
        k0 = 1.1
        # movement model
        # formulas:
        #  a(t) = k0^t *a0
        #  y = s(t) = 1/2 *a(t)*t^2
        # position
        y[:,0] =((a0 * np.power(np.ones([1,len(t)])*k0,t) * t**2)/2).T
        # velocity
        y[:,1] = (np.power(np.ones([1,len(t)])*k0,t)* a0 * t).T
        # acceleration
        y[:,2] = (np.power(np.ones([1,len(t)])*k0,t)* a0).T
        phi = np.mat([[1, dt, dt ** 2 * k0],
                      [0, 1, k0 * dt],
                      [0, 0, k0]])
        # generate the corresponding state transition matrix
        # and connection matrix
    elif model.lower() == 'accosc':
        # acceleration-oscillating model
        a0 = 2.0*np.array(np.sin(t))
        # print("a0:",a0)
        # movement model
        # formulas:
        #  a(t) = k0^t *a0
        #  y = s(t) = 1/2 *a(t)*t^2
        # position
        y[:, 0] = ((a0 * t**2) / 2).reshape([1,98]).T
        # velocity
        y[:, 1] = (a0 * t).reshape([1,98]).T
        # acceleration
        y[:, 2] = a0.reshape([1,98]).T
        phi = np.mat([[1, dt, dt ** 2 ],
                      [0, 1, dt ],
                      [0, 0, 1]])
        pass
    else:
        # non-linear
        features_num = 2
        y = np.mat(np.zeros([len(t), features_num]))
        H = np.mat(np.identity(features_num))
        # initialize velocity v=1.0. position =0 with non-linear function
        y[0,:] = f(np.mat([0,0.4]))
        for i in range(1, len(t)-1):
            # input matrix
            y[i,:] = f(y[i-1,:])
            pass
        #initial phi
        phi = df(y[0,:])
        print("Phi: ",phi)
    return t, y, H, phi


def generate_Noise(shape=4, model="Gauss", sigma =1.0):
    R = np.mat(np.identity(shape))
    Q = np.mat(np.identity(shape))
    if model.lower() == "gauss":
        for i in range(shape):
            R[i,i] = np.random.normal(0,sigma,1)

        rand_num = np.random.rand(1)
        for i in range(shape):
            Q[i, i] = np.random.normal(0, sigma, 1)
            pass

    elif model.lower() == "linear":
        R = np.mat(np.identity(shape) * sigma)
        Q = np.mat(np.identity(shape) * sigma)
    else:
        print("Don't recognize model:%s"%(model))
    return R, Q



def plot_data(x,y, estmated_y,actual_obv):
    s = { 0:"Position",1:"Velocity",2:"acceleration"}
    for i in range(y.shape[1]):
        plot = plt.figure()
        ax = plot.add_subplot(111)
        ax.scatter(x[:],actual_obv[:,i].flatten()[0],s=10,c='g',label = 'Observation')
        ax.scatter(x[:],y[:,i].flatten()[0],s=10,c='r',label = 'Expectation')
        ax.scatter(x[:],estmated_y[:,i].flatten()[0],s=10,c='b',label = 'Estimate')
        ax.set_ylabel(s[i])
        ax.set_xlabel("time")
        ax.legend(loc=1)
        plt.show()
    pass



def info_decorator(text):
    def wrapper(func):
        def wrapped_func():
            print(text)
            func()
            return
        return wrapped_func
    return wrapper

@info_decorator("Kalman filter test")
def KFTest():
    # load test data
    x, y, H, phi = loadData(model="accmove")

    # add noise term to expectation output to obtain actual observation
    actual_obv = np.mat(np.zeros([y.shape[0], y.shape[1]]))
    for i in range(y.shape[1]):
        actual_obv[:, i] = y[:, i] + np.mat(np.random.normal(0, 100, y.shape[0])).T
    print("x:", x.shape)
    print("y:", y.shape)
    print("actual:", actual_obv.shape)

    #generate noise model
    R,Q = generate_Noise(3, model="gauss", sigma= 0.01)
    print("R:",R.shape)
    print("Q:",Q.shape)

    # initialize Kalman filter params
    kf_obj = KFStruct(y[0,:],phi,R,Q,H)


    # predict data
    estimated_v = []
    for k in range(y.shape[0]):
        # input the control state vector
        if k+1 >= y.shape[0]:
            i=k
        else:
            i = k+1
        t = x[i]
        estimated_v.append(kf_obj.KFilter( actual_obv[i,:], y[i,:],t=t))
        # print("estimate:",estimated_v[-1])
        pass

    # plot data
    # estimated_v : filtered prediction output
    # y: expected output
    # actual_obv: actual observation data
    plot_data(x,y, np.mat(np.array(estimated_v)), actual_obv)
    pass




if __name__ == "__main__":
    KFTest()
    pass
