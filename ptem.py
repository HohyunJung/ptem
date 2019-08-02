
from __future__ import division
from numpy.linalg import inv, norm
from scipy.optimize import minimize
from copy import deepcopy
import numpy as np
from arspy.ars import adaptive_rejection_sampling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from scipy import special
import warnings
import time 

########## PTEM: A Popularity-based Topical Expertise Model
########## "PTEM: A Popularity-based Topical Expertise Model for Community Question Answering"
########## Jung, H., Lee, J., Lee, N., and Kim, S.
# @Author: Hohyun, Jung (hhjung@kaist.ac.kr)

class PTEM():
    def __init__(self, data, theta0=[-4.0,0.1,0.0,1.0], K=5, po_type=1, po_scale=1000, descriptions=True, T_obs=1425772800.0, time_unit='day', min_ans=20, max_df=0.25, min_df=0.01,
                 G=30, G0=10, alpha=None, gamma=0.01, gibbs_init=None, ofBdry=200, ars_a=-40, ars_b=40, opt_method='SLSQP'):
        self.ofBdry = ofBdry              
        self.po_type = po_type
        self.po_scale = po_scale
        self.max_df = max_df
        self.min_df = min_df
        self.gibbs_init = gibbs_init      
        self.time_unit = time_unit        # 'second', 'minute', 'hour', 'day' and any numbers
        self.td = self.td_()
        self.K = K                        # Number of topics
        self.G = G                        
        self.G0 = G0                      
        self.min_ans = min_ans           
        self.data = data                  # data format {userId:[{Id,Score,CreationDate,Accepted,numAcceptSofar,Topic},...]} 
        self.theta0 = np.asarray(theta0)
        self.opt_method = opt_method      
        self.T_obs = T_obs / self.td
        self.ars_a, self.ars_b, self.ars_domain = ars_a, ars_b, (float("-inf"), float("inf"))  
        self.users = self.users_()        
        self.n_pa = len(self.theta0)
        self.A = self.A_()              
        self.vo = self.vo_()            
        self.y_ua_m, self.y_ua_p = self.y_ua()    
        self.po = self.po_()            
        self.at = self.at_()           
        self.A_ukg = {user: [[None]*(self.G+1) for k in range(1,self.K+1)] for user in self.users}    
        self.W, self.words, self.startIdx, self.TF_uaw = self.words_()  
        self.alpha = self.alpha_(alpha)
        self.gamma = np.zeros(self.W) + gamma
        self.checkValidity = self.checkValidity_()
        self.z_uag = {user: np.zeros((self.A[user],self.G+1), dtype=int) for user in self.users}
        self.x_ukg = {user: np.zeros((self.K,self.G+1), dtype=float) for user in self.users}   
        self.N_ukg = {user: np.zeros((self.K, self.G+1), dtype=int) for user in self.users}  
        self.N_kwg = np.zeros((self.K,self.W,self.G+1), dtype=int)                           
        self.theta = np.asarray(theta0)
        self.thetaCov = None
        self.theta_list = list()    
        self.ofCount = 0
        if descriptions == True :
            print('%d users, %d answers, %d vocabs.' %(len(self.users), np.sum(list(self.A.values())), self.W))
    
    def alpha_(self, alpha):
        if alpha == None:
            return np.zeros(self.K) + 50.0 / self.K
        else:
            return np.zeros(self.K) + alpha
    
    def words_(self):
        A_contents = list()
        startIdx = dict()                                                          
        idx = 0
        for u in self.users:
            startIdx[u] = idx
            for a in range(self.A[u]):
                ans = self.data[u][a]
                A_contents.append(ans['Body'])
                idx += 1    
        my_additional_stop_words = ['does','doesn','did','don','given','used','didn','just','use','said','hasn','haven','having','like','likely','need','make','probably','wiki',
                                   'wikipedia','know','pmwiki','able','want','www','http','com','th','st','rd']
        stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
        tf_vectorizer_A = CountVectorizer(analyzer = 'word', token_pattern='[a-zA-Z][a-zA-Z]+', max_df=self.max_df, min_df=self.min_df, stop_words=stop_words)
        TF_A = tf_vectorizer_A.fit_transform(A_contents)
        W = TF_A.shape[1]
        words = tf_vectorizer_A.get_feature_names()
        TF_uaw = {user: deepcopy(TF_A[startIdx[user]:startIdx[user]+self.A[user]]) for user in self.users}    
        return W, words, startIdx, TF_uaw
            
    def A_uk_(self):
        A_uk = {user: [list() for x in range(self.K)] for user in self.users}
        for u in self.users:
            for a in range(self.A[u]):
                ans = self.data[u][a]
                topic = ans['Topic']
                A_uk[u][topic-1].append(a)
        return A_uk
            
    def checkValidity_(self):
        for u in self.users:
            timeList = self.at[u]
            assert(all(timeList[i] <= timeList[i+1] for i in range(len(timeList)-1)))

    def td_(self):
        if self.time_unit == 'second': td = 1
        elif self.time_unit == 'minute': td = 60
        elif self.time_unit == 'hour': td = 60 * 60
        elif self.time_unit == 'day': td = 60 * 60 * 24
        else:
            td = self.time_unit
        return td
    
    def vo_(self):
        vo = {user: list() for user in self.users}
        for u in self.users:
            for ans in self.data[u]:
                v_ua = ans['Score']
                if v_ua < 0:
                    warnings.warn("There are some negative votes. They are set 0.")
                    v_ua = 0
                vo[u].append(v_ua)
        return vo    
    
    def y_ua(self):
        y_ua_m = {user: list() for user in self.users}     
        y_ua_p = {user: list() for user in self.users}  
        for u in self.users:
            for a in range(self.A[u]+1):
                if a == self.A[u]:
                    ans_prev = self.data[u][a-1]
                    t_a = self.T_obs
                    num_acc_a = ans_prev['numAcceptSofar'] + int(ans_prev['Accepted'])
                    total_a = 1.5 * num_acc_a
                else:
                    ans = self.data[u][a]
                    t_a = ans['CreationDate'] / self.td
                    num_acc_a = ans['numAcceptSofar'] 
                    total_a = 1.5 * num_acc_a
                for a2 in range(a):
                    ans2 = self.data[u][a2]
                    t_a2 = ans2['CreationDate'] / self.td
                    v_a2 = max(ans2['Score'], 0)
                    total_a += v_a2 * (t_a - t_a2) / (self.T_obs - t_a2)
                y_ua_m[u].append(total_a)
                if a == self.A[u]:
                    y_ua_p[u].append(total_a)
                else:
                    y_ua_p[u].append(total_a + 1.5 * int(ans['Accepted']))
        return y_ua_m, y_ua_p
    
    def po_(self):
        po = {user: list() for user in self.users}
        for u in self.users:
            Area = 0
            t_a_next = self.T_obs
            for a in reversed(range(self.A[u])):
                ans = self.data[u][a]
                t_a = ans['CreationDate'] / self.td
                Area += 0.5 * (t_a_next - t_a) * (self.y_ua_p[u][a] + self.y_ua_m[u][a+1])
                t_a_next = deepcopy(t_a)
                y_ua = Area / (self.T_obs - t_a)
                po[u].append(y_ua / self.po_scale)
            po[u].reverse()
        return po
    
    def at_(self):
        at = {user: list() for user in self.users}
        for u in self.users:
            for ans in self.data[u]:
                at[u].append(ans['CreationDate'] / self.td)
        return at         
    
    def A_(self):
        return {user: len(self.data[user]) for user in self.users}

    def users_(self):
        return [user for user in self.data if len(self.data[user]) >= self.min_ans and user != None]

    def mf(self, ftn):                  
        def f(x):
            return -ftn(x)
        return f

    def x_pri_logdist(self, x):
        return -0.5 * x**2 

    def x_pos_logdist(self, x, theta, u, k, g):
        return self.loglik_ukg(x, theta, u, k, g) + self.x_pri_logdist(x)
    
    def loglik_ua(self, x, theta, u, a):            
        theta = np.asarray(theta)
        beta = theta[0:3]
        xi = theta[3]
        xx = np.array([1, x, self.po[u][a]])
        vv = beta.dot(xx)
        mu = (self.T_obs - self.at[u][a]) * np.exp(vv)
        vote = self.vo[u][a]
        if vv < self.ofBdry :  
            value = special.loggamma(xi + vote) - special.loggamma(xi) + xi * np.log(xi) + vote * vv - (xi + vote) * np.log(xi + mu)
        else :
            self.ofCount += 1                 
            value = special.loggamma(xi + vote) - special.loggamma(xi) + xi * np.log(xi) + vote * vv - (xi + vote) * (np.log(self.T_obs - self.at[u][a]) + vv)
        return value
    
    def loglik_ukg(self, x, theta, u, k, g):
        total = 0
        for a in self.A_ukg[u][k-1][g]:
            total += self.loglik_ua(x, theta, u, a)
        return total

    def loglik_ug(self, x, theta, u, g):                  
        total = 0
        for k in range(1, self.K+1):
            total += self.loglik_ukg(x, theta, u, k, g)
        return total

    def loglik_der_ua(self, x, theta, u, a):                
        theta = np.asarray(theta)
        beta = theta[0:3]
        xi = theta[3]
        xx = np.array([1, x, self.po[u][a]])
        vv = beta.dot(xx)
        mu = (self.T_obs - self.at[u][a]) * np.exp(vv)
        vote = self.vo[u][a]
        if vv < self.ofBdry :   
            der_beta = (-(xi + vote) / (xi + mu) * mu + vote) * xx
            der_xi = special.polygamma(0, xi + vote) - special.polygamma(0, xi) - np.log(xi + mu) - (xi + vote) / (xi + mu) + np.log(xi) + 1
            der_value = np.hstack((der_beta, der_xi))
        else :
            der_beta = (-xi) * xx
            der_xi = special.polygamma(0, xi + vote) - special.polygamma(0, xi) - (np.log(self.T_obs - self.at[u][a]) + vv) + np.log(xi) + 1
            der_value = np.hstack((der_beta, der_xi))
        return der_value

    def loglik_der_ukg(self, x, theta, u, k, g):
        total = 0
        for a in self.A_ukg[u][k-1][g]:
            total += self.loglik_der_ua(x, theta, u, a)
        return total

    def loglik_hess_ua(self, x, theta, u, a):  
        theta = np.asarray(theta)
        beta = theta[0:3]
        xi = theta[3]
        xx = np.array([1, x, self.po[u][a]])
        vv = beta.dot(xx)
        mu = (self.T_obs - self.at[u][a]) * np.exp(vv)
        vote = self.vo[u][a]
        if vv < self.ofBdry :  
            hess_beta = -(xi * (xi + vote) * mu / (xi + mu)**2) * np.outer(xx, xx)
            hess_xi = special.polygamma(1, xi + vote) - special.polygamma(1, xi) - 1 / (xi + mu) + 1 / xi - (mu - vote) / (xi + mu)**2
            hess_cross = ((vote - mu) * mu / (xi + mu)**2) * xx
            A = np.vstack((hess_beta, hess_cross))
            B = np.hstack((hess_cross, hess_xi))
            hess_value = np.vstack((A.transpose(),B))
        else :                 
            hess_beta = 0 * np.outer(xx, xx)
            hess_xi = 0
            hess_cross = -1 * xx
            A = np.vstack((hess_beta, hess_cross))
            B = np.hstack((hess_cross, hess_xi))
            hess_value = np.vstack((A.transpose(),B))
        return hess_value

    def loglik_hess_ukg(self, x, theta, u, k, g):
        total = 0
        for a in self.A_ukg[u][k-1][g]:
            total += self.loglik_hess_ua(x, theta, u, a)
        return total
    
    def Q_uk(self, theta_prev, u, k):                        
        def ftn(theta):
            value = 0
            for g in range(self.G0+1,self.G+1):
                value += self.loglik_ukg(self.x_ukg[u][k-1,g], theta, u, k, g) / (self.G - self.G0)
            return value
        return ftn

    def Q_der_uk(self, theta_prev, u, k):                        
        def ftn(theta):
            value = 0
            for g in range(self.G0+1,self.G+1):
                value += self.loglik_der_ukg(self.x_ukg[u][k-1,g], theta, u, k, g) / (self.G - self.G0)
            return value
        return ftn

    def Q_hess_uk(self, theta_prev, u, k):                    
        def ftn(theta):
            value = 0
            for g in range(self.G0+1,self.G+1):
                value += self.loglik_hess_ukg(self.x_ukg[u][k-1,g], theta, u, k, g) / (self.G - self.G0)
            return value
        return ftn

    def Q_u(self, theta_prev, u):                        
        def ftn(theta):
            value = 0
            for k in range(1, self.K+1):
                value += self.Q_uk(theta_prev, u, k)(theta)
            return value
        return ftn

    def Q_der_u(self, theta_prev, u):                        
        def ftn(theta):
            value = 0
            for k in range(1, self.K+1):
                value += self.Q_der_uk(theta_prev, u, k)(theta)
            return value
        return ftn

    def Q_hess_u(self, theta_prev, u):                       
        def ftn(theta):
            value = 0
            for k in range(1, self.K+1):
                value += self.Q_hess_uk(theta_prev, u, k)(theta)
            return value
        return ftn

    def Q_(self, theta_prev):                                
        def ftn(theta):
            value = 0
            for u in self.users:
                value += self.Q_u(theta_prev, u)(theta)
            return value
        return ftn

    def Q_der(self, theta_prev):                            
        def ftn(theta):
            value = 0
            for u in self.users:
                value += self.Q_der_u(theta_prev, u)(theta)
            return value
        return ftn

    def Q_hess(self, theta_prev):                               
        def ftn(theta):
            value = 0
            for u in self.users:
                value += self.Q_hess_u(theta_prev, u)(theta)
            return value
        return ftn
    
    def ars_(self, theta, u, k, g):
        num = 0
        a_add, b_add = 0, 0
        while num < 20:
            try:
                sample = adaptive_rejection_sampling(logpdf=lambda x: self.x_pos_logdist(x=x, theta=theta, u=u, k=k, g=g), 
                                              a=self.ars_a+a_add, b=self.ars_b+b_add, domain=self.ars_domain, n_samples=1)
                break
            except:
                a_add = self.ars_a * (np.random.random()-0.5)
                b_add = self.ars_b * (np.random.random()-0.5)
                num += 1
                print("(Assertion or Value)Error has occured by non-log-concativity.")
        return sample[0]

    def Estep_(self, theta_prev, it):           
        self.init(it)                            
        self.gibbsSampling(theta_prev)       
        return self.Q_(theta_prev)
    
    def init(self, it):
        if self.gibbs_init == None:
            if it == 1:                        
                for u in self.users:
                    for k in range(1,self.K+1):
                        self.A_ukg[u][k-1][0] = list()
                    for a in range(self.A[u]):
                        k_rand = np.random.choice(list(range(1,self.K+1)))
                        self.z_uag[u][a,0] = deepcopy(k_rand)
                        self.A_ukg[u][k_rand-1][0].append(a)
                        self.N_kwg[k_rand-1,:,0] += self.TF_uaw[u][a,:]
                    for k in range(1,self.K+1):
                        self.x_ukg[u][k-1,0] = 0
                        self.N_ukg[u][k-1,0] = [self.z_uag[u][a,0] for a in range(self.A[u])].count(k)
            else:                              
                for u in self.users:
                    self.z_uag[u][:,0] = self.z_uag[u][:,self.G]
                    self.x_ukg[u][:,0] = self.x_ukg[u][:,self.G]
                    for k in range(1,self.K+1):
                        self.A_ukg[u][k-1][0] = deepcopy(self.A_ukg[u][k-1][self.G])
                        self.N_ukg[u][k-1,0] = self.N_ukg[u][k-1,self.G]
                self.N_kwg[:,:,0] = self.N_kwg[:,:,self.G]  
        else:
            if it == 1:                        
                z_uag_init = self.gibbs_init[0]
                x_ukg_init = self.gibbs_init[1]
                A_ukg_init = self.gibbs_init[2]
                N_ukg_init = self.gibbs_init[3]
                N_kwg_init = self.gibbs_init[4]
                for u in self.users:
                    self.z_uag[u][:,0] = z_uag_init[u][:,-1]
                    self.x_ukg[u][:,0] = x_ukg_init[u][:,-1]
                    for k in range(1,self.K+1):
                        self.A_ukg[u][k-1][0] = deepcopy(A_ukg_init[u][k-1][-1])
                        self.N_ukg[u][k-1,0] = N_ukg_init[u][k-1,-1]
                self.N_kwg[:,:,0] = N_kwg_init[:,:,-1]              
            else:                             
                for u in self.users:
                    self.z_uag[u][:,0] = self.z_uag[u][:,self.G]
                    self.x_ukg[u][:,0] = self.x_ukg[u][:,self.G]
                    for k in range(1,self.K+1):
                        self.A_ukg[u][k-1][0] = deepcopy(self.A_ukg[u][k-1][self.G])
                        self.N_ukg[u][k-1,0] = self.N_ukg[u][k-1,self.G]
                self.N_kwg[:,:,0] = self.N_kwg[:,:,self.G]              

    def gibbsSampling(self, theta_prev):
        logweights = np.zeros(self.K)
        weights = np.zeros(self.K)
        for g in range(1,self.G+1):
            for u in self.users:
                self.N_ukg[u][:,g] = self.N_ukg[u][:,g-1]
                self.z_uag[u][:,g] = self.z_uag[u][:,g-1] 
                for k in range(1,self.K+1):
                    self.A_ukg[u][k-1][g] = deepcopy(self.A_ukg[u][k-1][g-1])
            self.N_kwg[:,:,g] = deepcopy(self.N_kwg[:,:,g-1])
            
            for u in self.users:
                # Sample topics of (u,a)
                for a in range(self.A[u]):             
                    k_ua = self.z_uag[u][a,g]
                    self.N_ukg[u][k_ua-1,g] -= 1
                    self.N_kwg[k_ua-1,:,g] -= self.TF_uaw[u][a]
                    self.A_ukg[u][k_ua-1][g].remove(a)
                    for k in range(1,self.K+1):
                        part1 = np.log(self.N_ukg[u][k-1,g] + self.alpha[k-1])
                        part2 = 0
                        part3 = 0
                        part4 = self.loglik_ua(self.x_ukg[u][k-1][g-1], theta_prev, u, a)
                        for w in self.TF_uaw[u][a,:].nonzero()[1]:             # (u,a)에서 frequency가 zero가 아닌 word만 고려
                            for b in range(1,self.TF_uaw[u][a,w]+1):
                                part2 += np.log(self.N_kwg[k-1,w,g] + self.gamma[w] + b - 1)
                        for c in range(1, np.sum(self.TF_uaw[u][a,:])+1):
                            part3 += np.log(np.sum(self.N_kwg[k-1,:,g]) + np.sum(self.gamma) + c - 1)
                        logweights[k-1] = part1 + part2 - part3 + part4
                    weights = np.exp(logweights - np.max(logweights))
                    k_ua_new = np.random.choice(list(range(1,self.K+1)), p=weights/np.sum(weights))
                    self.z_uag[u][a,g] = deepcopy(k_ua_new)
                    self.N_ukg[u][k_ua_new-1,g] += 1
                    self.N_kwg[k_ua_new-1,:,g] += self.TF_uaw[u][a]
                    self.A_ukg[u][k_ua_new-1][g].append(a)
                    
                # Sample topical expertise levels of (u,k)
                for k in range(1,self.K+1):
                    self.x_ukg[u][k-1,g] = self.ars_(theta_prev, u, k, g)

    def Mstep_(self, theta_prev, fun, max_it=1000):       
        it = 0
        bnds = [(None, None), (None, None), (None, None), (0, None)]  
        while it < max_it :
            if it == 0 : 
                new_x0 = deepcopy(theta_prev)
            else : 
                for idx in range(self.n_pa):
                    if idx < 3:
                        new_x0[idx] = theta_prev[idx] + np.random.rand() - 0.5
                    else:
                        new_x0[idx] = theta_prev[idx] * (np.random.rand() + 0.5)
            res = minimize(fun=self.mf(fun), x0=new_x0, method=self.opt_method, bounds=bnds) 
            succ = res.success
            it += 1
            if succ == True : break
            if it == max_it : print("NotConvergedMstep_", end=' ')
        theta = res.x                                                                                                    
        return theta

    def EM_(self, max_it=200, tol=0.0000001, num=10, disp=False):    
        theta_prev = self.theta0
        it = 0
        startTime = time.time()
        while it < max_it :
            self.theta_list.append(theta_prev)
            if disp == True : print(it, theta_prev, self.ofCount)
            it += 1
            Q = self.Estep_(theta_prev, it)
            theta = self.Mstep_(theta_prev, Q)
            if norm( theta - np.mean(np.array(self.theta_list[-num:]), axis=0) ) < tol:
                if disp == True : print("EM algorithm converged")
                break
            else:
                theta_prev = deepcopy(theta)
            if it == max_it : print("EM algorithm is not converged.")
        endTime = time.time()
        if disp == True : print("Running time : %.1f minutes." %((endTime - startTime)/60))
        self.theta_list.append(theta)
        self.theta = deepcopy(theta)
        self.Estep_(theta, 20000)
        self.thetaCov = self.theta_cov(theta)

    def infoMatrix(self, theta):                              
        part1 = - self.Q_hess(theta)(theta)
        part2 = 0
        Qder = self.Q_der(theta)(theta)
        part3 = np.outer(Qder, Qder)
        for g in range(self.G0+1, self.G+1):
            total = 0
            for u in self.users:
                for k in range(1, self.K+1):
                    total += self.loglik_der_ukg(self.x_ukg[u][k-1,g], theta, u, k, g)
            part2 += - np.outer(total, total) / (self.G - self.G0)
        return part1 + part2 + part3		

    def theta_cov(self, theta):                                 
        return inv(self.infoMatrix(theta))
    
    
if __name__ == '__main__':
    import pickle

    branch = input("Enter branch: ")
    
    # Load data
    f = open('data/pkls/' + branch + '_data.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    T_obs = 1425772800
    
    theta0 = [-5.70, 0.75, 0.66, 3.00]
    
    # Model Applications
    model = PTEM(data=data, theta0=theta0, K=5, G=250, G0=50, T_obs=T_obs, ars_a=-100, ars_b=20, max_df=0.25, min_df=0.01,
                 time_unit='day', min_ans=20, po_scale=1000)
    model.EM_(max_it=300, disp=True)    
    print("Estimated theta parameter : ", model.theta)
    print("Estimated standard error : ", np.sqrt(np.diag(model.thetaCov)))
    
    # Save 
    f = open('data/pkls/' + branch + '_result.pkl', 'wb')
    pickle.dump(model, f, protocol=0)
    f.close()
    
    
    
    













