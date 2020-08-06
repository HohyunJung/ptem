
from __future__ import division
from numpy.linalg import inv
from scipy.optimize import minimize
from copy import deepcopy
import numpy as np
from arspy.ars import adaptive_rejection_sampling
from sklearn.feature_extraction.text import CountVectorizer
from scipy import special
from scipy.special import xlogy, gammaln
import warnings
import time

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

########## PTEM: A Popularity-based Topical Expertise Model
########## "PTEM: A Popularity-based Topical Expertise Model for Community Question Answering"
########## Jung, H., Lee, J., Lee, N., and Kim, S.
# @Author: Hohyun, Jung (hhjung@kaist.ac.kr)

class PTEM():
    def __init__(self, data, theta0=[-4.0,0.1,0.0,1.0], K=10, descriptions=True, T0=0, time_unit='day', min_ans=20, max_df=0.25, min_df=0.01, vocabulary=None,
                 G=3, G0=0, alpha=None, gamma=0.01, gibbs_init=None, po_scale=1, ans_acc=False, po_normalize='time',  
                 ofBdry=200, ars_a=-100, ars_b=20, opt_method='SLSQP'):
        self.ofBdry = ofBdry              # prevents overflow error in exponential function
        self.po_scale = po_scale
        self.max_df = max_df
        self.min_df = min_df
        self.vocabulary = vocabulary
        self.ans_acc = ans_acc
        self.po_normalize = po_normalize
        self.gibbs_init = gibbs_init      # The previous results, Usage: gibbs_init = model_prev.z_uag, model_prev.x_ukg, model_prev.A_ukg, model_prev.N_ukg, model_prev.N_kwg
        self.time_unit = time_unit        # 'second', 'minute', 'hour', 'day' and any numbers
        self.td = self.td_()
        self.K = K                        # Number of topics
        self.G = G                        # Number of sampled fitness used in Gibbs sampling algorithm
        self.G0 = G0                      # Starting index of Gibbs sampling because initial samples are not follow the target distribution.
        self.min_ans = min_ans            # Consider users having at least min_ans number of answeres.
        self.data = data                  # data format {userId:[{Id,Score,CreationDate,Accepted,numAcceptSofar,Topic},...]}  {답변자: 답변들 리스트, 시간순서}
        self.theta0 = np.asarray(theta0)
        self.opt_method = opt_method      # Optimization method in E-step. See minimize function
        self.T0 = T0 / self.td
        self.ars_a, self.ars_b, self.ars_domain = ars_a, ars_b, (float("-inf"), float("inf"))           # ARS function's parameters
        self.users = self.users_()        # Selected M nodes if M is specified, otherwise it is all of the nodes
        self.n_pa = len(self.theta0)
        self.A = self.A_()                # Dictionary of the number of answers {userId: numberofanswers}
        self.vo = self.vo_()              # Dictionary of votes {userId: [votes of 1st answer, 2nd answer,... , A_uth answer]}
        self.po_raw, self.po = self.po_() # Dictionary of popularity measure {userId: [popularity of 1st answer, 2nd answer,... , A_uth answer]}
        self.at = self.at_()              # Dictionary of answer time {userId: [time of 1st answer, 2nd answer,... , A_uth answer]}
        self.A_ukg = {user: [[None]*(self.G+1) for k in range(1,self.K+1)] for user in self.users}     # Dictionary of set of answers for each topics, usage: A_ukg[u][k-1][g]
        self.W, self.words, self.startIdx, self.TF_uaw = self.words_()    # W: number of words, words: list of words, startIdx: see function words_(), TF_uaw: term frequency of (u,a)'s words, Usage: TF_uaw[u][a,w]
        self.alpha = self.alpha_(alpha)
        self.gamma = np.zeros(self.W) + gamma
        self.checkValidity = self.checkValidity_()
        self.z_uag = {user: np.zeros((self.A[user],self.G+1), dtype=int) for user in self.users}     # Dictionary of the gibbs samples of topic assignments, Usage: z_uag[u][a,g]
        self.x_ukg = {user: np.zeros((self.K,self.G+1), dtype=float) for user in self.users}     # Dictionary of the gibbs samples of topical expertise, Usage: x_ukg[u][k-1,g]
        self.N_ukg = {user: np.zeros((self.K, self.G+1), dtype=int) for user in self.users}    # Dictionary of the number of (u,k), Usage: N_ukg[u][k,g]
        self.N_kwg = np.zeros((self.K,self.W,self.G+1), dtype=int)                             # Dictionary of the number of (k,w), Usage: N_kwg[k,w,g]
        self.theta = np.asarray(theta0)
        self.thetaCov = None
        self.theta_list = list()           # List of beta's in EM algorithm
        self.ll_vote, self.ll_word = None, None
        self.aic = None
        self.ofCount = 0
        if descriptions == True :
            print('%d users, %d answers, %d vocabs.' %(len(self.users), np.sum(list(self.A.values())), self.W))
    
    def alpha_(self, alpha):
        if alpha == None:
            return np.zeros(self.K) + 50.0 / self.K
        else:
            return np.zeros(self.K) + alpha
        
    def clean_text(self, txt):
        STOPWORDS = stopwords.words('english')
        lemma = WordNetLemmatizer()
        tokenized_text = word_tokenize(txt.lower())
        cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
        lemmatized_text = [lemma.lemmatize(word) for word in cleaned_text]
        return " ".join(lemmatized_text)
    
    def words_(self):
        A_contents = list()
        startIdx = dict()                                                           #{userId: start index in A_contents}
        idx = 0
        for u in self.users:
            startIdx[u] = idx
            for a in range(self.A[u]):
                ans = self.data[u][a]
                A_contents.append(ans['Body'])
                idx += 1    
                
        A_contents_tokenized = []
        for txt in A_contents:
            A_contents_tokenized.append(self.clean_text(txt))
            
        tf_vectorizer_A = CountVectorizer(analyzer=str.split, max_df=self.max_df, min_df=self.min_df, vocabulary=self.vocabulary)  # '[a-zA-Z][a-zA-Z]+'
        TF_A = tf_vectorizer_A.fit_transform(A_contents_tokenized)
        W = TF_A.shape[1]
        words = tf_vectorizer_A.get_feature_names()
        TF_uaw = {user: deepcopy(TF_A[startIdx[user]:startIdx[user]+self.A[user]]) for user in self.users}      # Usage: TF_uaw[u][a,w]
        return W, words, startIdx, TF_uaw
            
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

    def po_(self):
        po_raw = {user: list() for user in self.users}
        po_normalized = {user: list() for user in self.users}      # y_u,t = y'_u,t / (t - T0)
        for u in self.users:
            po_u_sofar = 0
            po_raw[u].append(0.0)
            po_normalized[u].append(0.0)
            for a in range(1, self.A[u]):
                ans_prev = self.data[u][a-1]
                ans = self.data[u][a]
                v_ua_prev = max(ans_prev['Score'], 0)
                vstar_ua_prev = deepcopy(v_ua_prev)
                if self.ans_acc == True:
                    if ans_prev['Accepted'] == True:
                        vstar_ua_prev += 1.5
                po_u_sofar += vstar_ua_prev
                
                po_raw_value = po_u_sofar / self.po_scale
                po_raw[u].append(po_raw_value)
                
                if self.po_normalize == None:
                    po_normalized_value = po_raw_value
                elif self.po_normalize == 'time':
                    t_a = ans['CreationDate'] / self.td
                    po_normalized_value = po_raw_value / (t_a - self.T0)
                else:
                    raise NameError('Undefined variable: po_normalize')
                po_normalized[u].append(po_normalized_value)
        return po_raw, po_normalized
    
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
        mu = np.exp(vv)
        vote = self.vo[u][a]
        if vv < self.ofBdry :  
            value = special.loggamma(xi + vote) - special.loggamma(xi) + xi * np.log(xi) + vote * vv - (xi + vote) * np.log(xi + mu)
        else :
            self.ofCount += 1                 
            value = special.loggamma(xi + vote) - special.loggamma(xi) + xi * np.log(xi) + vote * vv - (xi + vote) * vv
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
        mu = np.exp(vv)
        vote = self.vo[u][a]
        if vv < self.ofBdry :   
            der_beta = (-(xi + vote) / (xi + mu) * mu + vote) * xx
            der_xi = special.polygamma(0, xi + vote) - special.polygamma(0, xi) - np.log(xi + mu) - (xi + vote) / (xi + mu) + np.log(xi) + 1
            der_value = np.hstack((der_beta, der_xi))
        else :
            der_beta = (-xi) * xx
            der_xi = special.polygamma(0, xi + vote) - special.polygamma(0, xi) - vv + np.log(xi) + 1
            der_value = np.hstack((der_beta, der_xi))
        return der_value

    def loglik_der_ukg(self, x, theta, u, k, g):
        total = 0
        for a in self.A_ukg[u][k-1][g]:
            total += self.loglik_der_ua(x, theta, u, a)
        return total

    def loglik_hess_ua(self, x, theta, u, a):            # hessian with respect to theta
        theta = np.asarray(theta)
        beta = theta[0:3]
        xi = theta[3]
        xx = np.array([1, x, self.po[u][a]])
        vv = beta.dot(xx)
        mu = np.exp(vv)
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
        self.init(it)                                  # initialization, g=0
        self.gibbsSampling(theta_prev)                  # Gibbs sampling, g=1,2,...,G
        return self.Q_(theta_prev)
    
    def init(self, it):
        if self.gibbs_init == None:
            if it == 1:                         # First EM iteration. Initialization of topics in random and zero topical expertise.
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
            else:                               # Initialization of topics and topical expertise levels by previous EM iteration's last topic assignments
                for u in self.users:
                    self.z_uag[u][:,0] = self.z_uag[u][:,self.G]
                    self.x_ukg[u][:,0] = self.x_ukg[u][:,self.G]
                    for k in range(1,self.K+1):
                        self.A_ukg[u][k-1][0] = deepcopy(self.A_ukg[u][k-1][self.G])
                        self.N_ukg[u][k-1,0] = self.N_ukg[u][k-1,self.G]
                self.N_kwg[:,:,0] = self.N_kwg[:,:,self.G]  
        else:
            if it == 1:                         # First EM iteration. Use initial gibbs conditions, Usage: gibbs_init = model_prev.z_uag, model_prev.x_ukg, model_prev.A_ukg, model_prev.N_ukg, model_prev.N_kwg
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
            else:                               # Initialization of topics and topical expertise levels by previous EM iteration's last topic assignments
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
            # Set initial numbers at iteration g
            for u in self.users:
                self.N_ukg[u][:,g] = self.N_ukg[u][:,g-1]
                self.z_uag[u][:,g] = self.z_uag[u][:,g-1] 
                for k in range(1,self.K+1):
                    self.A_ukg[u][k-1][g] = deepcopy(self.A_ukg[u][k-1][g-1])
            self.N_kwg[:,:,g] = deepcopy(self.N_kwg[:,:,g-1])
            
            # Algorithm of iteration g
            for u in self.users:
                # Sample topics of (u,a)
                for a in range(self.A[u]):             
                    # Construction of -(u,a) part
                    k_ua = self.z_uag[u][a,g]
                    self.N_ukg[u][k_ua-1,g] -= 1
                    self.N_kwg[k_ua-1,:,g] -= self.TF_uaw[u][a]
                    self.A_ukg[u][k_ua-1][g].remove(a)
                    # Draw topics of (u,a)
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
                    # Bring back
                    self.N_ukg[u][k_ua_new-1,g] += 1
                    self.N_kwg[k_ua_new-1,:,g] += self.TF_uaw[u][a]
                    self.A_ukg[u][k_ua_new-1][g].append(a)
                    
                # Sample topical expertise on (u,k)
                for k in range(1,self.K+1):
                    self.x_ukg[u][k-1,g] = self.ars_(theta_prev, u, k, g)

    def Mstep_(self, theta_prev, fun, max_it=1000):        # Input : theta_prev, E-step Q functions and its derivative, hessian   # Output : beta which maximize Q(theta)
        it = 0
        bnds = [(None, None), (None, None), (None, None), (0, None)]  # Constraint
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
            if disp == True : print(it, theta_prev)
            it += 1
            Q = self.Estep_(theta_prev, it)
            theta = self.Mstep_(theta_prev, Q)
            theta_prev = deepcopy(theta)
        endTime = time.time()
        if disp == True : print("Running time: %.1f minutes." %((endTime - startTime)/60))
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
    
    def model_selection_criteria(self):
        def multinomial_logpmf(x, n, p):
            return gammaln(n+1) + np.sum(xlogy(x, p) - gammaln(x+1), axis=-1)
        # Estimation of User Topical Distribution
        psi_ukg = {user: np.zeros((self.K, self.G+1), dtype=float) for user in self.users}
        for u in self.users:
            psi_ukg[u] += self.N_ukg[u]
            for g in range(self.G+1):
                psi_ukg[u][:,g] = psi_ukg[u][:,g] + self.alpha
                psi_ukg[u][:,g] = psi_ukg[u][:,g] / sum(psi_ukg[u][:,g])
        
        # Estimation of Topic word distribution
        phi_kwg = np.zeros((self.K,self.W,self.G+1), dtype=float)
        phi_kwg += self.N_kwg
        for k in range(1,self.K+1):
            for g in range(self.G+1):
                phi_kwg[k-1,:,g] = phi_kwg[k-1,:,g] + self.gamma
                phi_kwg[k-1,:,g] = phi_kwg[k-1,:,g] / sum(phi_kwg[k-1,:,g])
        
        phi_kw_hat = np.mean(phi_kwg[:,:,self.G0+1:self.G+1], axis=2)
        
        # User's topical expertise x_{u,k}
        x_uk_hat = {user: np.mean(self.x_ukg[user][:,self.G0+1:self.G+1], axis=1) for user in self.users}    
        
        # z_{u,a}'s distribution (numbers)
        z_ua_hat = {user: [np.argmax(np.array([list(self.z_uag[user][answer,self.G0+1:self.G+1]).count(k) for k in range(1,self.K+1)])) + 1 for answer in range(self.A[user])] for user in self.users}

        def vua_logpdf(x, theta, u, a):       
            theta = np.asarray(theta)
            beta = theta[0:3]
            xi = theta[3]
            xx = np.array([1, x, self.po[u][a]])
            vv = beta.dot(xx)
            mu = np.exp(vv)
            vote = self.vo[u][a]
            if vv < self.ofBdry :  
                value = special.loggamma(xi + vote) - special.loggamma(xi) - special.loggamma(vote + 1) + xi * np.log(xi) + vote * vv - (xi + vote) * np.log(xi + mu)
            else :             
                value = special.loggamma(xi + vote) - special.loggamma(xi) - special.loggamma(vote + 1) + xi * np.log(xi) + vote * vv - (xi + vote) * vv
            return value        

        # Vote part
        part1 = 0
        for u in self.users:
            for a in range(self.A[u]):
                topic = z_ua_hat[u][a]
                part1 += vua_logpdf(x_uk_hat[u][topic-1], self.theta, u, a)
        self.ll_vote = part1
        
        # word part
        part6 = 0
        for u in self.users:
            for a in range(self.A[u]):
                topic = z_ua_hat[u][a]
                tf = self.TF_uaw[u][a].toarray()
                part6 += multinomial_logpmf(tf, n=sum(sum(tf)), p=phi_kw_hat[topic-1])
        self.ll_word = part6
        
        U = len(self.users)
        sum_A = np.sum(list(self.A.values()))
        
        # AIC
        self.aic = -2 * (part1 + part6) + 2 * (4 + sum_A - U + (2 * U + self.W) * self.K)
        
    
if __name__ == '__main__':
    import pickle

    branch = 'philosophy'       # 'android', 'philosophy'
    
    # Load data
    f = open('data/pkls/' + branch + '_data.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    
    #T0 = 1241609310.977      # 'android': 1241609310.977
    T0 = 1301984146.710      # 'philosophy': 1301984146.710
    
    theta0 = [-0.00, 0.75, 0.20, 2.00]
    
    # Model Applications
    model = PTEM(data=data, theta0=theta0, K=15, min_ans=5, T0=T0)
    model.EM_(max_it=100, disp=True)    
    print("Estimated parameter: ", model.theta)
    
    # Save 
    f = open('data/pkls/' + branch + '_result.pkl', 'wb')
    pickle.dump(model, f, protocol=0)
    f.close()
    
    
   
