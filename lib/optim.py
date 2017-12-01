import numpy as np


""" Super Class """
class Optimizer(object):
	""" 
	This is a template for implementing the classes of optimizers
	"""
	def __init__(self, net, lr=1e-4):
		self.net = net  # the model
		self.lr = lr    # learning rate

	""" Make a step and update all parameters """
	def step(self):
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				pass


""" Classes """
class SGD(Optimizer):
	""" Some comments """
	def __init__(self, net, lr=1e-4):
		self.net = net
		self.lr = lr

	def step(self):
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dv = layer.grads[n]
				layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
	def __init__(self, net, lr=1e-4, momentum=0.0):
		self.net = net
		self.lr = lr
		self.momentum = momentum
		self.velocity = {}

	def step(self):
		#############################################################################
		Implement the SGD + Momentum                                        #
		#############################################################################	
	
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dv = layer.grads[n]
				if(n not in self.velocity):
					self.velocity = {n: np.zeros(dv.shape)}
				self.velocity[n] = self.momentum * self.velocity[n] - self.lr * dv
				layer.params[n]  += self.velocity[n]



class RMSProp(Optimizer):
	def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
		self.net = net
		self.lr = lr
		self.decay = decay
		self.eps = eps
		self.cache = {}  # decaying average of past squared gradients

	def step(self):
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dv = layer.grads[n]
				self.cache[n] = self.decay*self.cache.get(n,np.zeros_like(v)) + (1- self.decay)*np.square(dv)
				layer.params[n] -= (self.lr*dv)/(np.sqrt(self.cache[n] + self.eps))				
				#dv = layer.grads[n]
				#if(n not in self.cache):
				#	self.cache = {n: np.zeros(dv.shape)}
				#self.cache[n] = self.decay * self.cache[n] + (1-self.decay) * dv**2
				#layer.params[n]  += -self.lr * dv / (np.sqrt(self.cache[n]) + self.eps)



class Adam(Optimizer):
	def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
		self.net = net
		self.lr = lr
		self.beta1, self.beta2 = beta1, beta2
		self.eps = eps
		self.mt = {}
		self.vt = {}
		self.t = t

	def step(self):
		self.t+=1
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dv = layer.grads[n]
				self.mt[n] = self.beta1*self.mt.get(n,np.zeros_like(v)) + (1- self.beta1)*dv
				self.vt[n] = self.beta2*self.vt.get(n,np.zeros_like(v))+ (1- self.beta2)*np.square(dv)
				vt_hat = self.vt[n]/(1- self.beta2**self.t)
				mt_hat = self.mt[n]/(1 - self.beta1**self.t)
				layer.params[n] -= (self.lr*mt_hat/(np.sqrt(vt_hat) + self.eps))

