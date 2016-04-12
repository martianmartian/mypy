x, y = make_regression(n_samples=5, n_features=1, n_informative=1,random_state=0, noise=35)
# plt.plot(X[:],y,'o')
# plt.show()
m = x.shape[0]
alpha = 0.01
theta = np.random.random(2)
ones = np.ones(X.shape[0])
X = np.c_[ones,x]
J = []

for i in range(100):
    hypothesis = np.dot(X,theta)
    diff = hypothesis - y
    J.append(np.sum(diff**2)/(2*m))
    gradients = np.dot(diff,X)/m
#     print theta
    theta = theta - alpha*gradients
#     print theta

plt.plot(J,'o')
plt.show()