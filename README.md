# ML-Prac
These are some ML Notebooks I'm practicing as a revision, datasets will be alongside in the repo

Some notes I though would be useful

```python
from sklearn.base import clone
poly_scaler = Pipeline([    
	     ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
              ("std_scaler", StandardScaler())
              ])
              
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
penalty=None, learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
              sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
              y_val_predict = sgd_reg.predict(X_val_poly_scaled)
              val_error = mean_squared_error(y_val, y_val_predict)
              if val_error < minimum_val_error:
                  minimum_val_error = val_error
                  best_epoch = epoch
                  best_model = clone(sgd_reg)
```
Note that with warm_start=True, when the fit() method is called it continues training
where it left off, instead of restarting from scratch.
