import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
y = [7.5, 10.2, 12.8, 14.5, 15.6, 16.0, 15.8, 15.0, 13.5, 11.2]

x_squared = []
x_times_y = []
x_squared_times_y = []

for xi, yi in zip(x, y):
    xsq = xi**2
    xy = xi * yi
    x2y = xsq * yi

    x_squared.append(xsq)
    x_times_y.append(xy)
    x_squared_times_y.append(x2y)


print(f"Fertilizer kg(xᵢ) | Potatoes t(yᵢ) |{" "*3} xᵢ² {" "*2} |{" "*3} xᵢ·yᵢ{" "*2} | {" "*3}ᵢ²·yᵢ ")

for i in range(len(x)):

    print(f"{" "*6}x={x[i]:>3}{" "*6} |{" "*5}y={y[i]:>4}{" "*4} | x²={x_squared[i]:>6} | x·y={x_times_y[i]:>6.1f} | x²·y={x_squared_times_y[i]:>8.1f}")

x_np = np.array(x).reshape(-1, 1)
y_np = np.array(y)


poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_np)


# מודל רגרסיה פולינומית
model = LinearRegression()
model.fit(x_poly, y_np)


plt.scatter(x, y, color='red', label='Data')
x_line = np.linspace(1, 500, 100).reshape(-1, 1)
x_line_poly = poly.transform(x_line)
y_line = model.predict(x_line_poly)

plt.plot(x_line, y_line, color='blue', label='Potatoes Crop Line')
plt.xlabel('Fertilizer (kg)')
plt.ylabel("Potatoes (ton)" )
plt.title('Fertilizer vs. Potatoes Crop')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

