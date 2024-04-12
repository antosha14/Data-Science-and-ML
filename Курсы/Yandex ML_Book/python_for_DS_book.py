import numpy as np

# После переменной в интерактивном воркбуке можно поставить вопрос, чтобы получить более подробную информацию о ней
# Метод .format(var1, var2) на стринг вставит

ones_matrix = np.ones((1, 5))
print(ones_matrix)
ones_submatrix_view = ones_matrix[::2, ::2]  # creates a view, not copy
ones_matrix[::2, ::2] = np.zeros((3, 3))
ones_submatrix_view
